import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import lpips
from pytorch_msssim import ssim
import math

# ==========================================
# 1. Setup & Arguments
# ==========================================

parser = argparse.ArgumentParser(description='Compare images from two folders (GT vs Output)')
parser.add_argument('--gt_path', type=str, required=True, help='Path to Ground Truth images')
parser.add_argument('--pred_path', type=str, required=True, help='Path to Predicted/Output images')
parser.add_argument('--output_dir', type=str, default='comparison_results', help='Where to save graphs/logs')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# ==========================================
# 2. Metrics & Helpers
# ==========================================

# Initialize LPIPS (Learned Perceptual Image Patch Similarity)
# net='alex' is standard and faster. Use 'vgg' for potentially higher accuracy but slower.
lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

def calculate_psnr(img1, img2):
    """Calculates PSNR between two tensors in range [0, 1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def get_image_list(path):
    """Returns a set of filenames from a directory (extensions ignored for matching logic if needed)."""
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    return sorted([f for f in os.listdir(path) if f.lower().endswith(valid_exts)])

# Transform: Resize is optional, but ConvertToTensor is mandatory.
# We assume images are loaded as [0, 1] for PSNR/SSIM.
# We will scale them to [-1, 1] specifically for LPIPS.
to_tensor = transforms.Compose([
    transforms.ToTensor() # Converts (H, W, C) [0, 255] -> (C, H, W) [0.0, 1.0]
])

# ==========================================
# 3. Main Loop
# ==========================================

os.makedirs(args.output_dir, exist_ok=True)

gt_files = get_image_list(args.gt_path)
analysis_results = []

print(f"Found {len(gt_files)} images in GT folder. Starting comparison...")

for filename in gt_files:
    gt_full_path = os.path.join(args.gt_path, filename)
    pred_full_path = os.path.join(args.pred_path, filename)

    # Check if corresponding prediction exists
    if not os.path.exists(pred_full_path):
        print(f"Warning: {filename} not found in prediction path. Skipping.")
        continue

    try:
        # Load Images
        img_gt_pil = Image.open(gt_full_path).convert('RGB')
        img_pred_pil = Image.open(pred_full_path).convert('RGB')

        # Ensure sizes match (resize pred to gt if necessary, or crop)
        if img_gt_pil.size != img_pred_pil.size:
            img_pred_pil = img_pred_pil.resize(img_gt_pil.size, Image.BICUBIC)

        # Convert to Tensor [0, 1]
        t_gt = to_tensor(img_gt_pil).unsqueeze(0).to(device)
        t_pred = to_tensor(img_pred_pil).unsqueeze(0).to(device)

        # --- Calculate Metrics ---

        # 1. PSNR (Input: [0, 1])
        val_psnr = calculate_psnr(t_gt, t_pred)

        # 2. SSIM (Input: [0, 1])
        # data_range=1.0 because tensors are 0-1
        val_ssim = ssim(t_pred, t_gt, data_range=1.0, size_average=True).item()

        # 3. LPIPS (Input: [-1, 1])
        # We transform [0, 1] -> [-1, 1] by (x * 2) - 1
        t_gt_norm = t_gt * 2 - 1
        t_pred_norm = t_pred * 2 - 1
        val_lpips = lpips_fn(t_pred_norm, t_gt_norm).mean().item()

        # 4. Difficulty / Complexity Metric
        # Since we don't have "Frame 1" vs "Frame 3" motion, we calculate
        # the standard deviation of the GT image as a proxy for "Image Complexity".
        # Higher std dev = more texture/edges = usually harder to generate perfectly.
        complexity = torch.std(t_gt).item()

        analysis_results.append({
            "filename": filename,
            "difficulty": complexity,
            "psnr": val_psnr,
            "ssim": val_ssim,
            "lpips": val_lpips
        })
        
        # Optional: Print progress every 10 images
        if len(analysis_results) % 10 == 0:
            print(f"Processed {len(analysis_results)} images...", end='\r')

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print(f"\nProcessing complete. Total pairs analyzed: {len(analysis_results)}")

if len(analysis_results) == 0:
    print("No matching images found. Exiting.")
    exit()

# ==========================================
# 4. Global Stats & Bucketing
# ==========================================

avg_psnr = sum(r['psnr'] for r in analysis_results) / len(analysis_results)
avg_ssim = sum(r['ssim'] for r in analysis_results) / len(analysis_results)
avg_lpips = sum(r['lpips'] for r in analysis_results) / len(analysis_results)

print(f"\n--- Global Results ---")
print(f"Average PSNR:  {avg_psnr:.2f} dB")
print(f"Average SSIM:  {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")

# Logic for Low/Med/High buckets based on Difficulty (Image Complexity)
diff_scores = [r['difficulty'] for r in analysis_results]
low_thresh = np.percentile(diff_scores, 33)
high_thresh = np.percentile(diff_scores, 66)

buckets = {
    "Low":    {"data": [], "range": f"< {low_thresh:.4f}"},
    "Medium": {"data": [], "range": f"{low_thresh:.4f} - {high_thresh:.4f}"},
    "High":   {"data": [], "range": f"> {high_thresh:.4f}"}
}

for r in analysis_results:
    d = r['difficulty']
    if d <= low_thresh:
        buckets["Low"]["data"].append(r)
    elif d <= high_thresh:
        buckets["Medium"]["data"].append(r)
    else:
        buckets["High"]["data"].append(r)

print("\n--- Performance vs. Image Complexity (Variance) ---")
print(f"{'Category':<10} | {'Range (StdDev)':<18} | {'Count':<6} | {'PSNR':<8} | {'SSIM':<8} | {'LPIPS':<8}")
print("-" * 75)

for cat in ["Low", "Medium", "High"]:
    data = buckets[cat]["data"]
    if len(data) == 0:
        continue
    
    b_psnr = sum(x['psnr'] for x in data) / len(data)
    b_ssim = sum(x['ssim'] for x in data) / len(data)
    b_lpips = sum(x['lpips'] for x in data) / len(data)
    
    print(f"{cat:<10} | {buckets[cat]['range']:<18} | {len(data):<6} | {b_psnr:<8.2f} | {b_ssim:<8.4f} | {b_lpips:<8.4f}")

# ==========================================
# 5. Plotting
# ==========================================
print("\nGenerating graph...")

try:
    all_diff = [r['difficulty'] for r in analysis_results]
    all_psnr = [r['psnr'] for r in analysis_results]
    
    plt.figure(figsize=(10, 6))

    # Scatter plot
    plt.scatter(all_diff, all_psnr, alpha=0.5, c='blue', s=15, label='Samples')

    # Vertical lines for thresholds
    plt.axvline(x=low_thresh, color='green', linestyle='--', alpha=0.7, label='Low Complexity Boundary')
    plt.axvline(x=high_thresh, color='orange', linestyle='--', alpha=0.7, label='High Complexity Boundary')

    # Trendline
    if len(all_diff) > 1:
        z = np.polyfit(all_diff, all_psnr, 1)
        p = np.poly1d(z)
        sorted_diff = sorted(all_diff)
        plt.plot(sorted_diff, p(sorted_diff), "r--", linewidth=2, label="Trend")

    plt.title(f"Reconstruction Quality (PSNR) vs. Image Complexity\n(Proxy for Difficulty: Pixel Std Dev)")
    plt.xlabel("Image Complexity (Pixel Standard Deviation)")
    plt.ylabel("PSNR (dB)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plot_path = os.path.join(args.output_dir, "metrics_analysis.png")
    plt.savefig(plot_path)
    print(f"Graph saved to: {plot_path}")

except Exception as e:
    print(f"Error generating plot: {e}")
