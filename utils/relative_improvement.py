
def relative_improvement(values, eps: float = 1e-8) -> float:
    """
    Computes a *stability-aware* relative improvement over a sequence.

    The metric considers **all values**, not just first/last:

    1. Compute relative improvement from the mean of the first half
       to the mean of the second half
    2. Penalize non-stable sequences using normalized variance

    Returns:
        RI = improvement * stability

    Where:
        improvement = (mean_old - mean_new) / max(|mean_old|, eps)
        stability   = 1 / (1 + coeff_variation)
    """
    n = len(values)
    values = list(values)
    if n < 2:
        return 0.0

    # Split sequence
    mid = n // 2
    first = values[:mid]
    second = values[mid:]

    mean_old = sum(first) / len(first)
    mean_new = sum(second) / len(second)

    improvement = (mean_old - mean_new) / max(abs(mean_old), eps)

    # Stability: coefficient of variation over full window
    mean_all = sum(values) / n
    var = sum((v - mean_all) ** 2 for v in values) / n
    std = var ** 0.5
    coeff_variation = std / max(abs(mean_all), eps)

    stability = 1.0 / (1.0 + coeff_variation)

    return abs(improvement * stability)

