from data.seg_dataset import SegmentationDataset
from utils.distributed_gpu import prepare, run_parallel, setup, cleanup

def distributed_test(rank, world_size):
    try:
        print(f"Rank {rank} starting")
        setup(rank, world_size)

        dataset = SegmentationDataset(root_dir='data/sis_data', mode='train')
        dataloader = prepare(dataset, rank, world_size, batch_size=4)

        for i, batch in enumerate(dataloader):
            print(f"Rank {rank}, Batch {i}:")

        cleanup()
    except Exception as e:
        print(f"Rank {rank} crashed:", e)
        raise

def test_dataset():
    world_size = 4
    run_parallel(distributed_test, world_size)

if __name__ == "__main__":
    test_dataset()
