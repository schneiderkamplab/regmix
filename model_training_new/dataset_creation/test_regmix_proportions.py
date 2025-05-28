import yaml
import torch
from pathlib import Path
from typing import Union
from tqdm import tqdm
from packed_dataset import CombinedDataset
from regmix_sampler import create_dataloaders


# Example usage for real packed dataset
def test_regmix_dataloader_proportions(
        dataloader: torch.utils.data.DataLoader,
        config_path: Union[str, Path],
        split: str = "train",
        num_samples: int = 1000
):
    """
    Test if a RegMix dataloader's sampling proportions match the specified mixture proportions.

    This function is designed to work with the actual PackedDataset and CombinedDataset
    from the RegMix codebase.

    Args:
        dataloader: The RegMix dataloader to test
        config_path: Path to the YAML configuration file
        split: Which split to test ("train" or "valid")
        num_samples: Number of samples to use for testing

    Returns:
        Dictionary mapping dataset names to their observed sampling proportions
    """
    # For a real PackedDataset, we need a different approach to identify the source dataset
    # One approach is to examine a characteristic of the data that's unique to each dataset

    # Load the expected proportions from the config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Testing sampling proportions with approximately {num_samples} batches...")
    progress_bar = tqdm(total=num_samples)

    # Sample batches
    for i, batch in enumerate(dataloader):
        if i >= num_samples:
            break

        progress_bar.update(1)

    progress_bar.close()

    # Ensure the dataloader dataset is a CombinedDataset and get dataset lengths and proportions
    dataset_lengths = []
    if isinstance(dataloader.dataset, CombinedDataset):
        # Log lengths of datasets
        dataset_lengths = dataloader.dataset.log_lengths("./dataset_lengths")

    # Print comparison
    print("\nSampling Proportion Test Results:")
    print("-" * 80)
    print(f"{'Dataset id':<30} {'Expected':<10} {'Observed':<10} {'Difference':<10}")
    print("-" * 80)

    for length_tuple in dataset_lengths:
        dataset_id = length_tuple[0]
        expected = length_tuple[1]
        observed = length_tuple[3]
        diff = observed - expected
        print(f"{dataset_id:<30} {expected:.4f} {observed:.4f} {diff:+.4f}")

    print("-" * 80)

if __name__ == "__main__":

    train_dir = Path("../dfm_dataset_preproc/train")
    val_dir = Path("../dfm_dataset_preproc/valid")
    config_path = "../../mixture_config/config_1m/n1.yaml"

    # Create your RegMix dataloader
    train_loader, _ = create_dataloaders(
        config_path=config_path,
        batch_size=16,
        block_size=1024,
        train_data_dir=train_dir,
        seed=42
    )

    # Test the sampling proportions
    test_regmix_dataloader_proportions(
        dataloader=train_loader,
        config_path=config_path,
        split='train',
        num_samples=1000
    )
