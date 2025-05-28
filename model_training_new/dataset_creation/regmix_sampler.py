import glob
import random
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
import torch
from torch.utils.data import DataLoader

# Import the original classes from RegMix
# In your implementation, you would import these from the RegMix codebase
from packed_dataset import PackedDataset, CombinedDataset


def load_mixture_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load dataset mixture configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the parsed configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def create_train_dataloader(
        config_path: Union[str, Path],
        data_dir: Path,
        batch_size: int,
        block_size: int,
        shuffle: bool = True,
        seed: int = 12345,
        split: str = "train",
        world_size: int = 1,
        global_rank: int = 0
) -> DataLoader:
    """
    Create a dataloader based on a mixture configuration file.
    This function replicates the dataset creation logic from the original RegMix code.

    Args:
        config_path: Path to the YAML configuration file
        data_dir: Directory containing the data files
        batch_size: Batch size for the dataloader
        block_size: Size of data blocks to process
        shuffle: Whether to shuffle the data
        seed: Random seed
        split: Which split to use ("train" or "valid")
        world_size: Total number of processes (for distributed training)
        global_rank: Rank of the current process (for distributed training)

    Returns:
        DataLoader with the combined dataset
    """
    # Load configuration
    config = load_mixture_config(config_path)

    # Ensure the config has the requested split
    if split not in config:
        raise ValueError(f"Config file does not contain '{split}' section")

    # Extract dataset configurations
    data_config = []
    for name, weight in config[split].items():
        data_config.append((name, float(weight)))

    # Create datasets
    datasets = []

    # Check for validity of datasets (similar to the original code)
    for idx in range(len(data_config) - 1, -1, -1):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        if len(filenames) < world_size:
            print(f"Skip dataset {prefix} - not enough files for all processes")
            del data_config[idx]
            continue

    # Create dataset for each valid configuration
    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
        random.seed(seed)
        random.shuffle(filenames)
        print(f"Creating dataset for {prefix}")

        dataset = PackedDataset(
            filenames=filenames,
            n_chunks=1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + global_rank,
            num_processes=world_size,
            process_rank=global_rank
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure your data files exist."
        )

    # Normalize the weights
    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    normalized_weights = [w / sum_weights for w in weights]

    # Create the combined dataset
    combined_dataset = CombinedDataset(
        datasets=datasets,
        seed=seed,
        weights=normalized_weights
    )

    # Create and return the dataloader
    return DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


def create_val_dataloader(
        config_path: Union[str, Path],
        data_dir: Path,
        batch_size: int,
        block_size: int,
        shuffle: bool = False,
        seed: int = 12345,
        world_size: int = 1,
        global_rank: int = 0
) -> List[DataLoader]:
    """
    Create validation dataloaders based on a mixture configuration file.
    This function replicates the validation dataset creation logic from the original RegMix code.

    Args:
        config_path: Path to the YAML configuration file
        data_dir: Directory containing the data files
        batch_size: Batch size for the dataloader
        block_size: Size of data blocks to process
        shuffle: Whether to shuffle the data
        seed: Random seed
        world_size: Total number of processes (for distributed training)
        global_rank: Rank of the current process (for distributed training)

    Returns:
        List of DataLoaders with the combined validation datasets
    """
    # Load configuration
    config = load_mixture_config(config_path)

    # Ensure the config has the validation split
    if "valid" not in config:
        print("No validation configuration found in config file")
        return None

    # Extract validation dataset configurations
    # In the original code, val_data_config is a list of lists of tuples
    val_data_config = []
    for name, weight in config["valid"].items():
        val_data_config.append([(name, float(weight))])

    val_data_loaders = []

    # Check for validity of validation datasets
    for idx in range(len(val_data_config) - 1, -1, -1):
        data_config = val_data_config[idx]
        delete_val_flag = False
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
            len_filenames = len(filenames)
            if len(filenames) < world_size:
                print(f"Skip validation dataset {prefix} - not enough files for all processes")
                delete_val_flag = True
                break
        if delete_val_flag:
            del val_data_config[idx]

    # Create dataset for each valid validation configuration
    for data_config in val_data_config:
        datasets = []
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}-*")))
            random.seed(seed)
            random.shuffle(filenames)

            dataset = PackedDataset(
                filenames=filenames,
                n_chunks=1,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed + global_rank,
                num_processes=world_size,
                process_rank=global_rank
            )
            datasets.append(dataset)

        if not datasets:
            print(f"No validation data found at {data_dir}")
            continue

        # Normalize the weights
        weights = [weight for _, weight in data_config]
        sum_weights = sum(weights)
        normalized_weights = [w / sum_weights for w in weights]

        # Create flag to check if all datasets have files
        valid_datasets = True
        for dataset in datasets:
            if len(dataset._filenames) == 0:
                valid_datasets = False
                break

        if valid_datasets:
            # Create the combined dataset
            combined_dataset = CombinedDataset(
                datasets=datasets,
                seed=seed,
                weights=normalized_weights
            )

            # Create and add the dataloader
            val_data_loaders.append(
                DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=True
                )
            )
            print(f"Created validation dataset for {data_config}")
        else:
            print(f"Issues with validation dataset {data_config} - skipping")

    return val_data_loaders


def create_dataloaders(
        config_path: Union[str, Path],
        batch_size: int,
        block_size: int,
        train_data_dir: Path,
        val_data_dir: Optional[Path] = None,
        seed: int = 12345,
        world_size: int = 1,
        global_rank: int = 0
) -> Tuple[DataLoader, Optional[List[DataLoader]]]:
    """
    Create both training and validation dataloaders based on a mixture configuration file.
    This function replicates the main dataloader creation logic from the original RegMix code.

    Args:
        config_path: Path to the YAML configuration file
        batch_size: Batch size for the dataloader
        block_size: Size of data blocks to process
        train_data_dir: Directory containing training data files
        val_data_dir: Directory containing validation data files (optional)
        seed: Random seed
        world_size: Total number of processes (for distributed training)
        global_rank: Rank of the current process (for distributed training)

    Returns:
        Tuple of (train_dataloader, val_dataloaders)
    """
    # For training, increase block size by 1 to include the next token as the target
    effective_block_size = block_size + 1

    # Create training dataloader
    train_dataloader = create_train_dataloader(
        config_path=config_path,
        data_dir=train_data_dir,
        batch_size=batch_size,
        block_size=effective_block_size,
        shuffle=True,
        seed=seed,
        split="train",
        world_size=world_size,
        global_rank=global_rank
    )

    # Create validation dataloaders if validation directory is provided
    val_dataloaders = None
    if val_data_dir is not None:
        val_dataloaders = create_val_dataloader(
            config_path=config_path,
            data_dir=val_data_dir,
            batch_size=batch_size,
            block_size=effective_block_size,
            shuffle=False,
            seed=seed,
            world_size=world_size,
            global_rank=global_rank
        )

    return train_dataloader, val_dataloaders


def extract_model_config_from_yaml(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract model-related configuration parameters from the YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary of model configuration parameters
    """
    config = load_mixture_config(config_path)

    # Extract model-related parameters, excluding the train/valid dataset definitions
    model_config = {}
    for key, value in config.items():
        if key not in ["train", "valid"]:
            model_config[key] = value

    return model_config


# Example usage
if __name__ == "__main__":
    import os

    # Mock data directories
    train_dir = Path("../../model_training/dfm_data/train")
    val_dir = Path("../../model_training/dfm_data/valid")
    config_path = Path("../../mixture_config/config_1m/n1.yaml")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Extract model configuration
    config = load_mixture_config(config_path)
    print(f"Model configuration extracted from YAML:")
    for key, value in config.items():
        print(f"  {key}: {value}")