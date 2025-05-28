import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count, Pool

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import packed_dataset as packed_dataset
# Import huggingface tokenizers
from transformers import AutoTokenizer

# Filename for SlimPajama
sets = {
    "train": "train/*",
    "valid": "valid/*",
}


def prepare_full(
        source_path: Path,
        tokenizer_path: Path,
        destination_path: Path,
        chunk_size: int,
        shortname: str = "the_pile_unzip",
        split: str = "train",
        filenames_subset: List[str] = None,
        process_id: int = 0,
        use_huggingface: bool = True,
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer based on the parameter
    if use_huggingface:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        bos_id = tokenizer.bos_token_id
        vocab_size = len(tokenizer.get_vocab())
    else:
        raise NotImplementedError()

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset

    if not filenames:
        raise RuntimeError(
            f"No files matching {sets[split]} found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_{shortname}_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=bos_id,
        dtype="auto",
        vocab_size=vocab_size,
    )

    for filepath in filenames:
        print(f"Processing {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                # Encode with the appropriate tokenizer
                if use_huggingface:
                    # Hugging Face tokenizers return a dictionary, we want the input_ids
                    text_ids = tokenizer.encode(text, add_special_tokens=False)
                else:
                    text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))


def process_file(args):
    source_path, tokenizer_path, destination_path, chunk_size, subset_name, split, filename, index, use_huggingface = args
    prepare_full(source_path, tokenizer_path, destination_path, chunk_size, subset_name, split, [filename], index,
                 use_huggingface)


def prepare(
        source_path: Path = Path("sail/regmix-data"),
        tokenizer_path: Path = Path("tokenizer/gptneox"),
        destination_path: Path = Path("data/regmix_data"),
        short_name: str = "ind",
        chunk_size: int = 2049 * 256,
        split: str = "train",
        percentage: float = 1.0,
        use_huggingface: bool = True,
) -> None:
    import time

    filenames = glob.glob(os.path.join(source_path, sets[split]), recursive=True)
    filenames = filenames[:int(len(filenames) * percentage)]

    start_time = time.time()

    tasks = []
    for i, filename in enumerate(filenames):
        subset_name = short_name + "_" + filename.split("/")[-1].split(".")[0]
        tasks.append((source_path, tokenizer_path, destination_path, chunk_size, subset_name, split, filename, i,
                      use_huggingface))

    with Pool(processes=min(cpu_count(), len(filenames))) as pool:
        pool.map(process_file, tasks)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)