export DATASET_SHORT_NAME="dfm"

# python download_dataset.py --dataset_name sail/regmix-data-sample
# WARNING: you can choose to download the full dataset (around 1TB) by running the following command
# python download_data.py --dataset_name sail/regmix-data

python prepare_file_domain.py --source_path ../../model_training/dfm_data --tokenizer_path HuggingFaceTB/SmolLM2-135M --destination_path ../dfm_dataset_preproc/train --short_name $DATASET_SHORT_NAME --split train --use_huggingface True

# 131136 = 2049 * 64, which means the chunk size is relatively smaller due to the size of the validation set, especially for low-resource domains
python prepare_file_domain.py --source_path ../../model_training/dfm_data --tokenizer_path HuggingFaceTB/SmolLM2-135M --destination_path ../dfm_dataset_preproc/valid --short_name $DATASET_SHORT_NAME --split valid --chunk_size 131136 --use_huggingface True