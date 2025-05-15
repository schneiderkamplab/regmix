export WANDB_PROJECT=$(cat ../.env | grep WANDB_PROJECT | cut -d '=' -f2)
export WANDB_ENTITY=$(cat ../.env | grep WANDB_ENTITY | cut -d '=' -f2)
export WANDB_API_KEY=$(cat ../.env | grep WANDB_API_KEY | cut -d '=' -f2)

export MODEL_NAME=tinyllama_1M_n$1
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPUS=1
# you can specify the config index here or pass it as an argument
export CONFIG_INDEX=$1

# Force output to be displayed
export PYTHONUNBUFFERED=1

fabric run pretrain/tinyllama.py \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPUS \
    --devices $NUMBER_OF_GPUS \
    --train_data_dir dfm_data/train \
    --val_data_dir dfm_data/valid \
    --data_yaml_file ../mixture_config/config_1m/n$CONFIG_INDEX.yaml \
    --out_name $MODEL_NAME \
    --resume True

# 2>&1 | tee training_log.txt
