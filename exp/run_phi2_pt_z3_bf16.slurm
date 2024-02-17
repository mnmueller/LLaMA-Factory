#!/bin/bash
#SBATCH --job-name=phi_2_wiki_z3_bf16      # name
#SBATCH --nodes=1                   # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=128          # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 80:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./runs/%x-%j.out           # output file name
#SBATCH --error=./runs/%x-%j.err           # output file name
ref_dir=$PWD

source ~/miniconda3/etc/profile.d/conda.sh

conda activate mixtralLF

echo $CONDA_DEFAULT_ENV

export HF_TOKEN=$(< ~/.ssh/hf_token)
export HF_DATASETS_IN_MEMORY_MAX_SIZE=274877906944

export WANDB_PROJECT="bg_mixtral"

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export PYTHON_PATH=$PYTHON_PATH:$ref_dir
echo $CUDA_HOME

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

export NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=0
export NCCL_IB_TC=41
export NCCL_IB_QPS_PER_CONNECTION=4
export OMP_NUM_THREADS=128

huggingface-cli login --token $HF_TOKEN

srun --jobid "$SLURM_JOBID" \
    bash -c 'torchrun --nproc_per_node "$GPUS_PER_NODE" --nnodes "$SLURM_NNODES" --node_rank "$SLURM_PROCID" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    LF_train.py \
    --deepspeed zero3_bf16.json \
    --stage pt \
    --do_train \
    --model_name_or_path microsoft/phi-2 \
    --dataset wiki_demo \
    --finetuning_type full \
    --output_dir ./runs/debug_$SLURM_JOBID \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_step 1 \
    --learning_rate 1e-5 \
    --warmup_steps 100 \
    --num_train_epochs 1.0 \
    --report_to wandb \
    --bf16 \
    --flash_attn \
    --dataset_dir ../data \
    --cutoff_len 2048'
