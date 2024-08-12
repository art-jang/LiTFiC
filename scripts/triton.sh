#!/bin/bash

#SBATCH --job-name=llama                   # Job name
#SBATCH --mail-type=END,FAIL                # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=haran@robots.ox.ac.uk    # Where to send mail
#SBATCH --nodes=1                          # Node count
#SBATCH --ntasks-per-node=2                         # Total number of tasks across all nodes
#SBATCH --cpus-per-task=10                   # Number of CPU cores per task
#SBATCH --mem-per-cpu=24G                         # Job memory request
#SBATCH --time=20:00:00                     # Time limit hrs:min:sec
#SBATCH --partition=ddp-2way                   # Partition (compute (default) / gpu)
#SBATCH --gres=gpu:2                       # Requesting 4 GPUs
#SBATCH --constraint=a6000      # Request nodes with specific features


. ~/miniconda3/etc/profile.d/conda.sh
conda activate slt

export HYDRA_FULL_ERROR=1 
export CUDA_LAUNCH_BLOCKING=1

python src/train.py task_name=debug experiment=llama3_haran paths=haran_triton \
    data.train_data_fraction=0.0001 \
    data.dataset_config.load_features=False \
    data.dataset_config.max_previous_sentences=0 \
    data.dataset_config.sub_sample_pct=0.25 \
    model.net.llm_config.sub_sub=True \
    model.net.llm_config.use_pl_probs=False\
    model.net.llm_config.oracle=True


