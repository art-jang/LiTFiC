#!/bin/bash
#SBATCH --job-name=llama            # job name
#SBATCH --account=vvh@a100                # project code
#SBATCH -C a100
#SBATC -C v100-32g                       # to choose nodes with 32G GPU memory (i.e. gpu_p1)
#SBATCH --ntasks-per-node=4             # number of MPI tasks per node
#SBATCH --nodes=2                      # number of nodes
#SBATCH --gres=gpu:4                     # number of GPUs per node                         # number of nodes
#SBATCH --qos=qos_gpu-t3                  # (20h) jobs
#SBATCH --cpus-per-task=15                 # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATC --distribution=block:block        # we pin the tasks on contiguous cores
#SBATCH --time=15:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.out # output file name
#SBATCH --error=/gpfswork/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.err  # error file name

set -x # echo launched commands

module purge
# module load pytorch-gpu/py3/1.4.0 

. ${WORK}/miniconda3/etc/profile.d/conda.sh
conda activate slt

# cd ${SLURM_SUBMIT_DIR}
# cd "${WORK}/hrn/cslr2_t"

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline

srun python src/train.py task_name=llama_pl_lora experiment=llama3_haran paths=haran
# srun python src/eval.py task_name=llama_pl01 experiment=llama3_haran paths=haran