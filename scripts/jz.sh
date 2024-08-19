#!/bin/bash
#SBATCH --job-name=llama31_standard        # job name
#SBATCH --account=ewl@a100                # project code
#SBATCH -C a100
#SBATC -C v100-32g                       # to choose nodes with 32G GPU memory (i.e. gpu_p1)
#SBATCH --ntasks-per-node=4             # number of MPI tasks per node
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gres=gpu:4                     # number of GPUs per node                         # number of nodes
#SBATCH --qos=qos_gpu-t3                  # (20h) jobs
#SBATCH --cpus-per-task=15                 # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATC --distribution=block:block        # we pin the tasks on contiguous cores
#SBATCH --time=20:00:00                 # maximum execution time (HH:MM:SS)
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

srun python src/train.py task_name=llama31_feat_nlora experiment=llama3_haran paths=haran \
    data.dataset_config.max_previous_sentences=0\
    data.dataset_config.sub_sample_pct=1.0\
    model.net.llm_config.sub_sub=False\
    model.net.llm_config.use_pl_probs=False\
    model.net.llm_config.oracle=False\
    model.net.llm_config.use_pl_w_feats=False\
    model.net.llm_config.mix_in_pls=False\
    model.net.llm_config.lora=False\
    model.net.llm_config.mix_in_pls_prob=0.25\
    data.dataset_config.sub_sample_replace=False\

    
