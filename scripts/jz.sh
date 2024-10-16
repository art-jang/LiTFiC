#!/bin/bash
#SBATCH --job-name=llama_prev_exp # job name
#SBATCH --account=vvh@a100                # project code
#SBATCH -C a100
#SBATCH --ntasks-per-node=4             # number of MPI tasks per node
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gres=gpu:4                     # number of GPUs per node                         # number of nodes
#SBATCH --qos=qos_gpu_a100-t3                  # (20h) jobs
#SBATCH --cpus-per-task=15                 # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATCH --time=20:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.out # output file name
#SBATCH --error=/gpfswork/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.err  # error file name

set -x # echo launched commands

module purge
# module load pytorch-gpu/py3/1.4.0 

module load arch/a100

. ${WORK}/miniconda3/etc/profile.d/conda.sh
conda activate slt

# cd ${SLURM_SUBMIT_DIR}
# cd "${WORK}/hrn/cslr2_t"

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline
# paths.subtitles_path='/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/hy_data/acmmm_pseudo_subtitles_v3.pkl'\
srun python src/train.py task_name=llama_feats_bgw_pls experiment=llama3_haran paths=haran \
    model.net.llm_config.bg_desc=False\
    model.net.llm_config.use_rec_prev=False\
    model.net.llm_config.ret_sent=False\
    model.net.llm_config.use_prev_pls=False\
    model.net.llm_config.use_pl_w_feats=False\
    data.dataset_config.max_previous_sentences=0\
    data.dataset_config.filter_based_on_pls=False\
    model.net.llm_config.lora=True\
    model.net.llm_config.use_bg_words=True\
    model.net.llm_config.drop_bg_sw=True\
    model.context_len=1\
    model.net.llm_config.mix_in_ret_prob=1.0\
    model.net.llm_config.mix_in_pls_prob=0.0\
    model.net.llm_config.mix_in_prev_prob=1.0\
    model.net.llm_config.mix_in_bg_prob=1.0\
    model.net.llm_config.mix_in_prev_pls=1.0\

