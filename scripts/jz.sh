#!/bin/bash
#SBATCH --job-name=llama_feat_runs # job name
#SBATCH --account=ewl@h100                # project code
#SBATCH -C h100
#SBATCH --ntasks-per-node=4             # number of MPI tasks per node
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gres=gpu:4                     # number of GPUs per node                         # number of nodes
#SBATCH --qos=qos_gpu_h100-t4                  # (20h) jobs
#SBATCH --cpus-per-task=15                 # number of cores per tasks
#SBATCH --hint=nomultithread              # we get physical cores not logical
#SBATCH --time=40:00:00                 # maximum execution time (HH:MM:SS)
#SBATCH --output=/lustre/fswork/projects/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.out # output file name
#SBATCH --error=/lustre/fswork/projects/rech/vvh/upk96qz/hrn/vgg_slt/slurm_logs/llama_%j.err  # error file name

set -x # echo launched commands

module purge
# module load pytorch-gpu/py3/1.4.0 

module load arch/h100

eval "$(conda shell.bash hook)"
conda activate slt

# cd ${SLURM_SUBMIT_DIR}
# cd "${WORK}/hrn/cslr2_t"

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline
# paths.subtitles_path='/lustre/fswork/projects/rech/vvh/upk96qz/datasets/bobsl/hy_data/acmmm_pseudo_subtitles_v3.pkl'\
    # paths.llm_root="../hf/Llama-3.2-1B"\
    # model.net.mm_projector_config.hidden_size=2048\
srun python src/train.py task_name=llama_hf_sa_all experiment=llama3_haran paths=haran \
    model.net.llm_config.bg_desc=True\
    model.net.llm_config.use_pl_w_feats=True\
    model.net.llm_config.use_rec_prev=True\
    model.net.llm_config.use_prev_pls=False\
    model.net.llm_config.ret_sent=False\
    model.net.llm_config.use_gt_prev=False\
    model.net.llm_config.use_spottings=False\
    data.dataset_config.max_previous_sentences=1\
    data.dataset_config.filter_based_on_pls=False\
    data.dataset_config.aug_prev_neg=False\
    data.dataset_config.aug_prev_neg_prob=0.5\
    data.dataset_config.aug_prev=True\
    data.dataset_config.aug_prev_pct=0.5\
    data.dataset_config.train_cap_prob=0.5\
    data.dataset_config.sub_syn_aug_prob=0.0\
    data.dataset_config.sub_aug_drop=True\
    model.net.mm_projector_config.cslr2_options.use=False\
    model.net.llm_config.lora=True\
    model.net.llm_config.freeze_decoder=False\
    model.net.llm_config.use_bg_words=True\
    model.net.llm_config.drop_bg_sw=True\
    model.context_len=1\
    model.net.llm_config.mix_in_ret_prob=1.0\
    model.net.llm_config.mix_in_pls_prob=0.5\
    model.net.llm_config.mix_in_prev_prob=0.5\
    model.net.llm_config.mix_in_bg_prob=0.5\
    model.net.llm_config.mix_in_prev_pls=1.0\
    model.net.llm_config.mix_in_spottings=1.0\
    model.net.llm_config.drop_bgw_pct=0.5\
    model.net.llm_config.drop_pl_pct=0.5\


