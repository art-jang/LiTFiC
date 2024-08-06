#!/bin/bash
. /home/haran/miniconda3/etc/profile.d/conda.sh
conda activate slt

# cd ${SLURM_SUBMIT_DIR}
# cd "${WORK}/hrn/cslr2_t"

export CUDA_VISIBLE_DEVICES=0

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline

python src/train.py task_name=debug experiment=llama3_haran paths=haran_athena \
    data.dataset_config.sub_sample_shuffle=True \
    data.dataset_config.max_previous_sentences=1 \
    model.net.mm_projector_config.cslr2_options.ckpt_path='model_best.pth' \
    paths.llm_root='meta-llama/Meta-Llama-3-8B' \
    data.train_data_fraction=0.0001 \
    data.batch_size=1


