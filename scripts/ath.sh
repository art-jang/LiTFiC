#!/bin/bash

# cd ${SLURM_SUBMIT_DIR}
# cd "${WORK}/hrn/cslr2_t"

export CUDA_VISIBLE_DEVICES=1

export HYDRA_FULL_ERROR=1 # to get better error messages if job crashes
export WANDB_MODE=offline

python src/train.py task_name=debug experiment=llama3_haran paths=haran_athena \
    data.train_data_fraction=0.0001 \
    data.dataset_config.load_features=False \
    data.dataset_config.max_previous_sentences=0 \
    data.dataset_config.sub_sample_pct=0.25 \
    model.net.llm_config.sub_sub=False\
    model.net.llm_config.use_pl_probs=False\
    model.net.llm_config.oracle=False\
    model.net.llm_config.bg_desc=False\
    model.net.llm_config.use_pl_w_feats=False\
    data.dataset_config.sub_sample_replace=False\


