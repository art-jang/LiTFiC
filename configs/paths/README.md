# 📘 BOBSL Dataset Setup Guide

This README describes how to obtain and prepare all necessary files for using the **BOBSL** dataset for Sign Language Translation (SLT).

---

## 📦 Step 1: Download CSLR Annotations for BOBSL

Please download the **CSLR annotations for BOBSL** (approx. 1.9GB) from the official CSLR2 project page:

🔗 [https://gulvarol.github.io/cslr2/data.html](https://gulvarol.github.io/cslr2/data.html)


| Variable                 | Path                                                                             |
| ------------------------ | -------------------------------------------------------------------------------- |
| `subset2episode`         | `bobsl/splits/subset2episode.json`                                               |
| `vocab_pkl`              | `bobsl/vocab/8697_vocab.pkl`                                                     |
| `info_pkl`               | `bobsl/cslr_annotation/bobsl/unannotated-info/info-src_videos_2236.pkl`          |
| `subtitles_path`         | `bobsl/subtitles_pkl/best_delta_postpro_bobsl_best_traineval_0_pslab_ftune_.pkl` |
| `aligned_subtitles_path` | `bobsl/subtitles_pkl/manually-aligned.pkl`                                       |
| `synonyms_pkl`           | `bobsl/syns/synonym_pickle_english_and_signdict_and_signbank.pkl`                |


subset2episode: bobsl/splits/subset2episode.json #
vocab_pkl: bobsl/vocab/8697_vocab.pkl
info_pkl: bobsl/cslr_annotation/bobsl/unannotated-info/info-src_videos_2236.pkl
subtitles_path: bobsl/subtitles_pkl/best_delta_postpro_bobsl_best_traineval_0_pslab_ftune_.pkl
aligned_subtitles_path: bobsl/subtitles_pkl/manually-aligned.pkl
synonyms_pkl: bobsl/syns/synonym_pickle_english_and_signdict_and_signbank.pkl




llm_root: /mnt/bear1/users/jyj/VGG/lamma3/Meta-Llama-3-8B
bleurt_path: lucadiliello/BLEURT-20
annotations_pkl: /bobsl/lmdbs/lmdb-pl_vswin_t-bs256_float16/
vid_features_lmdb: /mnt/lynx2/datasets/bobsl/bobsl/cslr_annotation/bobsl/lmdbs/lmdb-feats_vswin_t-bs256_float16/
blip_cap_path: /home/jyj/Workspace/VGG/vgg_slt/add_files/blip2_captions_v2.pkl
val_episode_ind_path: /home/jyj/Workspace/VGG/vgg_slt/add_files/man_val_final_start_indices.json
test_episode_ind_path: /home/jyj/Workspace/VGG/vgg_slt/add_files/man_test_final_start_indices.json
train_cap_path: null
spottings_path: null
