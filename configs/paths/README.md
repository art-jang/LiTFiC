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

## 🦙 Step 2: Download LLaMA 3 (8B) Model

To use the LLaMA-based language model, download the **Meta-LLaMA-3-8B** model from Hugging Face:

🔗 [https://huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

| Variable                 | Path                                                                             |
| ------------------------ | -------------------------------------------------------------------------------- |
| `llm_root`         | {Meta-Llama-8-8B directory}                                               |


## 🗂 Step 3: Download Annotations and Video Features (LMDB)

Download both from the official BOBSL dataset page hosted by the VGG group.

🔗 [https://www.robots.ox.ac.uk/~vgg/data/bobsl/#download](https://www.robots.ox.ac.uk/~vgg/data/bobsl/#download)

---

Scroll to the **"Automatic Annotations"** section on the website, and download:

> **CONTINUOUS SIGN SEQUENCES - SWIN V2 FEATURES PSEUDO-LABELS → LMDB**
> **Backgounrd Captions - BLIP2 → download**

Scroll to the **"Video Features"** section on the website, and download:

> **SWIN FEATURES V2 → LMDB**


| Variable                 | Path                                                                             |
| ------------------------ | -------------------------------------------------------------------------------- |
| `annotations_pkl`        | {PSEUDO-LABELS FOLDER}                                                |
| `blip_cap_path`        | {BLIP2 CAPTION FILE}                                                |
| `vid_features_lmdb`      | {Video Features FOLDER}                                                       |

## 📊 Step 4: Download BLEURT Model for Evaluation

BLEURT is a learned evaluation metric used to assess the quality of generated text.

You can use the BLEURT-20 model from Hugging Face to automatically compute evaluation scores (No manual download is required).

| Variable                 | Path                                                                             |
| ------------------------ | -------------------------------------------------------------------------------- |
| `bleurt_path`            | lucadiliello/BLEURT-20 (will automatically download this model)                  |


## 🧪 Step 5: Download Additional Resources (Custom Captions & Split Info)

- `val_episode_ind_path`: Start indices of manually segmented validation episodes for DDP inference
- `test_episode_ind_path`: Start indices of manually segmented test episodes for DDP inference
- `train_cap_path`: Previous captions extracted from a model (Vid+PG+BG)

| Variable                 | File name
| ------------------------ | -------------------------------------------------------------------------------- |
| `val_episode_ind_path`   | val_start_indices.json [download](https://mm.kaist.ac.kr/projects/LiCFiC/val_start_indices.json)   |
| `test_episode_ind_path`   | test_start_indices.json [download](https://mm.kaist.ac.kr/projects/LiCFiC/test_start_indices.json)   |
| `train_cap_path`   | val_start_indices.json [download](https://mm.kaist.ac.kr/projects/LiCFiC/prev_captions.json)   |