<div align="center">

# Lost in Translation, Found in Context:<br>Sign Language Translation with Contextual Cues<br>(CVPR 2025)
---
### 🔥 **Official Code Repository and Usage Instructions**
</div>

## 📄 Paper & Project Links

- **[📘 Paper (arXiv)](https://arxiv.org/abs/2501.09754)**: Official research paper for *Lost in Translation, Found in Context*.

- **[🌐 Project Page](https://www.robots.ox.ac.uk/~vgg/research/litfic/)**: Detailed project webpage with demos, code, and additional resources.

## 🚀  Environment

```bash
# clone project
git clone https://github.com/art-jang/Lost-in-Translation-Found-in-Context.git
cd vgg_slt_pre_release

conda create -n myenv python=3.10
conda activate myenv

pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# [Optional]
conda install nvidia/label/cuda-12.1.0::cuda-toolkit

# install requirements
pip install -r requirements.txt
conda install bioconda::java-jdk
```


## 📥 Data Download for Training and Testing

To train and test the model properly, please follow the detailed instructions on how to download and prepare the datasets.

You can find the full explanation and configuration details here:  
➡️ [Dataset Paths and Download Instructions](https://github.com/art-jang/Lost-in-Translation-Found-in-Context/tree/main/configs/paths)

This guide covers all necessary datasets, including annotations, video features, subtitles, and more.


## Training with Different Modalities (Bash Commands)

Use the following commands to train the model with different modality combinations.  
Specify which GPUs to use with the `trainer.devices` option.

```bash
# 1) Train using only the video (vid) modality
python src/train.py trainer.devices=[0,1,2,3] task_name=vid experiment=vid

# 2) Train using video + pseudo-gloss captions (pg) modality
python src/train.py trainer.devices=[0,1,2,3] task_name=vid+pg experiment=vid+pg

# 3) Train using video + pseudo-gloss + previous sentence (prev) modality
python src/train.py trainer.devices=[0,1,2,3] task_name=vid+pg+prev experiment=vid+pg+prev

# 4) Train using video + pseudo-gloss + previous sentence + background (bg) modality
python src/train.py trainer.devices=[0,1,2,3] task_name=vid+pg+prev+bg experiment=vid+pg+prev+bg
```

### Viewing Training Logs
If you want to track and visualize training logs using Weights & Biases (W&B),  
add the option `logger=wandb` to your training command. For example:

```bash
python src/train.py trainer.devices=[0,1,2,3] task_name=vid experiment=vid logger=wandb
```

## Evaluation (Pretrained Checkpoint - Coming Soon)

To evaluate a trained model, run the following command:

```bash
python src/eval.py trainer.devices=[0,1,2,3] \
  task_name=vid+pg+prev+bg-eval \
  experiment=vid+pg+prev+bg \
  ckpt_path={CKPT_PATH}
```