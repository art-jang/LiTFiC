<div align="center">

# Lost in Translation, Found in Context:<br> Sign Language Translation with Contextual Cues (CVPR 2025)

---

### 🔥 **Official Code Repository and Usage Instructions**
</div>

## 📄 Paper & Project Links

- **[📘 Paper (arXiv)](https://arxiv.org/abs/2501.09754)**: Official research paper for *Lost in Translation, Found in Context*.

- **[🌐 Project Page](https://www.robots.ox.ac.uk/~vgg/research/litfic/)**: Detailed project webpage with demos, code, and additional resources.

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/Haran71/vgg_slt_pre_release.git
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

<br>

## ⚡  Your Superpowers

```bash
python src/train.py trainer=ddp trainer.devices=[0,1,2,3] task_name=vgg_slt data=bobsl data.batch_size=8 data.num_workers=16
```
