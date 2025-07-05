# 🏷️ Logo Few-Shot Learning & Self-Supervised Invariance

**Project Title:**  
**Cross-Domain Logo Recognition with Prototypical Networks and Self-Supervised Invariance Validation**

---

## 📌 Project Summary

This project implements a pipeline for **few-shot learning (FSL)** on logo images using **Prototypical Networks**, combined with **self-supervised learning (SSL)** to validate feature invariance properties.  
It demonstrates how rotation and domain invariance scores relate to few-shot classification performance, using both **t-SNE/UMAP visualizations** and **correlation plots**.

The goal:  
**Evaluate whether self-supervised rotation invariance signals correlate with how well logos generalize across domains (register ↔ product).**

---

## 📂 Directory Structure

```bash
logo_fsl_ssl/
├── data
│   ├── raw/
|   |   ├── product_annotations_train_split.csv # Not used for this research. But useful if the research goal is domain adaptation instead of domain generalization
|   |   ├── product_annotations_val_split.csv   # Used for cross-domain evaluation in evaluate_product.py
|   |   ├── register_tm_train_split.csv         # Used for model training in train_few_shot.py
|   |   ├── register_tm_val_split.csv           # Used for in-domain evaluation in evaluate_register.py
├── datasets/                # Logo dataset loader, few-shot sampler
│   ├── logo_fsl_dataset.py  # Loads registration & product splits
│   ├── few_shot_sampler.py  # Dynamically builds FSL tasks
├── models/                  # Encoder (ResNet18) + Prototypical Head
│   ├── logo_encoder.py     # ResNet-based feature extractor
│   ├── prototypical_head.py # Computes class prototypes
├── scripts/                 # All run scripts
│   ├── ssl_validate.py          # Computes SSL scores (rotation, domain)
│   ├── evaluate_product.py     # Runs few-shot cross-domain evaluation (product)
│   ├── evaluate_register.py    # Runs few-shot in-domain evaluation (register)
│   ├── visualize_embeddings.py # t-SNE / UMAP embeddings
│   ├── plot_ssl_correlation.py # Correlation plots
│   ├── plot_domain_correlation.py # domain Correlation plots
│   ├── plot_results.py         # results plots
│   ├── preprocess_register_and_product.py # downloads and preprocesses images
│   ├── save_summary.py # saves a summary of the results
│   ├── test_logo_dataset.py   # tests if labels are properly extracted during preprocessing
│   ├── train_few_shot.py # trains the registration dataset. The output is model weights in checkpoints
├── checkpoints/             # Model weights (use .gitignore for large files)
├── results/                 # CSVs, plots, summaries
├── data/processed/          # ⚠️ Not versioned — add your data
├── README.md                # This file
└── .gitignore               # Large data / zips excluded
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

git clone https://github.com/fabariagbora/logo_fsl_ssl/
cd logo_fsl_ssl

### 2️⃣ Create a Python environment

Example using Conda:
conda create -n logo-fsl python=3.10
conda activate logo-fsl

Or virtualenv:
python -m venv venv
source venv/bin/activate

### 3️⃣ Install dependencies

Create a requirements.txt if not already done:
torch>=1.13
torchvision
numpy
pandas
scikit-learn
umap-learn
matplotlib
tqdm
Pillow

Install:
pip install -r requirements.txt


## 📁 Add Your Data
Use data/raw or add your data

Your processed logo dataset should be in:
data/processed/

```bash
├── register/val/<mark_id>/*.jpg
├── product/val/<mark_id>/*.jpg
⚠️ This folder is not pushed to GitHub — it stays local.
```

## 🚀 How to Run

### ✅ Step 1 — Run Preprocessor

python scripts/preprocess_register_and_product.py
creates:
Downloaded images from URL links.
Cropped logos using gt_annotation bounding box from the product dataset.

handles:
Encoding errors (utf-8).
Download progress using tqdm

Saved in data/processed/

### ✅ Step 2 — Run Few-Shot Training

python scripts/train_few_shot.py
creates:
model weights in checkpoints/

*No training on product data to ensure true domain generalization.*

### ✅ Step 3 — Run few-shot evaluations
python scripts/evaluate_product.py
python scripts/evaluate_register.py

Creates:
results/product_per_class.csv
results/register_per_class.csv
results/eval_product.txt
results/eval_register.txt

Records csv and txt outputs in results/ 

### ✅ Step 4 — Run SSL validators

python scripts/ssl_validate.py
Creates:
results/ssl_rotation.csv
results/ssl_domain.csv
results/ssl_summary.txt



### ✅ Step 5 — Generate correlation plots

python scripts/plot_ssl_correlation.py
Creates:
results/ssl_rotation_vs_fewshot_product.png
results/ssl_rotation_vs_fewshot_register.png
results/ssl_correlation.csv # ⬅️ Save the Pearson r values too!

### ✅ Step 6 — Visualize embeddings

python scripts/visualize_embeddings.py
Creates:
results/tsne_register.png
results/umap_register.png
results/tsne_product.png
results/umap_product.png

## 📊 Example Metrics

Product domain few-shot accuracy: ~82.68%
Register domain few-shot accuracy: varies by run & split
SSL rotation invariance: saved per class
Correlation: Pearson r shows how well invariance aligns with few-shot performance.

## 🚫 What’s NOT Pushed to Git

To keep the repo lightweight:
Large processed data → data/processed/
Large zipped files → *.zip
Checkpoints (optionally use Git LFS)

Example .gitignore:
```bash
# Ignore data & archives
data/processed/
*.zip
checkpoints/
```

## 📝 License

MIT License
Feel free to adapt, reuse, and build on this work.

## 📚 Citation

This project uses the following open dataset.  
If you build on this work or use the dataset, please cite the original authors:

```bibtex
@software{Zhao_Open_set_cross_2025,
  author  = {Zhao, Xiaonan and Li, Chenge and Liu, Zongyi and Feng, Yarong and Chen, Qipin},
  month   = feb,
  title   = {{Open set cross domain few shot logo recognition}},
  url     = {https://github.com/wacv2025-image-quality-workshop2/cross-domain-logo-recognition},
  version = {1.0.4},
  year    = {2025}
}
```

## ✨ Acknowledgements

Inspired by standard Prototypical Networks (https://arxiv.org/abs/1703.05175) Snell et al.
Self-supervised methods inspired by SimCLR, BYOL, and basic SSL invariance tests.

## 🧑‍💻 Maintainer

Fabari Agbora
Data Scientist at Nepsix Technology Limited
📧 agfabariagbora@gmail.com


## 🗂️ Future Work

Experiment with other SSL tasks (jigsaw, colorization)
Try additional backbones (ViT, ConvNeXt)
Publish a cleaned dataset & benchmark config

✅✅✅ Happy few-shot learning!
If you use this, please ⭐️ the repo and open an issue if you find bugs or improvements.
