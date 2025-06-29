import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

from datasets.logo_fsl_dataset import LogoFSLDataset
from models.logo_encoder import LogoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader

from scipy.sparse import issparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_and_plot(domain_name, checkpoint_path, tsne_out, umap_out):
    print(f"🚀 Processing domain: {domain_name}")

    dataset = LogoFSLDataset(
        root_dir="data/processed",
        split="val",
        domain=domain_name,
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    encoder = LogoEncoder(backbone="resnet18").to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            feats = encoder(imgs).cpu().numpy()
            all_embeddings.append(feats)
            all_labels.extend(labels)  # ✅ no int() needed!

    embeddings = np.concatenate(all_embeddings, axis=0)

    labels_unique = sorted(set(all_labels))
    label2id = {label: idx for idx, label in enumerate(labels_unique)}
    numeric_labels = [label2id[label] for label in all_labels]

    tsne_proj = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_proj[:, 0], tsne_proj[:, 1], c=numeric_labels, s=3, cmap="tab20")
    plt.title(f"t-SNE: {domain_name.capitalize()} Domain")
    plt.savefig(tsne_out)
    print(f"✅ Saved: {tsne_out}")

    umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)
    umap_proj = np.asarray(umap_proj)

    plt.figure(figsize=(8, 6))
    plt.scatter(umap_proj[:, 0], umap_proj[:, 1], c=numeric_labels, s=3, cmap="tab20")
    plt.title(f"UMAP: {domain_name.capitalize()} Domain")
    plt.savefig(umap_out)
    print(f"✅ Saved: {umap_out}")

extract_and_plot(
    domain_name="register",
    checkpoint_path="checkpoints/encoder_episode_900.pth",
    tsne_out="results/tsne_register.png",
    umap_out="results/umap_register.png"
)

extract_and_plot(
    domain_name="product",
    checkpoint_path="checkpoints/encoder_episode_900.pth",
    tsne_out="results/tsne_product.png",
    umap_out="results/umap_product.png"
)

print("✅✅✅ All embeddings visualized and saved!")
