# scripts/train_few_shot.py

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datasets.logo_fsl_dataset import LogoFSLDataset
from datasets.few_shot_sampler import FewShotEpisodeSampler
from models.logo_encoder import LogoEncoder
from models.prototypical_head import PrototypicalHead

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    N_WAY = 5
    K_SHOT = 5
    M_QUERY = 5
    NUM_EPISODES = 1000
    LR = 1e-4

    torch.manual_seed(42)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = LogoFSLDataset(
        root_dir="data/processed",
        split="train",
        domain="register",
        transform=transform
    )

    sampler = FewShotEpisodeSampler(
        dataset=dataset,
        n_way=N_WAY,
        k_shot=K_SHOT,
        m_query=M_QUERY,
        num_episodes=NUM_EPISODES
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4,
        collate_fn=lambda batch: batch  # ✅ critical: get raw (img, label) pairs
    )

    encoder = LogoEncoder(backbone="resnet18").to(DEVICE)
    proto_head = PrototypicalHead(metric="euclidean").to(DEVICE)

    optimizer = optim.Adam(encoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)

    for episode_idx, batch in enumerate(tqdm(loader, total=NUM_EPISODES)):

        images, labels = zip(*batch)
        images = torch.stack(images).to(DEVICE)
        labels = list(labels)

        support_images, query_images = [], []
        support_labels, query_labels = [], []

        unique_labels = list(set(labels))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}

        for cls in unique_labels:
            cls_indices = [i for i, l in enumerate(labels) if l == cls]
            support_idx = cls_indices[:K_SHOT]
            query_idx = cls_indices[K_SHOT:]

            support_images.append(images[support_idx])
            query_images.append(images[query_idx])

            support_labels.extend([label2id[cls]] * K_SHOT)
            query_labels.extend([label2id[cls]] * M_QUERY)

        support_images = torch.cat(support_images).to(DEVICE)
        query_images = torch.cat(query_images).to(DEVICE)
        support_labels = torch.tensor(support_labels).to(DEVICE)
        query_labels = torch.tensor(query_labels).to(DEVICE)

        support_embeddings = encoder(support_images)
        query_embeddings = encoder(query_images)

        prototypes = []
        for cls_id in range(N_WAY):
            cls_embeddings = support_embeddings[support_labels == cls_id]
            proto = cls_embeddings.mean(dim=0)
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)

        logits = proto_head(query_embeddings, prototypes)
        loss = criterion(logits, query_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode_idx % 10 == 0:
            acc = (logits.argmax(dim=1) == query_labels).float().mean().item()
            print(f"Episode {episode_idx} | Loss: {loss.item():.4f} | Acc: {acc*100:.2f}%")

        if episode_idx % 100 == 0 and episode_idx != 0:
            checkpoint = {
                'encoder_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode': episode_idx
            }
            torch.save(checkpoint, f"checkpoints/encoder_episode_{episode_idx}.pth")
            print(f"✅ Saved checkpoint at episode {episode_idx}")

    print("\n✅ Done training few-shot encoder!")

if __name__ == "__main__":
    main()
