# scripts/evaluate_register.py

import os
import csv
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
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
    NUM_EPISODES = 200  # you can adjust this if needed

    torch.manual_seed(42)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = LogoFSLDataset(
        root_dir="data/processed",
        split="val",
        domain="register",   # ✅ in-domain
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
        collate_fn=lambda batch: batch
    )

    encoder = LogoEncoder(backbone="resnet18").to(DEVICE)
    proto_head = PrototypicalHead(metric="euclidean").to(DEVICE)

    # ✅ Load checkpoint
    checkpoint = torch.load("checkpoints/encoder_episode_900.pth", map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    results = []
    per_class_acc = {}  # ✅ mark_id -> list of accuracies

    with torch.no_grad():
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
            preds = logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()
            results.append(acc)

            # ✅ Track per-class for this episode
            for cls_id in range(N_WAY):
                cls_query_idx = (query_labels == cls_id).nonzero(as_tuple=True)[0]
                cls_acc = (preds[cls_query_idx] == query_labels[cls_query_idx]).float().mean().item()

                mark_id = unique_labels[cls_id]  # original mark_id string
                if mark_id not in per_class_acc:
                    per_class_acc[mark_id] = []
                per_class_acc[mark_id].append(cls_acc)

    avg_acc = sum(results) / len(results)
    print(f"✅ [REGISTER DOMAIN] Few-Shot Acc: {avg_acc*100:.2f}%")

    os.makedirs("results", exist_ok=True)

    # ✅ Save per-class CSV
    with open("results/register_per_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mark_id", "fewshot_acc"])
        for mark_id, acc_list in per_class_acc.items():
            avg_cls_acc = sum(acc_list) / len(acc_list)
            writer.writerow([mark_id, avg_cls_acc])
    print("✅ Saved: results/register_per_class.csv")

    with open("results/eval_register.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_acc*100:.2f}%\n")

if __name__ == "__main__":
    main()
