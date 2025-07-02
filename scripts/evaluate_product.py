import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from models.logo_encoder import LogoEncoder
from models.prototypical_head import PrototypicalHead


from torch.utils.data import Dataset
from PIL import Image

class ProductLogoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        self.classes = sorted(os.listdir(root_dir))  # mark_id folders

        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.endswith(('.jpg', '.png')):
                    self.samples.append( (os.path.join(cls_folder, fname), cls) )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

from datasets.few_shot_sampler import FewShotEpisodeSampler

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    N_WAY = 5
    K_SHOT = 5
    M_QUERY = 5
    NUM_EPISODES = 200

    torch.manual_seed(42)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    dataset = ProductLogoDataset(
        root_dir="data/processed/product/val",
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

    checkpoint = torch.load("checkpoints/encoder_episode_900.pth", map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    # For tracking per-class
    per_class_correct = {}
    per_class_total = {}

    results = []

    with torch.no_grad():
        for episode_idx, batch in enumerate(tqdm(loader, total=NUM_EPISODES)):
            images, labels = zip(*batch)
            images = torch.stack(images).to(DEVICE)
            labels = list(labels)

            support_images, query_images = [], []
            support_labels, query_labels = [], []

            unique_labels = list(set(labels))  # e.g. mark IDs like 'abc123'
            label2id = {label: idx for idx, label in enumerate(unique_labels)}  # maps mark ID → 0...N_WAY

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

            # Count correct per real mark_id
            for i in range(len(query_labels)):
                proto_cls_id = query_labels[i].item()
                mark_id = unique_labels[int(proto_cls_id)]

                if mark_id not in per_class_correct:
                    per_class_correct[mark_id] = 0
                    per_class_total[mark_id] = 0

                if preds[i].item() == proto_cls_id:
                    per_class_correct[mark_id] += 1

                per_class_total[mark_id] += 1

    avg_acc = sum(results) / len(results)
    print(f"[PRODUCT DOMAIN] Few-Shot Acc: {avg_acc*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open("results/eval_product.txt", "w") as f:
        f.write(f"Average Accuracy: {avg_acc*100:.2f}%\n")

    # Write per-class CSV
    per_class_acc = {mark_id: per_class_correct[mark_id]/per_class_total[mark_id]
                     for mark_id in per_class_correct}

    with open("results/product_per_class.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mark_id", "fewshot_acc"])
        for mark_id, acc in per_class_acc.items():
            writer.writerow([mark_id, acc])

    print("Saved: results/product_per_class.csv")

if __name__ == "__main__":
    main()
