import os
import csv
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from models.logo_encoder import LogoEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def rotation_validator(encoder, transform):
    angles = [0, 90, 180, 270]
    mark_scores = {}

    val_dir = "data/processed/register/val"

    with torch.no_grad():
        for mark_id in tqdm(os.listdir(val_dir), desc="Rotation Validator"):
            mark_dir = os.path.join(val_dir, mark_id)
            if not os.path.isdir(mark_dir):
                continue

            sims_per_image = []
            for fname in os.listdir(mark_dir):
                img_path = os.path.join(mark_dir, fname)
                img = Image.open(img_path).convert("RGB")

                emb_list = []
                for angle in angles:
                    rotated = img.rotate(angle)
                    tensor_img = transform(rotated).unsqueeze(0).to(DEVICE)
                    embedding = encoder(tensor_img).cpu()
                    emb_list.append(embedding)

                sims = []
                for i in range(len(angles)):
                    for j in range(i + 1, len(angles)):
                        sim = torch.nn.functional.cosine_similarity(
                            emb_list[i], emb_list[j]
                        ).item()
                        sims.append(sim)

                avg_sim = sum(sims) / len(sims)
                sims_per_image.append(avg_sim)

            mark_scores[mark_id] = sum(sims_per_image) / len(sims_per_image) if sims_per_image else 0.0

    # Write CSV
    with open("results/ssl_rotation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mark_id", "ssl_rotation"])
        writer.writeheader()
        for mark_id, score in mark_scores.items():
            writer.writerow({"mark_id": mark_id, "ssl_rotation": score})

    avg_all = sum(mark_scores.values()) / len(mark_scores)
    print(f"Saved: results/ssl_rotation.csv")
    print(f"Rotation Invariance Score (mean): {avg_all:.4f}")
    return avg_all


def domain_invariance_validator(encoder, transform):
    reg_dir = "data/processed/register/val"
    prod_dir = "data/processed/product/val"

    shared_ids = set(os.listdir(reg_dir)).intersection(os.listdir(prod_dir))
    mark_scores = {}

    with torch.no_grad():
        for mark_id in tqdm(shared_ids, desc="Domain Invariance Validator"):
            reg_imgs = os.listdir(os.path.join(reg_dir, mark_id))
            prod_imgs = os.listdir(os.path.join(prod_dir, mark_id))

            if not reg_imgs or not prod_imgs:
                continue

            reg_embs = []
            prod_embs = []

            for img in reg_imgs:
                path = os.path.join(reg_dir, mark_id, img)
                tensor_img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
                reg_embs.append(encoder(tensor_img).cpu())

            for img in prod_imgs:
                path = os.path.join(prod_dir, mark_id, img)
                tensor_img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
                prod_embs.append(encoder(tensor_img).cpu())

            reg_mean = torch.stack(reg_embs).mean(dim=0)
            prod_mean = torch.stack(prod_embs).mean(dim=0)

            sim = torch.nn.functional.cosine_similarity(reg_mean, prod_mean).item()
            mark_scores[mark_id] = sim

    with open("results/ssl_domain.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mark_id", "ssl_domain"])
        writer.writeheader()
        for mark_id, score in mark_scores.items():
            writer.writerow({"mark_id": mark_id, "ssl_domain": score})

    avg_sim = sum(mark_scores.values()) / len(mark_scores)
    print(f"Saved: results/ssl_domain.csv")
    print(f"Domain Invariance Score (mean): {avg_sim:.4f}")
    return avg_sim


def main():
    encoder = LogoEncoder(backbone="resnet18").to(DEVICE)
    checkpoint = torch.load("checkpoints/encoder_episode_900.pth", map_location=DEVICE)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    rot_score = rotation_validator(encoder, transform)
    dom_score = domain_invariance_validator(encoder, transform)

    with open("results/ssl_summary.txt", "w") as f:
        f.write(f"Rotation Invariance Score: {rot_score:.4f}\n")
        f.write(f"Domain Invariance Score: {dom_score:.4f}\n")

    print("Saved SSL Validation Summary: results/ssl_summary.txt")

if __name__ == "__main__":
    main()
