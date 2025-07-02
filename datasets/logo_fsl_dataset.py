import os
from torch.utils.data import Dataset
from PIL import Image

class LogoFSLDataset(Dataset):
    def __init__(self, root_dir, split="train", domain="register", transform=None):
        self.root_dir = os.path.join(root_dir, domain, split)
        self.transform = transform

        # Only keep valid classes with images
        self.classes = []
        self.samples = []

        for mark_id in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, mark_id)
            if not os.path.isdir(class_dir):
                continue
            images = os.listdir(class_dir)
            if len(images) == 0:
                continue  # skip empty
            for img in images:
                self.samples.append((os.path.join(class_dir, img), mark_id))
            self.classes.append(mark_id)

        assert len(self.samples) > 0, f"No valid images found in {self.root_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mark_id = self.samples[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, mark_id
