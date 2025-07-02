from datasets.logo_fsl_dataset import LogoFSLDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

# Example transform:
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# Load Registration split for test
dataset = LogoFSLDataset(
    root_dir="data/processed",
    split="train",
    domain="register",
    transform=transform
)

print(f"Number of samples: {len(dataset)}")

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for imgs, labels in loader:
    print(f"Batch image shape: {imgs.shape}")  # e.g. [4, 3, 224, 224]
    print(f"Labels: {labels}")
    break  # just test first batch
