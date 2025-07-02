import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm

# ---- PATHS ---- #
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"

REGISTER_SPLITS = {
    "train": os.path.join(DATA_RAW, "register_tm_train_split.csv"),
    "val": os.path.join(DATA_RAW, "register_tm_val_split.csv"),
}

PRODUCT_SPLITS = {
    "train": os.path.join(DATA_RAW, "product_annotations_train_split.csv"),
    "val": os.path.join(DATA_RAW, "product_annotations_val_split.csv"),
}

# ---- UTILS ---- #
def download_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Download failed: {url}\n{e}")
        return None


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# ---- PROCESSORS ---- #
def process_register(split_name, csv_path):
    print(f"\n Processing Registration split: {split_name}")
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        mark_id = row["mark_id"]
        url = row["content"]
        img = download_image(url)
        if img:
            img_path = os.path.join(DATA_PROCESSED, "register", split_name, mark_id, f"{row['id']}.jpg")
            save_image(img, img_path)


def process_product(split_name, csv_path):
    print(f"\n Processing Product split: {split_name}")
    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        url = row["url"]
        img = download_image(url)
        if img is None:
            continue

        try:
            annotations = json.loads(row["gt_annotation"].replace("'", "\""))
        except Exception as e:
            print(f"JSON parsing failed for {row['id']}:\n{e}")
            continue

        for i, ann in enumerate(annotations):
            mark_id = ann["label"]
            x, y, w, h = int(ann["left"]), int(ann["top"]), int(ann["width"]), int(ann["height"])
            crop = img.crop((x, y, x + w, y + h))

            crop_path = os.path.join(
                DATA_PROCESSED, "product", split_name, mark_id, f"{row['id']}_crop{i}.jpg"
            )
            save_image(crop, crop_path)


# ---- MAIN EXECUTION ---- #
if __name__ == "__main__":
    print("Starting Preprocessing...\n")

    for split, csv_path in REGISTER_SPLITS.items():
        process_register(split, csv_path)

    for split, csv_path in PRODUCT_SPLITS.items():
        process_product(split, csv_path)

    print("\n Done: All registration and product logo images have been processed and saved.")
