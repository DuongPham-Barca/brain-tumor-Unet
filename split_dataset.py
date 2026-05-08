import os
from sklearn.model_selection import train_test_split
import shutil

data_dir = "kaggle_3m"
base_dir = "datasets"

image_paths = []
mask_paths = []


for case in os.listdir(data_dir):
    case_path = os.path.join(data_dir, case)

    if not os.path.isdir(case_path):
        continue

    files = os.listdir(case_path)

    for file in files:
        if "_mask" not in file:
            image_file = file
            mask_file = file.replace(".tif", "_mask.tif")

            image_path = os.path.join(case_path, image_file)
            mask_path = os.path.join(case_path, mask_file)

            if os.path.exists(image_path) and os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)

print(len(image_paths), len(mask_paths))
print(image_paths[0])
print(mask_paths[0])

train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=0.5, random_state=42
)

splits = {
    "train": (train_imgs, train_masks),
    "val": (val_imgs, val_masks),
    "test": (test_imgs, test_masks)
}

for split in splits:
    os.makedirs(os.path.join(base_dir, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, split, "masks"), exist_ok=True)

for split, (imgs, masks) in splits.items():
    for img, mask in zip(imgs, masks):
        shutil.copy(img, os.path.join(base_dir, split, "images", os.path.basename(img)))
        shutil.copy(mask, os.path.join(base_dir, split, "masks", os.path.basename(mask)))



