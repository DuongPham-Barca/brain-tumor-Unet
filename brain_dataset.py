import os
import cv2
import torch
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

        self.valid_pairs = []

        for img_name in self.images:
            base = os.path.splitext(img_name)[0]
            mask_name = base + "_mask.tif"

            mask_path = os.path.join(mask_dir, mask_name)
            img_path = os.path.join(image_dir, img_name)

            if os.path.exists(mask_path):
                self.valid_pairs.append((img_name, mask_name))


    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None:
            raise ValueError(f"Broken image: {img_path}")
        if mask is None:
            raise ValueError(f"Broken mask: {mask_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug["image"]
            mask = aug["mask"]

        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, mask