import os
import cv2
import torch
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]        
            mask = augmented["mask"]
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            mask = torch.tensor(mask, dtype=torch.float32)    

        mask = mask.float().unsqueeze(0) / 255.0  # (1, H, W), values 0-1
        return img, mask