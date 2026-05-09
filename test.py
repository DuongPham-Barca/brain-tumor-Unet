import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

image_path = "datasets/test/images/TCGA_HT_A5RC_19990831_14.tif"
mask_path = "datasets/test/masks/TCGA_HT_A5RC_19990831_14_mask.tif"
model_path = "best_model.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError(f"Cannot load image: {image_path}")

if len(img.shape) == 2:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, 0)

aug = transform(image=img)
img_tensor = aug["image"].unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(img_tensor)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()

pred_bin = (pred > 0.5).astype(np.uint8)
pred_bin = cv2.resize(pred_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

overlay = img.copy()
overlay[pred_bin == 1] = [255, 0, 0]

alpha = 0.4
overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)

plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Ground Truth")
plt.imshow(mask, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Prediction")
plt.imshow(pred_bin, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(overlay_img)
plt.axis("off")

plt.tight_layout()
plt.show()