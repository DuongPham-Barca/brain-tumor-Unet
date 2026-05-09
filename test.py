import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2


image_path = "datasets/test/images/TCGA_CS_4941_19960909_15.tif"
mask_path = "datasets/test/masks/TCGA_CS_4941_19960909_15_mask.tif"
model_path = "weights/best_model.pt"


device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()


transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

mask = cv2.imread(mask_path, 0)

aug = transform(image=img)
img_tensor = aug["image"].unsqueeze(0).to(device)


with torch.no_grad():
    pred = model(img_tensor)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()

pred_bin = (pred > 0.5).astype(np.uint8)


overlay = img.copy()
overlay[pred_bin == 1] = [255, 0, 0]   # tô đỏ vùng tumor

alpha = 0.4
overlay_img = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)


plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image")
plt.imshow(img)

plt.subplot(1, 4, 2)
plt.title("Ground Truth")
plt.imshow(mask, cmap="gray")

plt.subplot(1, 4, 3)
plt.title("Prediction")
plt.imshow(pred_bin, cmap="gray")

plt.subplot(1, 4, 4)
plt.title("Overlay")
plt.imshow(overlay_img)

plt.show()