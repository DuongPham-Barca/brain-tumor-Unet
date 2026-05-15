import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Thay đổi đường dẫn cho ảnh thực tế của bạn
image_path = "images.jpg"  # .jpg, không .tif
mask_path = "datasets/test/masks/brisc2025_test_00945_pi_sa_t1.png"   # .png, không .tif
model_path = "best_model.pt"
     
device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Normalize cho 1 channel
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2()
])

# Đọc ảnh dưới dạng grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Cannot load image: {image_path}")

mask = cv2.imread(mask_path, 0)
if mask is None:
    raise ValueError(f"Cannot load mask: {mask_path}")

# Transform + đưa vào model
aug = transform(image=img)
img_tensor = aug["image"].unsqueeze(0).to(device)  # [1, 1, 256, 256]

with torch.no_grad():
    pred = model(img_tensor)
    pred = torch.sigmoid(pred).squeeze().cpu().numpy()
# Thử cả 3 threshold cùng lúc
print("\n=== Test threshold ===")
for threshold in [0.5, 0.6, 0.7]:
    pred_bin_test = (pred > threshold).astype(np.uint8)
    print(f"Threshold {threshold}: pixel u = {pred_bin_test.sum()}")
print("=" * 25 + "\n")
pred_bin = (pred > 0.5).astype(np.uint8)
pred_bin = cv2.resize(pred_bin, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# Convert grayscale to RGB cho display
img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
pred_bin_display = cv2.cvtColor(pred_bin * 255, cv2.COLOR_GRAY2BGR)

# Overlay: vẽ màu xanh lên vùng dự đoán
overlay = img_display.copy()
overlay[pred_bin == 1] = [0, 255, 0]  # Xanh lá

alpha = 0.4
overlay_img = cv2.addWeighted(img_display, 1 - alpha, overlay, alpha, 0)

# Visualization
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.title("Image")
plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
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
plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()