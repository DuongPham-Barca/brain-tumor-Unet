import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

src_dirs = [
    r"c:\Users\acer\Documents\brain_tumor_backup\brisc2025\classification_task\train\no_tumor",
    r"c:\Users\acer\Documents\brain_tumor_backup\brisc2025\classification_task\test\no_tumor",
]

out_dirs = {
    "train": {
        "images": r"c:\Users\acer\Documents\brain_tumor_backup\datasets\train\images",
        "masks":  r"c:\Users\acer\Documents\brain_tumor_backup\datasets\train\masks",
    },
    "val": {
        "images": r"c:\Users\acer\Documents\brain_tumor_backup\datasets\val\images",
        "masks":  r"c:\Users\acer\Documents\brain_tumor_backup\datasets\val\masks",
    },
    "test": {
        "images": r"c:\Users\acer\Documents\brain_tumor_backup\datasets\test\images",
        "masks":  r"c:\Users\acer\Documents\brain_tumor_backup\datasets\test\masks",
    }
}

for split in out_dirs:
    os.makedirs(out_dirs[split]["images"], exist_ok=True)
    os.makedirs(out_dirs[split]["masks"], exist_ok=True)

all_images = []
for src_dir in src_dirs:
    for f in os.listdir(src_dir):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(src_dir, f))

train_imgs, temp_imgs = train_test_split(all_images, test_size=0.3, random_state=42, shuffle=True)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42, shuffle=True)

def save_split(image_paths, dst_img_dir, dst_mask_dir):
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)

        shutil.copy(img_path, os.path.join(dst_img_dir, filename))

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        black_mask = img * 0

        mask_name = name + ".png"
        cv2.imwrite(os.path.join(dst_mask_dir, mask_name), black_mask)

save_split(train_imgs, out_dirs["train"]["images"], out_dirs["train"]["masks"])
save_split(val_imgs, out_dirs["val"]["images"], out_dirs["val"]["masks"])
save_split(test_imgs, out_dirs["test"]["images"], out_dirs["test"]["masks"])

print("Done: split no_tumor into train/val/test with black masks.")