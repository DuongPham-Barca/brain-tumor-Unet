from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from django.conf import settings
from django.shortcuts import render
from PIL import Image

from unet import UNet


MODEL_PATH = Path(settings.BASE_DIR) / "best_model.pt"
IMAGE_SIZE = (256, 256)
# normalization cho 1 channel (grayscale)
MEAN = torch.tensor([0.485], dtype=torch.float32).view(1, 1, 1)
STD = torch.tensor([0.229], dtype=torch.float32).view(1, 1, 1)


@lru_cache(maxsize=1)
def load_model() -> tuple[UNet, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def to_base64(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def preprocess_image(image_rgb: np.ndarray) -> torch.Tensor:
    # Chuyển sang grayscale (model hiện tại nhận 1 channel)
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(image_gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float().unsqueeze(0) / 255.0  # shape (1, H, W)
    return (tensor - MEAN) / STD


def home(request):
    context: dict[str, str] = {}

    if request.method == "POST" and request.FILES.get("image"):
        uploaded_file = request.FILES["image"]
        raw = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)

        if image is None:
            context["error"] = "Khong the doc hinh anh da tai len."
            return render(request, "segmentation/index.html", context)

        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = preprocess_image(image_rgb).unsqueeze(0)

        model, device = load_model()
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            prediction = torch.sigmoid(model(image_tensor)).squeeze().cpu().numpy()

        prediction_mask = (prediction > 0.5).astype(np.uint8)
        prediction_mask = cv2.resize(
            prediction_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        overlay = image_rgb.copy()
        overlay[prediction_mask == 1] = [255, 0, 0]
        blended = (image_rgb * 0.6 + overlay * 0.4).astype(np.uint8)

        context.update(
            {
                "original_image": to_base64(image_rgb),
                "mask_image": to_base64((prediction_mask * 255).astype(np.uint8)),
                "overlay_image": to_base64(blended),
                "filename": uploaded_file.name,
            }
        )

    return render(request, "segmentation/index.html", context)