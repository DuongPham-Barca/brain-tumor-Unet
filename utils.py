import os
import numpy as np
from PIL import Image
import torch
from unet import UNet


class TumorSegmentationModel:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_channels=3, n_classes=1).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            self.model.load_state_dict(state["state_dict"], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)

        self.model.eval()

    def preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((256, 256))
        image_array = np.array(image).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_array - mean) / std

        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor, np.array(image)

    def predict_mask(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
            output = torch.sigmoid(output).squeeze(0).squeeze(0).cpu().numpy()
            mask = (output > 0.5).astype(np.uint8)
            return mask

    def overlay_mask_on_image(self, image_array, mask):
        overlay = image_array.copy()
        red = np.array([255, 0, 0], dtype=np.uint8)
        overlay[mask == 1] = red
        blended = (0.6 * image_array + 0.4 * overlay).astype(np.uint8)
        return blended


class TumorReportGenerator:
    def extract_tumor_features(self, mask, image_shape):
        tumor_pixels = int(np.sum(mask))
        total_pixels = int(image_shape[0] * image_shape[1])
        area_percent = (tumor_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        bbox = self._get_bounding_box(mask)
        return {
            "tumor_pixels": tumor_pixels,
            "area_percent": area_percent,
            "bbox": bbox,
            "image_shape": image_shape
        }

    def _get_bounding_box(self, mask):
        coords = np.argwhere(mask == 1)
        if coords.size == 0:
            return None
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        return [int(y0), int(x0), int(y1), int(x1)]

    def generate_llm_report(self, features):
        if features["bbox"] is None:
            return "No tumor detected in the image."
        return (
            f"A suspected tumor region was detected covering {features['area_percent']:.2f}% of the image. "
            f"The tumor area is approximately {features['tumor_pixels']} pixels."
        )

    def call_llm_api(self, prompt):
        return prompt
