import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation


class DepthEstimator:
    def __init__(self, device="cuda"):
        print("Initializing depth estimator...")

        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas").to(device)
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            "Intel/dpt-hybrid-midas")

    
    def get_depth_map(self, image):
        original_size = image.size

        image = self.feature_extractor(
            images=image, return_tensors="pt").pixel_values.to("cuda")

        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=original_size[::-1],
            mode="bicubic",
            align_corners=False,
        )

        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

        return image
