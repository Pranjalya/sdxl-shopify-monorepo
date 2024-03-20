# Based on https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
import os
import requests

import cv2
import math
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


class Upsampler:
    def __init__(self, weight_path="weights/RealESRGAN_x2plus.pth", device="cuda"):
        print("Initializing upscaler...")
        if not os.path.exists(weight_path):
            weight_dir, _ = os.path.split(weight_path)
            os.makedirs(weight_dir, exist_ok=True)
            url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            response = requests.get(url)
            with open(weight_path, 'wb') as f:
                f.write(response.content)
        
        self.UPSCALE_PIXEL_THRESHOLD = 1
        self.DOWNSCALE_PIXEL_THRESHOLD = 1
        
        model = RRDBNet(
            num_in_ch=3, 
            num_out_ch=3, 
            num_feat=64, 
            num_block=23, 
            num_grow_ch=32, 
            scale=2
        )

        self.upsampler = RealESRGANer(scale=2, model_path=weight_path, model=model, device=device)
        
    
    def upscale(self, image):
        original_numpy = np.array(image)
        original_opencv = cv2.cvtColor(original_numpy, cv2.COLOR_RGB2BGR)

        output, _ = self.upsampler.enhance(original_opencv, outscale=2)
        upscaled = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        return upscaled

    
    def maybe_upscale(self, original, megapixels=1.0):
        original_width, original_height = original.size
        original_pixels = original_width * original_height
        target_pixels = megapixels * 1024 * 1024

        if (original_pixels < target_pixels):
            scale_by = math.sqrt(target_pixels / original_pixels)
            target_width = original_width * scale_by
            target_height = original_height * scale_by

            if (target_width - original_width >= 1 or target_height - original_height >= self.UPSCALE_PIXEL_THRESHOLD):
                print("Upscaling...")
                upscaled = self.upscale(original)
                print("Upscaled size:", upscaled.size)
                return upscaled

        print("Not upscaling")
        return original


    def maybe_downscale(self, original, megapixels=1.0):
        original_width, original_height = original.size
        original_pixels = original_width * original_height
        target_pixels = megapixels * 1024 * 1024

        if (original_pixels > target_pixels):
            scale_by = math.sqrt(target_pixels / original_pixels)
            target_width = original_width * scale_by
            target_height = original_height * scale_by

            if (original_width - target_width >= 1 or original_height - target_height >= self.DOWNSCALE_PIXEL_THRESHOLD):
                print("Downscaling...")
                target_width = round(target_width)
                target_height = round(target_height)
                downscaled = original.resize(
                    (target_width, target_height), 
                    Image.LANCZOS
                )
                print("Downscaled size:", downscaled.size)
                return downscaled

        print("Not downscaling")
        return original

    def ensure_resolution(self, original, megapixels=1.0):
        return self.maybe_downscale(self.maybe_upscale(original, megapixels), megapixels)