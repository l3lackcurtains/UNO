import torch
from PIL import Image
from pathlib import Path
import time
from datetime import datetime
import os
from uno.flux.pipeline import UNOPipeline

class ImageGenerator:
    def __init__(self):
        # Set default model paths
        os.environ["FLUX_DEV"] = "./models/flux1-dev.safetensors"
        os.environ["AE"] = "./models/ae.safetensors"
        os.environ["T5"] = "./models/xflux_text_encoders"
        os.environ["CLIP"] = "./models/clip-vit-large-patch14"
        os.environ["LORA"] = "./models/dit_lora.safetensors"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required but not available")

        self.device = "cuda"
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the UNO pipeline"""
        self.pipeline = UNOPipeline(
            model_type="flux-dev",
            device=self.device,
            only_lora=True,
            lora_rank=512
        )

    def generate(self, prompt: str, height: int = 512, width: int = 512, model_type: str = "flux-dev"):
        """Generate an image from a prompt"""
        try:
            # Ensure dimensions are multiples of 16
            height = 16 * (height // 16)
            width = 16 * (width // 16)

            # Generate random seed if not provided
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

            # Generate the image
            output_image = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                guidance=4.0,
                num_steps=25,
                seed=seed,
                ref_imgs=[],  # Pass empty list instead of None
                pe='d'
            )

            return output_image

        except Exception as e:
            raise RuntimeError(f"Image generation failed: {str(e)}")

    def get_system_info(self):
        """Get system information including GPU status"""
        gpu_info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated": f"{torch.cuda.memory_allocated()/1024**3:.2f}GB" if torch.cuda.is_available() else None,
            "memory_cached": f"{torch.cuda.memory_reserved()/1024**3:.2f}GB" if torch.cuda.is_available() else None
        }
        
        return {
            "gpu": gpu_info,
            "model_type": "flux-dev",
            "device": self.device
        }
