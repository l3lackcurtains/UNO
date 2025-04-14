import torch
from PIL import Image
from uno.flux.pipeline import UNOPipeline
import time
from datetime import datetime
from pathlib import Path
import os

# Set default model paths - using correct model identifiers
os.environ["FLUX_DEV"] = "./models/flux1-dev.safetensors"
os.environ["AE"] = "./models/ae.safetensors"
os.environ["T5"] = "xlabs-ai/xflux_text_encoders"  # Use HF repo ID
os.environ["CLIP"] = "openai/clip-vit-large-patch14"  # Use HF repo ID
os.environ["LORA"] = "./models/dit_lora.safetensors"

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} sec"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"

def format_memory(bytes):
    return f"{bytes/1024**3:.2f} GB"

def generate(
    prompt: str,
    ref_image_paths: list[str] = None,
    width: int = 512,
    height: int = 512,
    guidance: float = 4.0,
    num_steps: int = 25,
    seed: int = -1,
    pe: str = 'd'
):
    total_start = time.time()
    
    print("Initializing pipeline...")
    pipeline = UNOPipeline(
        model_type="flux-dev",
        device="cuda" if torch.cuda.is_available() else "cpu",
        only_lora=True,
        lora_rank=512
    )
    init_time = time.time() - total_start
    
    if torch.cuda.is_available():
        init_memory = torch.cuda.memory_allocated()
    
    ref_imgs = []
    if ref_image_paths:
        ref_imgs = [Image.open(path) for path in ref_image_paths if path is not None]
    
    if seed == -1:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    
    generation_start = time.time()
    output_image = pipeline(
        prompt=prompt,
        width=width,
        height=height,
        guidance=guidance,
        num_steps=num_steps,
        seed=seed,
        ref_imgs=ref_imgs,
        pe=pe
    )
    generation_time = time.time() - generation_start
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{timestamp}.png"
    output_image.save(output_path)
    
    total_time = time.time() - total_start
    
    print(f"\nTiming:")
    print(f"Initialization: {format_time(init_time)}")
    print(f"Generation: {format_time(generation_time)}")
    print(f"Total time: {format_time(total_time)}")
    
    if torch.cuda.is_available():
        print(f"\nMemory:")
        print(f"After initialization: {format_memory(init_memory)}")
        print(f"Peak: {format_memory(peak_memory)}")
    
    return str(output_path)

if __name__ == "__main__":
    # Simple usage example
    output_path = generate(
        prompt="handsome woman in the city",
        width=512,
        height=512,
        guidance=4.0,
        num_steps=25,
        seed=-1
    )
    print(f"\nImage saved to: {output_path}")
