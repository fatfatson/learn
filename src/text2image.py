"""
Text-to-Image Generation using Multiple Models

Supported models on Apple Silicon (MPS) and NVIDIA GPU (CUDA):
  - sd15: Stable Diffusion v1.5 (0.9B) - Classic, stable, lightweight (~4GB)
  - sdxl: Stable Diffusion XL (2.6B) - High quality, moderate size (~6.5GB)
  - hunyuandit: HunyuanDiT v1.1 (1.5B) - DiT architecture, good quality
  - hunyuanimage: HunyuanImage 2.1 - Latest Hunyuan model, higher quality

Models will be automatically downloaded from HuggingFace on first run.
Set HF_ENDPOINT=https://hf-mirror.com for faster downloads in China.

Usage:
    python src/text2image.py --prompt "A cute cat sitting on grass"
    python src/text2image.py --prompt "一只可爱的猫咪" --model sd15
    python src/text2image.py --prompt "一只可爱的猫咪" --model sdxl --resolution 1280x720
    python src/text2image.py --prompt "一只可爱的猫咪" --model hunyuandit
    python src/text2image.py --prompt "A sunset over mountains" --num-inference-steps 25
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "text2image"

# Available models configuration
MODEL_CONFIGS = {
    "sd15": {
        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "dtype": torch.float16,
        "description": "Stable Diffusion v1.5 - Classic, stable, lightweight (~4GB)",
        "native_resolution": (512, 512),
        "pipeline_class": "StableDiffusionPipeline",
        "default_guidance_scale": 7.5,
    },
    "sdxl": {
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "dtype": torch.float16,
        "description": "Stable Diffusion XL - High quality, moderate size (~6.5GB)",
        "native_resolution": (1024, 1024),
        "pipeline_class": "StableDiffusionXLPipeline",
        "default_guidance_scale": 7.5,
    },
    "hunyuandit": {
        "repo_id": "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        "dtype": torch.float16,
        "description": "HunyuanDiT v1.1 1.5B - DiT architecture, good quality, moderate memory",
        "native_resolution": (1024, 1024),
        "pipeline_class": "HunyuanDiTPipeline",
        "default_guidance_scale": 5.0,
    },
    "hunyuanimage": {
        "repo_id": "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
        "dtype": torch.float16,
        "description": "HunyuanImage 2.1 - latest model, higher quality, more memory",
        "native_resolution": (1024, 1024),
        "pipeline_class": "HunyuanImagePipeline",
        "default_guidance_scale": 5.0,
    },
}


def get_device():
    """Detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_pipeline(config, device, low_memory=False):
    """Load a diffusion pipeline with MPS/CUDA compatibility.

    Uses the pipeline_class from model config to dynamically import the
    correct diffusers pipeline class.
    """
    import diffusers

    repo_id = config["repo_id"]
    pipeline_class_name = config["pipeline_class"]
    PipelineClass = getattr(diffusers, pipeline_class_name)

    print(f"Loading model: {repo_id}")
    print(f"Pipeline: {pipeline_class_name}")
    print(f"Device: {device}, dtype: {config['dtype']}")
    if low_memory:
        print("  Low memory mode: ENABLED")

    # For MPS, we need float32 for stability (float16 has issues on some MPS ops)
    load_dtype = torch.float32 if device == "mps" else config["dtype"]

    # Suppress harmless "UNEXPECTED" key warnings during loading.
    # These occur when the model repo contains extra weights (e.g., safety_checker)
    # that don't match the pipeline class — this is expected and harmless.
    import logging
    logging.getLogger("diffusers.models.modeling_utils").setLevel(logging.ERROR)

    pipe = PipelineClass.from_pretrained(
        repo_id,
        torch_dtype=load_dtype,
    )

    # Move to device with explicit dtype to avoid MPS float64 conversion errors
    pipe = pipe.to(device, dtype=load_dtype)

    # MPS-specific optimizations
    if device == "mps":
        try:
            if low_memory:
                pipe.enable_attention_slicing("max")
            else:
                pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # CUDA optimizations
    if device == "cuda":
        if low_memory:
            try:
                pipe.enable_attention_slicing("max")
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass

    return pipe


def parse_resolution(resolution_str, model_name):
    """Parse resolution string and return (width, height).

    Supports: '1080p' (1920x1080), '720p' (1280x720), '480p' (854x480),
    or custom 'WxH' (e.g., '512x512').
    Returns None if no resizing needed (native resolution).
    """
    if resolution_str is None:
        return None  # use native resolution

    # Model native resolutions
    native = {
        "sd15": (512, 512),
        "sdxl": (1024, 1024),
        "hunyuandit": (1024, 1024),
        "hunyuanimage": (1024, 1024),
    }
    native_w, native_h = native.get(model_name, (1024, 1024))

    res = resolution_str.strip().lower()
    if res == "1080p":
        return (1920, 1080)
    elif res == "720p":
        return (1280, 720)
    elif res == "480p":
        return (854, 480)
    elif "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))

    print(f"WARNING: Cannot parse resolution '{resolution_str}'. Using native resolution.")
    return None


def check_prompt_tokens(pipe, prompt, model_name):
    """Check if prompt exceeds the text encoder's token limit and warn.

    CLIP-based models (sd15, sdxl) have a 77-token limit. Chinese text
    is especially prone to exceeding this because each CJK character
    typically consumes 2-3 tokens.
    """
    # Token limits per model family
    TOKEN_LIMITS = {
        "sd15": 77,
        "sdxl": 77,  # dual encoder, each still 77
        "hunyuandit": 256,
        "hunyuanimage": 256,
    }
    limit = TOKEN_LIMITS.get(model_name, 77)

    # Get tokenizer from pipeline
    tokenizer = None
    if hasattr(pipe, 'tokenizer'):
        tokenizer = pipe.tokenizer
    elif hasattr(pipe, 'tokenizers') and pipe.tokenizers:
        tokenizer = pipe.tokenizers[0]

    if tokenizer is None:
        # Cannot check, skip
        return

    try:
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        num_tokens = len(token_ids)
    except Exception:
        # Fallback: rough estimate for CJK text
        num_tokens = len(prompt) * 2  # rough upper bound
        limit = TOKEN_LIMITS.get(model_name, 77)

    if num_tokens > limit:
        print(f"\n⚠️  WARNING: Prompt has {num_tokens} tokens, exceeding the {limit}-token limit!")
        print(f"    The text beyond {limit} tokens will be TRUNCATED and IGNORED.")
        print(f"    Tips to fix:")

        if model_name in ("sd15", "sdxl"):
            print(f"      1. Shorten your prompt (remove less important details)")
            if model_name == "sd15":
                print(f"      2. Use English prompts (more token-efficient for CLIP)")
            print(f"      3. Switch to --model hunyuandit (supports 256 tokens + native Chinese)")
            print(f"      4. Switch to --model hunyuanimage (supports 256 tokens + native Chinese)")

        # Try to show what gets truncated
        if tokenizer is not None:
            try:
                truncated_ids = tokenizer.encode(prompt, add_special_tokens=True, max_length=limit, truncation=True)
                full_text = tokenizer.decode(token_ids[1:-1], skip_special_tokens=True)  # remove BOS/EOS
                truncated_text = tokenizer.decode(truncated_ids[1:-1], skip_special_tokens=True)
                if len(full_text) > len(truncated_text):
                    lost_part = full_text[len(truncated_text):].strip()
                    if lost_part:
                        print(f"\n    ❌ Truncated part: \"{lost_part}\"")
            except Exception:
                pass
        print()


def generate_image(pipe, prompt, model_name, num_inference_steps, device, guidance_scale=7.5, negative_prompt=None, generator=None):
    """Generate image from text prompt."""
    print(f"\nGenerating image...")
    print(f"  Prompt: {prompt}")
    if negative_prompt:
        print(f"  Negative prompt: {negative_prompt}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {num_inference_steps}")
    print(f"  Guidance scale: {guidance_scale}")

    # Check prompt token length before generation
    check_prompt_tokens(pipe, prompt, model_name)

    # Clear MPS cache before generation
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    start_time = time.time()

    # MPS fallback: catch float16/float64 errors and retry with float32
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=1,
            generator=generator,
        ).images[0]
    except RuntimeError as e:
        err_msg = str(e)
        if "MPS" in err_msg and ("float16" in err_msg or "float64" in err_msg):
            print(f"\nMPS dtype error detected, retrying with float32...")
            pipe = pipe.to(dtype=torch.float32)
            image = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                generator=generator,
            ).images[0]
        elif "out of memory" in err_msg:
            print(f"\nOut of memory! Error: {err_msg}")
            print("Try: 1) Add --low-memory flag, 2) Reduce --num-inference-steps")
            raise
        else:
            raise

    # Clear cache after generation
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    elapsed = time.time() - start_time
    print(f"  Generation time: {elapsed:.1f}s")

    return image


def resize_image(image, target_width, target_height):
    """Resize a PIL Image to target resolution."""
    from PIL import Image

    if isinstance(image, Image.Image):
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.LANCZOS)
        print(f"  Resized to: {target_width}x{target_height}")
        return image
    else:
        # numpy array
        img = Image.fromarray(image)
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.LANCZOS)
        print(f"  Resized to: {target_width}x{target_height}")
        return np.array(img)


def save_image(image, output_path):
    """Save image to file. Supports PIL Image or numpy array."""
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        img = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        arr = image.cpu().numpy().astype(np.uint8)
        img = Image.fromarray(arr)
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.fromarray(np.array(image, dtype=np.uint8))

    img.save(str(output_path))
    print(f"Image saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Text-to-Image Generation with Multiple Models")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for image generation (supports Chinese and English)",
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=None,
        help="Negative prompt (what to avoid in the image)",
    )
    parser.add_argument(
        "--model", type=str, default="sd15",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: sd15)",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: output/text2image/<timestamp>.png)",
    )
    parser.add_argument(
        "--format", type=str, default="png", choices=["png", "jpg", "webp"],
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--low-memory", action="store_true", default=False,
        help="Enable low memory mode (aggressive memory optimizations)",
    )
    parser.add_argument(
        "--resolution", type=str, default=None,
        help="Output image resolution. Options: '1080p' (1920x1080), '720p' (1280x720), "
             "'480p' (854x480), or custom WxH (e.g., '512x512'). Default: native model resolution",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--hf-mirror", action="store_true", default=False,
        help="Use hf-mirror.com endpoint for faster downloads in China",
    )

    args = parser.parse_args()

    # Set HuggingFace mirror endpoint if requested or not already set
    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"HF mirror enabled: {os.environ['HF_ENDPOINT']}")
    elif "HF_ENDPOINT" not in os.environ:
        # Auto-detect: hint user if download seems slow
        print("Tip: Add --hf-mirror flag for faster downloads in China")

    # Set MPS memory environment variables for better memory management
    device = get_device()
    if device == "mps":
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    # Validate model config
    config = MODEL_CONFIGS[args.model]
    print(f"Device: {device}")
    print(f"Model: {config['description']}")

    if device == "cpu":
        print("WARNING: No GPU detected. Image generation will be extremely slow on CPU.")
        print("Consider using a machine with CUDA or Apple Silicon GPU.")

    # MPS memory warning for larger models
    if device == "mps" and not args.low_memory and args.model in ("hunyuanimage", "sdxl"):
        print(f"\n⚠️  MPS Memory Warning:")
        print(f"    {args.model} is a larger model and may cause OOM on MPS.")
        print(f"    If you hit 'MPS backend out of memory', try:")
        print(f"      1. Add --low-memory flag")
        print(f"      2. Reduce --num-inference-steps (e.g., 25)")
        print(f"      3. Switch to --model sd15 (lighter model)")
        print()

    # Set random seed for reproducibility
    if args.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        print(f"  Seed: {args.seed}")
    else:
        generator = None

    # Load model
    pipe = load_pipeline(config, device, args.low_memory)

    # Generate
    image = generate_image(
        pipe, args.prompt, args.model,
        args.num_inference_steps, device,
        guidance_scale=config["default_guidance_scale"],
        negative_prompt=args.negative_prompt,
        generator=generator,
    )

    # Resize if custom resolution requested
    target_res = parse_resolution(args.resolution, args.model)
    if target_res is not None:
        target_w, target_h = target_res
        image = resize_image(image, target_w, target_h)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.prompt[:30])
        output_path = OUTPUT_DIR / f"{safe_prompt}_{timestamp}"

    # Save
    save_image(image, output_path.with_suffix(f".{args.format}"))

    print("\nDone!")


if __name__ == "__main__":
    main()
