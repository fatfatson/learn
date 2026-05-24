"""
Text-to-Video Generation using CogVideoX / Wan2.1

Supports running on Apple Silicon (MPS) and NVIDIA GPU (CUDA).
Models will be automatically downloaded from HuggingFace on first run.

Usage:
    python src/text2video.py --prompt "A cat running on grass"
    python src/text2video.py --prompt "A cat running on grass" --model cogvideox-5b
    python src/text2video.py --prompt "A cat running on grass" --model wan2.1-1.3b
    python src/text2video.py --prompt "一只小猫在草地上奔跑" --num-frames 49
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "text2video"


# Available models configuration
MODEL_CONFIGS = {
    "cogvideox-2b": {
        "repo_id": "THUDM/CogVideoX-2b",
        "dtype": torch.float16,
        "description": "CogVideoX 2B - lightweight, fast, good quality",
    },
    "cogvideox-5b": {
        "repo_id": "THUDM/CogVideoX-5b",
        "dtype": torch.float16,
        "description": "CogVideoX 5B - higher quality, slower",
    },
    "wan2.1-1.3b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "dtype": torch.float16,
        "description": "Wan2.1 1.3B - lightweight, good for quick tests",
    },
    "wan2.1-14b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-14B",
        "dtype": torch.float16,
        "description": "Wan2.1 14B - SOTA quality, requires more memory",
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


def load_cogvideox_pipeline(repo_id, dtype, device, low_memory=False):
    """Load CogVideoX pipeline with MPS/CUDA compatibility."""
    from diffusers import CogVideoXPipeline

    print(f"Loading CogVideoX model: {repo_id}")
    print(f"Device: {device}, dtype: {dtype}")
    if low_memory:
        print("  Low memory mode: ENABLED")

    # For MPS, we need float32 for stability (float16 has issues on some MPS ops)
    load_dtype = torch.float32 if device == "mps" else dtype

    pipe = CogVideoXPipeline.from_pretrained(
        repo_id,
        torch_dtype=load_dtype,
    )

    # Move to device with explicit dtype to avoid MPS float64 conversion errors
    pipe = pipe.to(device, dtype=load_dtype)

    # MPS-specific optimizations
    if device == "mps":
        # Enable attention slicing to reduce memory pressure
        try:
            if low_memory:
                # More aggressive slicing: split into smaller chunks
                pipe.enable_attention_slicing("max")
            else:
                pipe.enable_attention_slicing()
        except Exception:
            pass
        # Enable VAE slicing if available (reduces peak memory during decoding)
        try:
            pipe.enable_vae_slicing()
        except Exception:
            pass
        # Clear MPS cache before generation
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


def load_wan_pipeline(repo_id, dtype, device, low_memory=False):
    """Load Wan2.1 pipeline."""
    from diffusers import WanPipeline

    print(f"Loading Wan2.1 model: {repo_id}")
    print(f"Device: {device}, dtype: {dtype}")
    if low_memory:
        print("  Low memory mode: ENABLED")

    load_dtype = torch.float32 if device == "mps" else dtype

    pipe = WanPipeline.from_pretrained(
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
    
    Supports: '720p' (1280x720), '480p' (854x480), or custom 'WxH' (e.g., '1920x1080').
    Returns None if no resizing needed (native resolution).
    """
    if resolution_str is None:
        return None  # use native resolution

    # Model native resolutions
    native = {
        "cogvideox-2b": (720, 480),
        "cogvideox-5b": (720, 480),
        "wan2.1-1.3b": (832, 480),
        "wan2.1-14b": (1280, 720),
    }
    native_w, native_h = native.get(model_name, (720, 480))

    res = resolution_str.strip().lower()
    if res == "720p":
        return (1280, 720)
    elif res == "480p":
        return (854, 480)
    elif "x" in res:
        parts = res.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
    
    print(f"WARNING: Cannot parse resolution '{resolution_str}'. Using native resolution.")
    return None


def resize_frames(frames, target_width, target_height):
    """Resize video frames to target resolution using PIL."""
    from PIL import Image

    resized = []
    for frame in frames:
        # Convert to PIL Image if needed
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            img = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            img = frame
        elif isinstance(frame, torch.Tensor):
            arr = frame.cpu().numpy().astype(np.uint8)
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(np.array(frame, dtype=np.uint8))

        # Resize with LANCZOS (high quality) interpolation
        if img.size != (target_width, target_height):
            img = img.resize((target_width, target_height), Image.LANCZOS)
        resized.append(np.array(img))

    print(f"  Resized frames: {target_width}x{target_height}")
    return resized


def generate_video(pipe, prompt, model_name, num_frames, num_inference_steps, device):
    """Generate video from text prompt."""
    print(f"\nGenerating video...")
    print(f"  Prompt: {prompt}")
    print(f"  Model: {model_name}")
    print(f"  Frames: {num_frames}")
    print(f"  Steps: {num_inference_steps}")

    # Clear MPS cache before generation
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    start_time = time.time()

    # MPS fallback: catch float16/float64 errors and retry with float32
    # Also catch OOM errors and retry with fewer frames
    try:
        if "cogvideox" in model_name:
            video_frames = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=6.0,
            ).frames[0]
        else:
            # Wan2.1
            video_frames = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
            ).frames[0]
    except RuntimeError as e:
        err_msg = str(e)
        if "MPS" in err_msg and ("float16" in err_msg or "float64" in err_msg):
            print(f"\nMPS dtype error detected, retrying with float32...")
            pipe = pipe.to(dtype=torch.float32)
            if "cogvideox" in model_name:
                video_frames = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                    guidance_scale=6.0,
                ).frames[0]
            else:
                video_frames = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=num_frames,
                ).frames[0]
        elif "out of memory" in err_msg:
            print(f"\nMPS out of memory! Error: {err_msg}")
            half_frames = max(num_frames // 2, 9)  # minimum 9 frames
            print(f"Retrying with {half_frames} frames (was {num_frames})...")
            # Clear cache and retry
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            if "cogvideox" in model_name:
                video_frames = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=half_frames,
                    guidance_scale=6.0,
                ).frames[0]
            else:
                video_frames = pipe(
                    prompt=prompt,
                    num_videos_per_prompt=1,
                    num_inference_steps=num_inference_steps,
                    num_frames=half_frames,
                ).frames[0]
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

    return video_frames


def save_video(frames, output_path, fps=8):
    """Save video frames to MP4 file."""
    import imageio
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert frames to numpy arrays
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()

    processed_frames = []
    for frame in frames:
        # Handle PIL Image objects (CogVideoX returns PIL Images)
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        elif isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        # Ensure numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        # Normalize to [0, 255] uint8
        if frame.dtype in (np.float32, np.float64):
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        elif frame.dtype == np.uint8:
            pass  # already in correct format
        else:
            frame = frame.astype(np.uint8)
        processed_frames.append(frame)

    imageio.mimwrite(str(output_path), processed_frames, fps=fps, codec="libx264")
    print(f"Video saved to: {output_path}")
    return output_path


def save_gif(frames, output_path, fps=8):
    """Save video frames as GIF (smaller file, wider compatibility)."""
    import imageio
    from PIL import Image

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()

    processed_frames = []
    for frame in frames:
        # Handle PIL Image objects (CogVideoX returns PIL Images)
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
        elif isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        # Ensure numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if frame.dtype in (np.float32, np.float64):
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        elif frame.dtype == np.uint8:
            pass  # already in correct format
        else:
            frame = frame.astype(np.uint8)
        processed_frames.append(frame)

    imageio.mimwrite(str(output_path), processed_frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Text-to-Video Generation")
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Text prompt for video generation (supports Chinese and English)",
    )
    parser.add_argument(
        "--model", type=str, default="cogvideox-2b",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: cogvideox-2b)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=49,
        help="Number of video frames (default: 49, ~6s at 8fps)",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50,
        help="Number of denoising steps (default: 50)",
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Output video FPS (default: 8)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: output/text2video/<timestamp>.mp4)",
    )
    parser.add_argument(
        "--format", type=str, default="mp4", choices=["mp4", "gif", "both"],
        help="Output format (default: mp4)",
    )
    parser.add_argument(
        "--low-memory", action="store_true", default=False,
        help="Enable low memory mode (aggressive memory optimizations, may be slower)",
    )
    parser.add_argument(
        "--resolution", type=str, default=None,
        help="Output video resolution. Options: '720p' (1280x720), '480p' (854x480), "
             "or custom WxH (e.g., '1920x1080'). Default: native model resolution",
    )

    args = parser.parse_args()

    # Set MPS memory environment variables for better memory management
    device = get_device()
    if device == "mps":
        # Disable MPS memory upper limit to prevent OOM on large allocations
        # WARNING: This may cause system instability if other apps need memory
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    # Validate model config
    config = MODEL_CONFIGS[args.model]
    print(f"Device: {device}")
    print(f"Model: {config['description']}")

    if device == "cpu":
        print("WARNING: No GPU detected. Video generation will be extremely slow on CPU.")
        print("Consider using a machine with CUDA or Apple Silicon GPU.")

    # MPS memory warning
    if device == "mps" and not args.low_memory and args.num_frames > 25:
        print(f"\n⚠️  MPS Memory Warning:")
        print(f"    Generating {args.num_frames} frames on MPS may cause OOM.")
        print(f"    If you hit 'MPS backend out of memory', try:")
        print(f"      1. Add --low-memory flag")
        print(f"      2. Reduce --num-frames (e.g., 13 or 9)")
        print(f"      3. Reduce --num-inference-steps (e.g., 25)")
        print()

    # Load model
    if "cogvideox" in args.model:
        pipe = load_cogvideox_pipeline(config["repo_id"], config["dtype"], device, args.low_memory)
    else:
        pipe = load_wan_pipeline(config["repo_id"], config["dtype"], device, args.low_memory)

    # Generate
    video_frames = generate_video(
        pipe, args.prompt, args.model,
        args.num_frames, args.num_inference_steps, device,
    )

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.prompt[:30])
        output_path = OUTPUT_DIR / f"{safe_prompt}_{timestamp}"

    # Resize frames if custom resolution requested
    target_res = parse_resolution(args.resolution, args.model)
    if target_res is not None:
        target_w, target_h = target_res
        print(f"  Resizing to: {target_w}x{target_h}")
        video_frames = resize_frames(video_frames, target_w, target_h)

    # Save
    if args.format in ("mp4", "both"):
        save_video(video_frames, output_path.with_suffix(".mp4"), fps=args.fps)
    if args.format in ("gif", "both"):
        save_gif(video_frames, output_path.with_suffix(".gif"), fps=args.fps)

    print("\nDone!")

if __name__ == "__main__":
    main()
