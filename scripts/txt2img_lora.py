#!/usr/bin/env python3
"""
Inference script for Stable Diffusion with LoRA weights
Generates images using fine-tuned LoRA model
"""

import argparse
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import peft.tuners.lora
from PIL import Image
import logging

# Monkey-patch PEFT Linear to ignore the 'scale' argument passed by diffusers
original_forward = peft.tuners.lora.Linear.forward
def patched_forward(self, x: torch.Tensor, *args, **kwargs):
    kwargs.pop("scale", None)
    return original_forward(self, x, *args, **kwargs)
peft.tuners.lora.Linear.forward = patched_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAStableDiffusion:
    """Wrapper for Stable Diffusion with LoRA weights"""

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        ckpt: str = None,
        lora_weights: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        if ckpt and os.path.exists(ckpt):
            logger.info(f"Loading Stable Diffusion model from local checkpoint: {ckpt}")
            self.pipe = StableDiffusionPipeline.from_single_file(
                ckpt,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            logger.info(f"Loading Stable Diffusion model: {model_name}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

        # Load LoRA weights if provided
        if lora_weights:
            logger.info(f"Loading LoRA weights from: {lora_weights}")
            
            # Load text encoder LoRA if available
            text_encoder_lora_path = os.path.join(lora_weights, "text_encoder_lora")
            if os.path.exists(text_encoder_lora_path):
                logger.info("Loading text encoder LoRA")
                self.pipe.text_encoder = PeftModel.from_pretrained(
                    self.pipe.text_encoder,
                    text_encoder_lora_path,
                    device_map=device,
                )
            
            # Load UNet LoRA
            unet_lora_path = os.path.join(lora_weights, "unet_lora")
            if os.path.exists(unet_lora_path):
                logger.info("Loading UNet LoRA")
                self.pipe.unet = PeftModel.from_pretrained(
                    self.pipe.unet,
                    unet_lora_path,
                    device_map=device,
                )

        self.pipe = self.pipe.to(device)
        logger.info(f"Model loaded on {device}")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        lora_scale: float = 1.0,
        height: int = 512,
        width: int = 512,
        num_images_per_prompt: int = 1,
        seed: int = None,
    ) -> list:
        """
        Generate images with LoRA weights

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale (7.5 recommended)
            lora_scale: LoRA influence scale (0.5-1.5, 1.0 default)
            height: Image height
            width: Image width
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of PIL Image objects
        """
        if seed is not None:
            torch.manual_seed(seed)

        logger.info(f"Generating images with prompt: {prompt}")
        logger.info(f"LoRA scale: {lora_scale}")

        # Dynamically apply LoRA scale to PEFT layers
        for component in [self.pipe.unet, self.pipe.text_encoder]:
            for module in component.modules():
                if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
                    for adapter_name in module.scaling.keys():
                        alpha = module.lora_alpha[adapter_name] if isinstance(module.lora_alpha, dict) else module.lora_alpha
                        r = module.r[adapter_name] if isinstance(module.r, dict) else module.r
                        module.scaling[adapter_name] = (alpha / r) * lora_scale

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_images_per_prompt=num_images_per_prompt,
            )

        return result.images

    def save_images(self, images: list, output_dir: str = "outputs/lora_samples", prefix: str = "lora"):
        """Save generated images"""
        os.makedirs(output_dir, exist_ok=True)

        # Find next available index
        existing_files = list(Path(output_dir).glob(f"{prefix}_*.png"))
        next_idx = len(existing_files)

        saved_paths = []
        for i, image in enumerate(images):
            output_path = os.path.join(output_dir, f"{prefix}_{next_idx + i}.png")
            image.save(output_path)
            saved_paths.append(output_path)
            logger.info(f"Saved: {output_path}")

        return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion + LoRA")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights directory",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base model name",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="LoRA influence scale (0.5-1.5)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Image width",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/lora_samples",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16"],
        default="float32",
        help="Model dtype",
    )

    args = parser.parse_args()

    # Setup dtype
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Initialize model
    logger.info("Initializing model with LoRA weights...")
    model = LoRAStableDiffusion(
        model_name=args.model_name,
        ckpt=args.ckpt,
        lora_weights=args.lora_weights,
        dtype=dtype,
    )

    # Generate images
    images = model.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        lora_scale=args.lora_scale,
        height=args.height,
        width=args.width,
        num_images_per_prompt=args.num_images,
        seed=args.seed,
    )

    # Save images
    saved_paths = model.save_images(images, args.output_dir, prefix="lora")
    logger.info(f"Saved {len(saved_paths)} images")

    # Display summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Prompt: {args.prompt}")
    print(f"LoRA Weights: {args.lora_weights}")
    print(f"Guidance Scale: {args.guidance_scale}")
    print(f"LoRA Scale: {args.lora_scale}")
    print(f"Output Directory: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
