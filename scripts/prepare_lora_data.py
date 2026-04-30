#!/usr/bin/env python3
"""
Data preparation script for LoRA fine-tuning
Preprocesses images: resizes, crops, and prepares dataset structure
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def resize_and_center_crop(image: Image.Image, size: int = 512) -> Image.Image:
    """Resize and center crop image to square"""
    # Resize maintaining aspect ratio
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    
    # Center crop
    left = (image.width - size) // 2
    top = (image.height - size) // 2
    right = left + size
    bottom = top + size
    
    image = image.crop((left, top, right, bottom))
    return image


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    size: int = 512,
    instance_prompt: str = "[name]",
    skip_existing: bool = False,
):
    """
    Prepare dataset for LoRA fine-tuning
    
    Args:
        input_dir: Directory with original images
        output_dir: Directory to save processed images
        size: Target image size
        instance_prompt: Instance prompt for captions
        skip_existing: Skip existing files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in input_path.glob("*") if f.suffix.lower() in image_extensions]

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(images)} images to process")

    processed_count = 0
    captions = {}

    # Process each image
    for image_file in tqdm(images, desc="Processing images"):
        output_file = output_path / image_file.name

        # Skip if already processed
        if output_file.exists() and skip_existing:
            logger.info(f"Skipping existing: {image_file.name}")
            continue

        try:
            # Open and process image
            image = Image.open(image_file).convert("RGB")
            image = resize_and_center_crop(image, size)

            # Save processed image
            image.save(output_file, quality=95)
            logger.info(f"Saved: {output_file}")

            # Generate caption
            filename = image_file.name
            caption = instance_prompt + " " + filename.replace("_", " ").replace("-", " ")
            captions[filename] = caption

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {image_file}: {e}")
            continue

    # Save captions.json
    captions_file = output_path / "captions.json"
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)
    logger.info(f"Saved captions to: {captions_file}")

    logger.info(f"Successfully processed {processed_count}/{len(images)} images")
    logger.info(f"Output directory: {output_path}")


def create_directory_structure(base_dir: str, person_name: str):
    """Create recommended directory structure"""
    directories = [
        base_dir,
        os.path.join(base_dir, "lora_training"),
        os.path.join(base_dir, "lora_training", person_name),
        os.path.join(base_dir, "lora_training", person_name, "raw"),
        os.path.join(base_dir, "lora_training", person_name, "processed"),
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created: {dir_path}")

    # Create README
    readme_path = os.path.join(base_dir, "lora_training", person_name, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""# LoRA Training Data for {person_name}

## Directory Structure
- `raw/`: Original images (place your images here)
- `processed/`: Preprocessed images ready for training

## Setup Instructions
1. Place your images in the `raw/` directory
2. Run preprocessing: `python scripts/prepare_lora_data.py --input_dir data/lora_training/{person_name}/raw --output_dir data/lora_training/{person_name}/processed`
3. Start training: `python scripts/train_lora.py --instance_data_dir data/lora_training/{person_name}/processed --instance_prompt "a photo of [{person_name}_token]"`

## Guidelines
- Collect 10-20 high-quality images
- Vary: lighting, angles, poses, clothing, backgrounds
- Minimum resolution: 512x512
- Aspect ratio: mostly square (0.8-1.2)

## Expected Results
Training typically takes 20-60 minutes on RTX 3090
Checkpoint saved every 100 steps
Final model saved in `lora_weights/`
""")
    logger.info(f"Created: {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for LoRA fine-tuning")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with original images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed images",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Target image size",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="[token]",
        help="Instance prompt for captions",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip existing files",
    )
    parser.add_argument(
        "--create_structure",
        type=str,
        default=None,
        help="Create directory structure for given person name",
    )

    args = parser.parse_args()

    if args.create_structure:
        logger.info(f"Creating directory structure for {args.create_structure}")
        create_directory_structure("data", args.create_structure)

    # Prepare dataset
    logger.info("="*60)
    logger.info("Data Preparation for LoRA Fine-tuning")
    logger.info("="*60)

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        size=args.size,
        instance_prompt=args.instance_prompt,
        skip_existing=args.skip_existing,
    )

    logger.info("="*60)
    logger.info("Data preparation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
