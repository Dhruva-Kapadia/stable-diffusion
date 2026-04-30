#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Stable Diffusion
Uses diffusers and peft libraries for efficient fine-tuning
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import get_peft_model, LoraConfig, TaskType

# ------------- MONKEY PATCH FOR PEFT -------------
import peft.tuners.lora
_orig_linear_forward = getattr(peft.tuners.lora.Linear, "forward", None)
if _orig_linear_forward:
    def _patched_linear_forward(self, x, *args, **kwargs):
        kwargs.pop('scale', None)
        return _orig_linear_forward(self, x, *args, **kwargs)
    peft.tuners.lora.Linear.forward = _patched_linear_forward

_orig_conv2d_forward = getattr(peft.tuners.lora.Conv2d, "forward", None) if hasattr(peft.tuners.lora, 'Conv2d') else None
if _orig_conv2d_forward:
    def _patched_conv2d_forward(self, x, *args, **kwargs):
        kwargs.pop('scale', None)
        return _orig_conv2d_forward(self, x, *args, **kwargs)
    peft.tuners.lora.Conv2d.forward = _patched_conv2d_forward
# -------------------------------------------------
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DreamBoothDataset(Dataset):
    """Dataset for LoRA fine-tuning with DreamBooth-style training"""

    def __init__(
        self,
        instance_data_dir: str,
        instance_prompt: str,
        class_data_dir: Optional[str] = None,
        class_prompt: Optional[str] = None,
        size: int = 512,
        center_crop: bool = True,
        captions_file: Optional[str] = None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        # Load instance images
        self.instance_images = list(Path(instance_data_dir).glob("*.jpg"))
        self.instance_images.extend(Path(instance_data_dir).glob("*.png"))
        self.instance_images.extend(Path(instance_data_dir).glob("*.jpeg"))

        logger.info(f"Found {len(self.instance_images)} instance images")

        # Load captions if available
        self.captions = {}
        if captions_file and os.path.exists(captions_file):
            with open(captions_file, 'r') as f:
                self.captions = json.load(f)

        # Load class/regularization images
        self.class_images = []
        if class_data_dir:
            self.class_images = list(Path(class_data_dir).glob("*.jpg"))
            self.class_images.extend(Path(class_data_dir).glob("*.png"))
            self.class_images.extend(Path(class_data_dir).glob("*.jpeg"))
            logger.info(f"Found {len(self.class_images)} class images for regularization")

        # Image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, index):
        # Get instance image
        img_path = self.instance_images[index]
        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        # Get caption
        caption_key = img_path.name
        if caption_key in self.captions:
            prompt = self.captions[caption_key]
        else:
            prompt = self.instance_prompt

        # Get class image for regularization (if available)
        class_image = None
        class_prompt = self.class_prompt
        if self.class_images:
            class_idx = index % len(self.class_images)
            class_img_path = self.class_images[class_idx]
            class_image = Image.open(class_img_path).convert("RGB")
            class_image = self.image_transforms(class_image)

        res = {
            "instance_images": image,
            "instance_prompts": prompt,
        }
        if class_image is not None:
            res["class_images"] = class_image
            if class_prompt is not None:
                res["class_prompts"] = class_prompt
        return res


class LoRAFineTuner(pl.LightningModule):
    """PyTorch Lightning module for LoRA fine-tuning"""

    def __init__(
        self,
        model_name: str = "runwayml/stable-diffusion-v1-5",
        learning_rate: float = 1e-4,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_text_encoder: bool = True,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # Load models
        logger.info(f"Loading model: {model_name}")
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

        # Freeze VAE and text encoder
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        self.unet = self.pipeline.unet
        self.tokenizer = self.pipeline.tokenizer
        self.scheduler = self.pipeline.scheduler

        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        # Setup LoRA for UNet
        logger.info("Setting up LoRA for UNet")
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.1,
            bias="none",
        )
        self.unet = get_peft_model(self.unet, unet_lora_config)

        # Setup LoRA for Text Encoder (optional)
        if lora_text_encoder:
            logger.info("Setting up LoRA for Text Encoder")
            text_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, text_lora_config)

        logger.info("LoRA setup complete")
        self.print_trainable_parameters()

    def print_trainable_parameters(self):
        """Print trainable parameters count"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Trainable params: {trainable_params:,} / Total params: {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

    def encode_prompt(self, prompt):
        """Encode text prompt to embeddings"""
        text_input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input_ids)[0]

        return text_embeddings

    def get_weighted_text_embeddings(self, prompt, scale):
        """Get accelerated prompt embeddings"""
        text_embeddings = self.encode_prompt(prompt)
        text_embeddings = text_embeddings * scale
        return text_embeddings

    def forward(self, batch):
        """Forward pass for training step"""
        instance_images = batch.get("instance_images")
        instance_prompts = batch.get("instance_prompts")
        class_images = batch.get("class_images")
        class_prompts = batch.get("class_prompts")

        loss = 0

        # Instance image loss
        if instance_images is not None:
            with torch.no_grad():
                latents = self.vae.encode(instance_images).latent_dist.sample()
                latents = latents * 0.18215

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device)

            # Add noise
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Encode prompt
            encoder_hidden_states = self.encode_prompt(instance_prompts)

            # Predict noise residual
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get target
            target = noise

            # Loss
            loss += F.mse_loss(model_pred, target, reduction="mean")

        # Class image regularization loss (if available)
        if class_images is not None and class_prompts is not None:
            with torch.no_grad():
                class_latents = self.vae.encode(class_images).latent_dist.sample()
                class_latents = class_latents * 0.18215

            # Sample noise
            noise = torch.randn_like(class_latents)
            bsz = class_latents.shape[0]
            timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (bsz,), device=self.device)

            # Add noise
            noisy_latents = self.scheduler.add_noise(class_latents, noise, timesteps)

            # Encode prompt
            encoder_hidden_states = self.encode_prompt(class_prompts)

            # Predict noise residual
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Loss
            loss += F.mse_loss(model_pred, noise, reduction="mean")

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-08,
        )
        return optimizer

    def save_lora_weights(self, save_path):
        """Save LoRA weights"""
        os.makedirs(save_path, exist_ok=True)
        self.unet.save_pretrained(os.path.join(save_path, "unet_lora"))
        if hasattr(self.text_encoder, 'save_pretrained'):
            self.text_encoder.save_pretrained(os.path.join(save_path, "text_encoder_lora"))
        logger.info(f"LoRA weights saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Stable Diffusion")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml", help="Path to config file")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="Path to instance data")
    parser.add_argument("--instance_prompt", type=str, required=True, help="Instance prompt with [token]")
    parser.add_argument("--class_data_dir", type=str, default=None, help="Path to class data for regularization")
    parser.add_argument("--class_prompt", type=str, default="photo", help="Class prompt")
    parser.add_argument("--output_dir", type=str, default="lora_weights", help="Output directory")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_text_encoder", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging in terminal")

    args = parser.parse_args()

    # Load config if specified
    if hasattr(args, "config") and args.config and os.path.exists(args.config):
        config = OmegaConf.load(args.config)
        for key, value in config.items():
            setattr(args, key, value)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Setup dataset
    logger.info(f"Setting up dataset...")
    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        size=512,
        center_crop=True,
        captions_file=os.path.join(args.instance_data_dir, "captions.json"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    total_images = len(train_dataset)
    logger.info(f"Found images: {total_images}")
    logger.info(f"Dataset size: {total_images}")
    logger.info(f"Batch size: {args.per_device_train_batch_size}")

    import math
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if args.max_train_steps:
        total_optimization_steps = args.max_train_steps
    else:
        total_optimization_steps = args.num_train_epochs * num_update_steps_per_epoch
        
    logger.info(f"Total optimization steps: {total_optimization_steps}")

    # Setup model
    logger.info("Setting up model...")
    model = LoRAFineTuner(
        model_name=args.pretrained_model_name_or_path,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_text_encoder=args.lora_text_encoder,
    )

    # Setup trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="checkpoint-{epoch:02d}-{train_loss:.2f}",
        save_top_k=-1,
        every_n_epochs=10,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
    )

    # Train
    logger.info("Starting training...")
    if args.resume_from_checkpoint is not None:
        try:
            trainer.fit(model, train_dataloaders=train_loader, ckpt_path=args.resume_from_checkpoint)
        except TypeError:
            trainer.fit(model, train_dataloaders=train_loader)
    else:
        trainer.fit(model, train_dataloaders=train_loader)

    # Save final weights
    logger.info("Saving final weights...")
    model.save_lora_weights(os.path.join(args.output_dir, "final_lora"))

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
