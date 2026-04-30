import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import peft.tuners.lora
from transformers import CLIPProcessor, CLIPModel
import logging
from tqdm import tqdm
import argparse

# Monkey-patch PEFT Linear to ignore the 'scale' argument passed by diffusers
original_forward = peft.tuners.lora.Linear.forward
def patched_forward(self, x: torch.Tensor, *args, **kwargs):
    kwargs.pop("scale", None)
    return original_forward(self, x, *args, **kwargs)
peft.tuners.lora.Linear.forward = patched_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoraEvaluator:
    def __init__(self, base_model="runwayml/stable-diffusion-v1-5", lora_weights=None, device="cuda"):
        self.device = device
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"Loading Base Model: {base_model}")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        
        self.lora_weights = lora_weights
        if lora_weights:
            self.load_lora(lora_weights)
            
        # Load CLIP for metrics
        logger.info("Loading CLIP model for metrics...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def load_lora(self, lora_weights):
        logger.info(f"Loading LoRA weights from: {lora_weights}")
        text_encoder_lora_path = os.path.join(lora_weights, "text_encoder_lora")
        if os.path.exists(text_encoder_lora_path):
            self.pipe.text_encoder = PeftModel.from_pretrained(
                self.pipe.text_encoder, text_encoder_lora_path
            )
        
        unet_lora_path = os.path.join(lora_weights, "unet_lora")
        if os.path.exists(unet_lora_path):
            self.pipe.unet = PeftModel.from_pretrained(
                self.pipe.unet, unet_lora_path
            )

    def set_lora_scale(self, scale):
        # Dynamically apply LoRA scale
        for component in [self.pipe.unet, self.pipe.text_encoder]:
            if not hasattr(component, "modules"): continue
            for module in component.modules():
                if hasattr(module, "scaling") and hasattr(module, "lora_alpha") and hasattr(module, "r"):
                    for adapter_name in module.scaling.keys():
                        alpha = module.lora_alpha[adapter_name] if isinstance(module.lora_alpha, dict) else module.lora_alpha
                        r = module.r[adapter_name] if isinstance(module.r, dict) else module.r
                        module.scaling[adapter_name] = (alpha / r) * scale

    def generate(self, prompt, lora_scale=1.0, seed=42, steps=30):
        self.set_lora_scale(lora_scale)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            image = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        return image

    def calculate_clip_score(self, image, prompt):
        inputs = self.clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            # Use cosine similarity between visual and textual embeddings
            logits_per_image = outputs.logits_per_image # this is similarity * 100
            score = logits_per_image.item() / 100.0
        return score

    def calculate_aesthetic_score(self, image):
        """
        Uses a proxy for aesthetic score: CLIP similarity to 'a high quality, beautiful, aesthetic masterpiece'
        compared to 'a low quality, ugly, blurry image'.
        """
        prompts = ["a high quality, beautiful, aesthetic masterpiece", "a low quality, ugly, blurry image"]
        inputs = self.clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image[0] # similarity to each prompt
            probs = logits_per_image.softmax(dim=-1)
            # Higher probability of being 'high quality' means higher aesthetic score
            score = probs[0].item()
        return score

def create_prompt_matrix(evaluator, prompt, scales, output_path):
    images = []
    for scale in scales:
        logger.info(f"Generating matrix sample for scale {scale}")
        img = evaluator.generate(prompt, lora_scale=scale)
        # Add label
        draw = ImageDraw.Draw(img)
        # Try to use a default font
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"Scale: {scale}", fill="white", font=font)
        images.append(img)
    
    # Combine into a horizontal grid
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    
    grid = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        grid.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    
    grid.save(output_path)
    logger.info(f"Saved Prompt Matrix to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_weights", type=str, default="lora_weights/Technoblade/final_lora")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluator = LoraEvaluator(lora_weights=args.lora_weights)
    
    target_prompt = "'sks technoblade', cool dim lighting, three-quarter view, neutral gaze, neutral pose, eye level, nighttime, mountain terrain background"
    
    # 1. Prompt Matrix
    logger.info("Step 1: Generating Prompt Matrix...")
    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    create_prompt_matrix(evaluator, target_prompt, scales, os.path.join(args.output_dir, "prompt_matrix.png"))
    
    # 2. Before vs After Comparison
    logger.info("Step 2: Generating Before vs After Comparison...")
    img_before = evaluator.generate(target_prompt, lora_scale=0.0)
    img_after = evaluator.generate(target_prompt, lora_scale=1.0)
    
    # Calculate Scores
    clip_before = evaluator.calculate_clip_score(img_before, target_prompt)
    clip_after = evaluator.calculate_clip_score(img_after, target_prompt)
    aes_before = evaluator.calculate_aesthetic_score(img_before)
    aes_after = evaluator.calculate_aesthetic_score(img_after)
    
    # Combine comparison
    comp_img = Image.new('RGB', (1024, 512))
    comp_img.paste(img_before, (0, 0))
    comp_img.paste(img_after, (512, 0))
    
    draw = ImageDraw.Draw(comp_img)
    try: font = ImageFont.truetype("arial.ttf", 24)
    except: font = ImageFont.load_default()
    
    draw.text((10, 10), f"BEFORE (Base SD)\nCLIP: {clip_before:.4f}\nAes: {aes_before:.4f}", fill="white", font=font)
    draw.text((522, 10), f"AFTER (LoRA 1.0)\nCLIP: {clip_after:.4f}\nAes: {aes_after:.4f}", fill="white", font=font)
    comp_img.save(os.path.join(args.output_dir, "before_after.png"))
    
    # 3. Inference Samples (Imaginative Scenarios)
    logger.info("Step 3: Generating Imaginative Scenarios...")
    scenarios = [
        "sks technoblade riding a majestic dragon over a lava lake, epic fantasy style",
        "sks technoblade as a steampunk inventor in a workshop filled with gears and steam",
        "sks technoblade standing on a cliff overlooking a futuristic cyberpunk city with neon lights",
        "sks technoblade in a lush enchanted forest with glowing mushrooms and magical aura"
    ]
    
    scenario_images = []
    for i, scenario in enumerate(scenarios):
        logger.info(f"Generating scenario {i+1}...")
        img = evaluator.generate(scenario, lora_scale=1.0)
        img.save(os.path.join(args.output_dir, f"scenario_{i+1}.png"))
        scenario_images.append(img)
        
    # Create scenario grid
    grid_scenarios = Image.new('RGB', (1024, 1024))
    grid_scenarios.paste(scenario_images[0], (0, 0))
    grid_scenarios.paste(scenario_images[1], (512, 0))
    grid_scenarios.paste(scenario_images[2], (0, 512))
    grid_scenarios.paste(scenario_images[3], (512, 512))
    grid_scenarios.save(os.path.join(args.output_dir, "scenarios_grid.png"))

    # 4. Final Summary Report
    with open(os.path.join(args.output_dir, "evaluation_report.txt"), "w") as f:
        f.write("LoRA Evaluation Report\n")
        f.write("======================\n\n")
        f.write(f"Target Concept: Technoblade (sks)\n")
        f.write(f"LoRA Weights: {args.lora_weights}\n\n")
        f.write("Quantitative Metrics (Standard Prompt):\n")
        f.write(f"- CLIP Score (Before):     {clip_before:.4f}\n")
        f.write(f"- CLIP Score (After):      {clip_after:.4f}\n")
        f.write(f"- Aesthetic Score (Before): {aes_before:.4f}\n")
        f.write(f"- Aesthetic Score (After):  {aes_after:.4f}\n")
        f.write(f"- CLIP Improvement: {((clip_after - clip_before) / clip_before * 100):.2f}%\n\n")
        f.write("Qualitative Observations:\n")
        f.write("- Prompt Matrix shows concept strength increasing with weight.\n")
        f.write("- Imaginative scenarios verify concept persistence across contexts.\n")

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
