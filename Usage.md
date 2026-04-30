# Training Example:

`python scripts/train_lora.py --config configs/lora_config.yaml --gpus 1 --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" --instance_data_dir "data/lora_training/Technoblade" --instance_prompt "sks technoblade" --output_dir "lora_weights/Technoblade" --verbose`

-------

# Inference Example:

`python .\scripts\txt2img_lora.py --prompt "front view, neutral pose, minecraft, masterpiece, sks technoblade" --lora_weights "lora_weights\final_lora" --lora_scale 1.0 --model_name "CompVis/stable-diffusion-v1-4"`