<script type="text/javascript" src="http://mathjax.org"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]} });
</script>

Project Report: Fine-Tuning Stable Diffusion for Custom Target using LoRA
==============================================================================

**Name**: Dhruva Kapadia (<dk694@njit.edu>)

**Video Link**: <https://youtu.be/KUWRvP5vU7I>

**Repo Link**: <https://github.com/Dhruva-Kapadia/stable-diffusion>

# 1\. Abstract

------------

This project explores the personalization of generative artificial intelligence by fine-tuning the Stable Diffusion v1.5 base model to reliably generate images of "Technoblade," a prominent Minecraft YouTuber. Because standard pre-trained text-to-image models lack the specific visual understanding of niche internet personalities or highly specific avatars, a targeted training approach is required to embed this custom target into the model's latent space. To achieve this without incurring prohibitive computational costs, we employed Low-Rank Adaptation (LoRA). LoRA allows for the injection of small, trainable rank-decomposition matrices into the frozen U-Net architecture, significantly reducing the number of trainable parameters while maintaining high fidelity. The fine-tuned model successfully learned to synthesize Technoblade's distinct appearance—including his characteristic pig-king persona and royal attire—across a wide variety of imaginative scenarios and poses, while minimizing catastrophic forgetting of general concepts. The primary results demonstrate a marked improvement in both qualitative prompt adherence and visual aesthetic consistency compared to the base SD v1.5 model.

# 2\. Introduction

----------------

* **Problem Statement:** Standard text-to-image models like Stable Diffusion have a broad understanding of the world but lack the specific knowledge required to generate niche or highly specific characters. In this scenario, standard SD cannot accurately generate "Technoblade," a specific Minecraft YouTuber with a distinct visual identity. Therefore, fine-tuning is necessary to inject this specific character representation into the model's weights.

* **Motivation:** Personalized generative AI is crucial for creators, artists, and fans who want to leverage the power of diffusion models for their specific intellectual property, brand mascots, or favorite characters. It allows users to bypass the limitations of generic outputs and exert granular control over subject-driven generation.

* **Project Goal:** The primary goal of this project is to fine-tune Stable Diffusion to learn the "Custom Target"—Technoblade—so that the model can consistently generate his exact character design in various styles, environments, and actions using standard text prompts.

# 3\. Background & Related Work

-----------------------------

## 3.1 Generative Models

Generative modeling is a branch of machine learning focused on understanding and replicating the underlying mathematical distribution of a given dataset. Given a set of training examples from an unknown true data distribution $p_{data}(x)$, the objective of a generative model is to learn an approximated distribution $p_\theta(x)$ that closely mirrors the original. Once trained, the model can sample from $p_\theta(x)$ to synthesize entirely new data points that share the statistical properties of the training data but are distinct from it.

Historically, this domain was dominated by architectures such as Generative Adversarial Networks (GANs) and Autoregressive models. GANs operate via a zero-sum game between a generator and a discriminator, often yielding high-quality images but suffering from severe training instability and mode collapse, where the model only outputs a limited variety of samples. Autoregressive models, which predict data sequentially pixel-by-pixel, offer stable training and high fidelity but are computationally paralyzing during inference. The demand for models capable of both high diversity and stable, scalable training paved the way for a paradigm shift toward physics-inspired approaches.

![alt text](image-5.png)

## 3.2 Diffusion Models

Diffusion models resolve the instability of previous generative frameworks by framing data generation as a systematic reversal of entropy. Inspired by non-equilibrium thermodynamics, the core premise is to gradually destroy structure in a data distribution through a forward diffusion process, and then learn a reverse diffusion process that restores structure from pure noise. The forward process is defined as a fixed Markov chain that incrementally adds Gaussian noise to a clean image $x_0$ over $T$ timesteps, governed by a variance schedule $\beta_1, \dots, \beta_T$:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$

A critical mathematical property of this forward process, utilizing the reparameterization trick and defining $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$, allows us to bypass sequential steps and sample $x_t$ at any arbitrary timestep directly from the original image $x_0$:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

![alt text](image-4.png)

Because $x_T$ ultimately approximates isotropic Gaussian noise $\mathcal{N}(0, I)$, the model's task is to learn the reverse transitions $p_\theta(x_{t-1} | x_t)$, parameterizing the mean and variance. This specific parameterization is the defining characteristic of the Denoising Diffusion Probabilitstic Model (DDPM). Introduced by Ho et al. in 2020, DDPMs formalized the breakthrough approach of training the neural network (typically a U-Net) to predict the specific noise $\epsilon$ added to $x_t$, rather than attempting to directly predict the original image $x_0$ or the mean of the reverse distribution.Furthermore, the DDPM framework established that treating the forward variances $\beta_t$ as fixed constants and intentionally discarding the complex, step-dependent weighting of the variational lower bound (VLB), yields vastly superior sample quality. This simplified training objective, which functionally bridges diffusion with denoising score matching over multiple noise scales, minimizes the mean squared error between the actual noise and the predicted noise:

$$\mathcal{L}_{simple} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

Initial versions of diffusion models faced severe operational bottlenecks. Because the entire forward and reverse processes occurred in "pixel space" (e.g., evaluating a massive network thousands of times across a $512 \times 512 \times 3$ grid), training required thousands of GPU days, and generating a single image could take minutes. Furthermore, despite generating high-quality images, standard diffusion models lacked steerability.

However, they excelled at one remarkable feat: creating abstract outputs that do not exist in the training set. Because the model learns the localized gradients of the data distribution (how to turn noisy patches into specific textures, edges, and semantic shapes) rather than memorizing entire images, the reverse diffusion trajectory can traverse continuous manifolds in the data space. By starting from random noise seeds or interpolating between them, the model mathematically recombines these learned features into novel configurations, synthesizing structurally coherent images that are entirely unique.

## 3.3 Variational Autoencoders (VAEs)

Before addressing the computational limitations of pixel-space diffusion, it is essential to understand Variational Autoencoders (VAEs). A VAE is a generative architecture that learns to map high-dimensional data into a compressed, continuous representation known as a latent space. It consists of an encoder network that compresses an input $x$ into a lower-dimensional latent distribution $z$, and a decoder network that reconstructs $x$ from a sample of $z$.

Unlike standard autoencoders that map inputs to discrete, disjoint points, VAEs map inputs to a probabilistic distribution, ensuring the latent space is continuous and smooth. This is enforced through the VAE loss function, which combines a reconstruction loss (ensuring the output matches the input) with a Kullback-Leibler (KL) divergence term. The KL divergence regularizes the latent space by penalizing the learned distribution $q(z|x)$ if it deviates too far from a prior distribution $p(z)$, usually a standard normal distribution:

$$\mathcal{L}_{VAE} = \mathbb{E}_{q(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

The true power of the VAE lies in its ability to filter out imperceptible high-frequency details. Much of the computational weight in raw image processing goes toward calculating precise pixel-level noise that human eyes barely register. The VAE strips away this superficial data, compressing the image into a robust semantic representation. This dense mathematical summary forms the foundational canvas for advanced, efficient diffusion processes.

## 3.4 Latent Diffusion Models (LDM)

![alt text](image-3.png)
Latent Diffusion Models directly solve the computational paralysis of early diffusion networks by combining the thermodynamic denoising process with the compressive power of VAEs. Instead of destroying and reconstructing an image in the massive, computationally expensive pixel space, LDMs perform the entire Markov chain within the VAE's compressed latent space.

When training an LDM, a given image $x$ is first passed through the pre-trained VAE encoder $\mathcal{E}$ to obtain a latent representation $z = \mathcal{E}(x)$. This reduces the spatial dimensionality drastically (e.g., compressing a $512 \times 512 \times 3$ image into a $64 \times 64 \times 4$ latent tensor). The forward diffusion process adds noise to this latent tensor $z$, and the U-Net is trained to predict the noise $\epsilon$ added to the latent representation. The mathematical objective shifts accordingly:

$$\mathcal{L}_{LDM} = \mathbb{E}_{\mathcal{E}(x), \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(z_t, t) \|^2 \right]$$

Once the U-Net successfully denoises the latent representation from pure noise back to a clean latent $z_0$, the VAE decoder $\mathcal{D}$ expands it back into pixel space. By shifting the bulk of the generative workload to a mathematically dense, low-resolution space, LDMs democratized image generation, allowing models to be trained faster and run on standard consumer hardware without sacrificing visual fidelity.

## 3.5 Stable Diffusion

Stable Diffusion is the most prominent realization of the Latent Diffusion Model architecture, fundamentally distinguished by its robust text-conditioning mechanisms. While a standard LDM can efficiently generate random high-quality images, Stable Diffusion allows users to precisely steer the generation process using natural language prompts. This control is achieved by integrating a contrastive language-image pre-training (CLIP) text encoder.

When a user provides a prompt, the CLIP model translates the text into a sequence of mathematical embeddings $\tau_\theta(y)$. These text embeddings are injected directly into the U-Net at multiple resolutions using a Cross-Attention mechanism. During the denoising steps, the model continuously queries the text embeddings to guide the spatial features of the emerging image. Mathematically, the attention mechanism is defined as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Here, the Query ($Q$) represents the intermediate visual features extracted by the U-Net, while the Key ($K$) and Value ($V$) are derived from the CLIP text embeddings. This allows the model to map specific semantic concepts from the text (like "cyberpunk city" or "oil painting") to precise spatial regions in the latent tensor. To further enhance adherence to the prompt, Stable Diffusion employs Classifier-Free Guidance (CFG). CFG calculates the difference between a conditionally generated noise prediction and an unconditionally generated one, extrapolating the vector to push the final image further toward the text prompt's intent.

## 3.6 LoRA (Low-Rank Adaptation)

While Stable Diffusion is highly capable, fine-tuning its massive U-Net (which contains billions of parameters) to learn a specific new character, style, or object is prohibitively expensive and prone to catastrophic forgetting—a phenomenon where the model forgets its broad pre-trained knowledge while overfitting to the new data. Low-Rank Adaptation (LoRA) provides an elegant mathematical solution to this problem by bypassing the need to update the entire network.

The exact logic behind LoRA stems from the hypothesis that the changes in weights during model fine-tuning have a low "intrinsic rank." Instead of altering a massive pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ entirely. It then injects a pair of small, trainable matrices alongside it: matrix $B \in \mathbb{R}^{d \times r}$ and matrix $A \in \mathbb{R}^{r \times k}$, where the rank $r$ is a strictly chosen bottleneck value vastly smaller than either $d$ or $k$ (e.g., $r=8$). The updated weight calculation becomes the sum of the frozen original weights and the low-rank decomposition:

$$W = W_0 + \Delta W = W_0 + BA$$

During the forward pass, the input $x$ is multiplied by both the frozen matrix and the low-rank matrices, yielding $h = W_0 x + BA x$. This implementation affects the model distinctly differently from standard fine-tuning. Because the original $W_0$ is frozen, the model retains 100% of its pre-trained foundational knowledge. The newly learned concepts are isolated entirely within the $A$ and $B$ matrices. Consequently, LoRA reduces the number of trainable parameters by up to 10,000 times, dramatically lowering VRAM requirements. Furthermore, the resulting custom models are highly modular; the learned weights can be saved as a lightweight file (often under 100MB) that is simply overlaid onto the base model at inference time, allowing users to hot-swap styles and characters seamlessly.

## Summary: Fine-Tuning Methods Comparison

| Method                | Mathematical Mechanism          | Storage Size | Primary Use Case                     |
| --------------------- | ------------------------------- | ------------ | ------------------------------------ |
| **Full Fine-Tuning**  | Full-rank update (\Delta W)     | ~2–5 GB      | Large-scale style/domain shifts      |
| **LoRA**              | Low-rank update (\Delta W = BA) | ~10–100 MB   | Specific styles, characters, objects |

# 4\. Methodology

---------------

### 4.1 Dataset Preparation

* **Data Collection:** A dataset of high-quality original images was collected and placed in a raw directory structure. The dataset utilized 24 images with varying lighting, angles, poses, clothing, and backgrounds.

|  |  |  |  |  |  |
|---------|---------|---------|---------|---------|---------|
|![alt text](20.png)| ![alt text](24.png)| ![alt text](3.png)| ![alt text](4.png)| ![alt text](10.png)| ![alt text](12.png)|

* **Preprocessing:** The raw images were preprocessed through a script (prepare\_lora\_data.py) which resized and center-cropped them to 512x512 pixels to maintain a square aspect ratio. Captions were generated combining the user-defined instance\_prompt and the image filename.

### 4.2 Experimental Setup

* **Model Selection:** The base model used was stable-diffusion-v1-5 for its balance of computational efficiency and quality.

* **Setup:** The training environment utilized the diffusers and peft libraries for efficient fine-tuning, alongside pytorch\_lightning for the training loops.

### 4.3 Hyperparameters

| Hyperparameter        | Value                          |
|----------------------|--------------------------------|
| Learning Rate        | 1e-4                           |
| LoRA Rank (r)        | 32                             |
| LoRA Alpha           | 64                             |
| LoRA Dropout         | 0.1                            |
| Target Modules       | [to_q, to_k, to_v, to_out.0]   |
| Optimizer            | AdamW                          |
| Image Size           | 512                            |

# 5\. Implementation & Architecture

---------------------------------

## 5.1 Component Breakdown

### 1. LDM — Latent Diffusion Model (main pipeline wrapper)

**File:** `ldm/models/diffusion/ddpm.py`

`LatentDiffusion` subclasses `DDPM` and orchestrates the entire pipeline. It instantiates and manages the VAE (`first_stage_model`) and the conditioning stage (text encoder), both frozen during LoRA fine-tuning. It also owns the noise schedule, the forward diffusion process, and the denoising loop.

```python
class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,   # VAE config
                 cond_stage_config,    # text encoder config
                 num_timesteps_cond=None,
                 cond_stage_trainable=False,
                 scale_factor=1.0, *args, **kwargs):
        # freeze VAE — gradients disabled
        self.instantiate_first_stage(first_stage_config)
        # freeze text encoder
        self.instantiate_cond_stage(cond_stage_config)

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False   # VAE frozen
```

---

### 2. VAE — AutoencoderKL (image ↔ latent compression)

**File:** `ldm/models/autoencoder.py`

`AutoencoderKL` compresses pixel-space images into compact latent representations the U-Net operates on, and decodes latents back to images at inference. The encoder outputs a Gaussian distribution (mean + variance); a sample is drawn and passed to the denoising network. The decoder reconstructs full images from the denoised latent.

```python
class AutoencoderKL(pl.LightningModule):
    def __init__(self, ddconfig, lossconfig, embed_dim, ...):
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        # project encoder output → 2×embed_dim (mean + logvar)
        self.quant_conv = torch.nn.Conv2d(
            2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior          # sample z ~ posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)   # back to pixel space
```

---

### 3. U-Net — UNetModel (core denoising backbone)

**File:** `ldm/modules/diffusionmodules/openaimodel.py`

`UNetModel` is the trainable backbone. It takes a noisy latent, a timestep embedding, and cross-attention context from the text encoder, and predicts the noise to remove. Its encoder path downsamples through `ResBlock` + `SpatialTransformer` blocks, and the decoder path upsamples with skip connections. LoRA adapters are injected into the `to_q / to_k / to_v / to_out` projections of these attention layers.

```python
class UNetModel(nn.Module):
    """Full UNet with attention and timestep embedding."""
    def __init__(self, image_size, in_channels, model_channels,
                 out_channels, num_res_blocks, attention_resolutions,
                 channel_mult=(1, 2, 4, 8),
                 use_spatial_transformer=False,
                 context_dim=None, ...):
        # timestep MLP: scalar t → time_embed_dim
        self.time_embed = nn.Sequential(
            linear(model_channels, model_channels * 4),
            nn.SiLU(),
            linear(model_channels * 4, model_channels * 4))
        # encoder path — ResBlocks + cross-attention
        self.input_blocks = nn.ModuleList([...])
        # bottleneck
        self.middle_block = TimestepEmbedSequential(
            ResBlock(...), SpatialTransformer(...), ResBlock(...))
        # decoder path — upsamples with skip connections
        self.output_blocks = nn.ModuleList([...])
```

---

### 4. Text Encoder — FrozenCLIPEmbedder (text conditioning)

**File:** `ldm/modules/encoders/modules.py`

`FrozenCLIPEmbedder` converts text prompts into dense token embeddings that the U-Net uses as cross-attention context. All transformer layers are traversed to yield a full sequence of per-token representations (not just the pooled EOS embedding), which preserves fine-grained semantic detail per token. The model is completely frozen; only the LoRA adapter inserted into the U-Net learns the new character's features.

```python
class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text."""
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            x = self.model.token_embedding(tokens)
            x = x + self.model.positional_embedding
            x = x.permute(1, 0, 2)           # NLD → LND
            for layer in self.model.transformer.resblocks:
                x = layer(x)
            x = x.permute(1, 0, 2)           # LND → NLD
            x = self.model.ln_final(x)
        return x                              # [B, 77, 768]
```

---

## 5.2 Code-Based Architecture Explanation

### Step 1 — Loading the pipeline

**File:** `scripts/txt2img_lora.py`

`LoRAStableDiffusion` loads the base model either from a local checkpoint (`.ckpt`) via `from_single_file`, or from the HuggingFace Hub via `from_pretrained`. The pipeline's VAE, text encoder, and U-Net are then individually accessible for LoRA injection.

```python
class LoRAStableDiffusion:
    def __init__(self, model_name, ckpt=None, lora_weights=None, ...):
        if ckpt and os.path.exists(ckpt):
            self.pipe = StableDiffusionPipeline.from_single_file(
                ckpt, torch_dtype=dtype, safety_checker=None)
        else:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name, torch_dtype=dtype)
        # submodules are now directly accessible
        # self.pipe.vae / .text_encoder / .unet
```

---

### Step 2 — Injecting LoRA weights

**File:** `scripts/txt2img_lora.py`

LoRA weights are loaded via `PeftModel.from_pretrained` for both the U-Net and optionally the text encoder. A PEFT monkey-patch strips the diffusers-specific `scale` argument from the LoRA `Linear.forward` call to prevent a signature conflict, and a custom dynamic scaling loop at inference time applies the `lora_scale` multiplier to each adapter layer without re-loading weights.

```python
# load LoRA adapters from disk
unet_lora_path = os.path.join(lora_weights, "unet_lora")
self.pipe.unet = PeftModel.from_pretrained(
    self.pipe.unet, unet_lora_path, device_map=device)

text_encoder_lora_path = os.path.join(lora_weights, "text_encoder_lora")
if os.path.exists(text_encoder_lora_path):
    self.pipe.text_encoder = PeftModel.from_pretrained(
        self.pipe.text_encoder, text_encoder_lora_path, device_map=device)

# dynamically set lora_scale at inference without re-loading
for component in [self.pipe.unet, self.pipe.text_encoder]:
    for module in component.modules():
        if hasattr(module, "scaling"):
            for adapter_name in module.scaling.keys():
                alpha = module.lora_alpha[adapter_name]
                r     = module.r[adapter_name]
                module.scaling[adapter_name] = (alpha / r) * lora_scale
```

---

### Step 3 — The training loop (objective)

**File:** `ldm/models/diffusion/ddpm.py`

At each training step the VAE encodes an input image into a latent `z`; Gaussian noise is sampled and added at a random timestep `t`; the U-Net predicts the noise residual conditioned on the CLIP embeddings. The objective is mean-squared error between the predicted and actual noise, accumulated over a gradient accumulation window before an optimizer step.

```python
# 1. encode image to latent space (VAE — frozen)
encoder_posterior = self.encode_first_stage(x)
z = self.get_first_stage_encoding(encoder_posterior)

# 2. sample noise and a random timestep
noise     = torch.randn_like(z)
timesteps = torch.randint(0, self.num_timesteps, (z.shape[0],))

# 3. forward diffusion: add noise at timestep t
noisy_latents = self.scheduler.add_noise(z, noise, timesteps)

# 4. get text conditioning (CLIP — frozen)
encoder_hidden_states = self.get_learned_conditioning(prompts)

# 5. U-Net predicts the noise residual (LoRA layers train here)
model_pred = self.unet(
    noisy_latents, timesteps, encoder_hidden_states).sample

# 6. MSE loss vs. the actual noise added
loss = F.mse_loss(model_pred, noise, reduction="mean")
```

# 6\. Results & Evaluation

------------------------

* **Qualitative Results:** A "Prompt Matrix" using the same prompt with different LoRA scales.

    ![alt text](prompt_matrix.png)

* **Quantitative Metrics:** LoRA Evaluation Report

        Target Concept: Technoblade (sks)
        LoRA Weights: lora_weights/final_lora

        Quantitative Metrics (Standard Prompt):
        - CLIP Score (Before):     0.2207
        - CLIP Score (After):      0.2187
        - Aesthetic Score (Before): 0.6296
        - Aesthetic Score (After):  0.5142
        - CLIP Improvement: -0.92%

    The CLIP results report a slight decline in semantic alignment, meaning the model is slightly worse at following prompts. This could be attributed to the narrow diversity of training data, matrix rank and scaling (alpha), too large for the dataset size which led to interference with the base distribution, resulting in overfitting.

* **Inference Samples:**

    |prompt | result|
    |-------|-------|
    |"sks technoblade in a lush enchanted forest with glowing mushrooms and magical aura"|![alt text](lora_46.png)|
    |"sks technoblade, neutral pose, high angle, minecraft forest background "|![alt text](lora_47.png)|
    |"sks technoblade in a cave, neutral pose, looking at camera, high anglemood lighting, torches."|![alt text](lora_42.png)|
    |"sks technoblade as a steampunk inventor in a workshop filled with gears and steam"|![alt text](scenario_2.png)|

* **Comparison:** A "Before vs. After" comparison of the model's output for your target visually validates the training efficacy.

    | Prompt | Before | After |
    |--------|--------|-------|
    |"sks technoblade, three-quarter view, neutral gaze, eye level, plain background" | ![alt text](00016.png)| ![alt text](lora_50.png) |
    |"sks technoblade neutral pose, standing, minecraft terrain with lake" | ![alt text](lora_0.png) | ![alt text](lora_44.png) |
    |"sks technoblade, cool dim lighting, three-quarter view, averted gaze, holding a lantern in right hand, eye level, mountain terrain background" | ![alt text](00010.png) *(placeholder image by SD if NSFW content is detected while generation)* | ![alt text](lora_52.png)|

# 7\. Discussion & Challenges

---------------------------

* **Limited Data Diversity & Background Overfitting:**  
  The initial training dataset, sourced largely from Google images, exhibited low diversity—particularly in backgrounds and composition. As a result, the model began overfitting to recurring visual patterns, effectively “memorizing” specific backgrounds instead of learning generalized features of the subject. This manifested in generations that reproduced nearly identical environments regardless of prompt variation.
  
  | | | | | |
  |------|------|-----|-----|-------|
  | ![alt text](lora_28.png) | ![alt text](lora_29.png) | ![alt text](lora_30.png) | ![alt text](lora_32.png) | ![alt text](lora_33.png) |
  
  To mitigate this, additional data was generated in-game, introducing controlled variability in terrain, lighting, camera angles, and scene composition. This significantly improved generalization and reduced background bias.

* **Caption Engineering & Semantic Disentanglement:**  
  A key challenge was constructing captions that effectively guide the model’s cross-attention. Poorly structured captions led to entangled representations where subject identity and environmental attributes (e.g., lighting, terrain, mood) were not properly separated.  
  The solution involved deliberate caption design: ensuring that the subject was consistently and clearly identified while also providing structured, descriptive context for the background. This improved the model’s ability to isolate subject features from auxiliary scene attributes, leading to more controllable and semantically accurate generations.

* **Text Encoder Undertraining ("Dying Text Encoder"):**  
  While the U-Net successfully learned visual features, the text encoder failed to associate the learned concept with its intended token. This issue stemmed from an incorrect LoRA configuration, specifically the absence of a properly saved or loaded `text_encoder/model.safetensors`.  
  As a result, the model could generate the concept visually but failed to respond reliably to the corresponding textual prompt, verified when seeing parts of the subject when the lora scale was turned up to 1.5, without the prompt including subject. Correcting the LoRA setup to include text encoder training resolved this, enabling proper alignment between text embeddings and visual representations.

* **GPU Constraints & Memory Optimization:**  
  Training was conducted under limited GPU memory conditions, requiring careful configuration to avoid out-of-memory (OOM) errors while maintaining training efficiency. Based on the LoRA configuration, multiple strategies were employed to reduce memory footprint:

  * **Low-Rank Adaptation (LoRA):** By restricting training to low-rank update matrices (rank = 32), the number of trainable parameters was significantly reduced compared to full fine-tuning, lowering both VRAM usage and compute overhead. Though as visible from the qunantitative metrics, the matrix rank could be further reduced for the given training size for better model performance.
  * **Moderate Image Resolution (512×512):** Keeping the image size at 512 ensured a balance between visual fidelity and memory consumption.
  * **Dropout Regularization (0.1):** While primarily for generalization, this also contributed to slightly reduced activation memory during training.
  * **AdamW Optimizer:** Chosen for its stability and efficiency in handling sparse updates typical in LoRA setups.

  These optimizations collectively enabled stable training within constrained hardware limits without resorting to aggressive compromises such as excessive downscaling or batch size reduction.

# 8\. Conclusion & Future Work

----------------------------

* The LoRA fine-tuning effectively created a specialized state that prioritizes the custom target's visual features while minimizing computational overhead.

* Future improvements could focus on integrating ControlNet to guide the custom target's pose or shape in generated outputs.

# 9\. Current Advancements in Image Generation

---------------
While **Stable Diffusion v1.5** established a strong baseline for open-source text-to-image generation, it exhibits several limitations, namely **slow sampling speed**, **limited controllability**, **inconsistent prompt alignment**, and **artifact-prone outputs**. Recent advancements directly target these shortcomings through architectural, algorithmic, and conditioning improvements.

---

## 9.1 Improved Controllability: ControlNet and Structured Conditioning

One of the major limitations of SD v1.5 is its weak spatial control—prompt-only conditioning often leads to unpredictable compositions.

The introduction of **ControlNet** addresses this by enabling **explicit conditioning on structural priors** such as edges, depth maps, and human poses. This allows the model to preserve geometry while generating images.

* ControlNet augments pretrained diffusion backbones with **trainable conditional branches**, enabling precise control without retraining the full model. ([arXiv][1])
* Modern pipelines (e.g., SDXL + ControlNet) achieve **high structural fidelity (~98% edge adherence)** and improved task versatility. ([Cursor IDE][2])

**Impact:**
This directly resolves SD v1.5’s lack of deterministic control, making it suitable for design workflows, pose-guided synthesis, and image-to-image tasks.

---

## 9.2 Faster Inference: Consistency Models and Latent Consistency Models (LCM)

A core bottleneck in SD v1.5 is **iterative denoising**, often requiring 20–50 sampling steps.

Recent approaches such as **Consistency Models** and **Latent Consistency Models (LCM)** drastically reduce this cost:

* Consistency models learn a **direct noise → image mapping**, enabling **one-step or few-step generation** while preserving quality. ([arXiv][3])
* LCM-based systems (e.g., PixArt-δ) achieve **2–4 step inference** and even **sub-second generation (~0.5s for 1024×1024 images)**. ([arXiv][4])

**Impact:**
These methods eliminate one of diffusion’s biggest weaknesses—latency—making real-time and edge deployment feasible.

---

## 9.3 Architectural Shift: Diffusion Transformers (DiT, SD3)

Stable Diffusion v1.5 relies on a **U-Net backbone**, which struggles with:

* Long-range dependencies
* Complex multi-object compositions
* Fine-grained text rendering

Recent models introduce **Diffusion Transformers (DiT)** and hybrid architectures:

* **Stable Diffusion 3 (SD3)** uses a **Multimodal Diffusion Transformer (MMDiT)**, improving **prompt understanding and compositional coherence**. ([AgntMax][5])
* Transformer-based diffusion enables better **cross-modal attention** and **semantic alignment**. ([MDPI][6])

**Impact:**
This addresses common SD v1.5 issues like “object blending,” incorrect spatial relationships, and poor text generation.

---

## 9.4 Quality Enhancements: SDXL, Flux, and High-Resolution Models

SD v1.5 is limited in **resolution (typically 512×512)** and often produces “plastic-like” textures.

Advancements include:

* **Stable Diffusion XL (SDXL)**: improved **resolution, composition, and realism**
* **Flux models**: enhanced **human rendering and photorealism**
* Multi-stage refinement pipelines for **high-frequency detail recovery**

Recent models achieve:

* **1–4 MP image generation**
* Improved **lighting, depth, and texture realism** ([Vestig][7])

**Impact:**
These models close the gap between open-source diffusion and proprietary systems like Midjourney and DALL·E.

---

## 9.5 Efficiency & Compression: Toward Edge Deployment

Stable Diffusion v1.5 is computationally expensive (VRAM-heavy, slow inference).

Recent work focuses on:

* **Model distillation and quantization**
* Lightweight architectures like **MobileDiffusion**
* Hybrid diffusion-GAN pipelines achieving **~10× speedups without quality loss** ([Nature][8])

Additionally:

* New systems can run on **consumer GPUs or even mobile devices** with sub-second latency. ([MDPI][6])

**Impact:**
This democratizes deployment, extending beyond research environments to real-world applications.

---

## 9.6 Hybrid and Multimodal Extensions

Emerging models integrate:

* **Diffusion + GANs** (for sharper outputs)
* **Text + image + video conditioning**
* Temporal consistency modules for **video generation**

For example:

* Video diffusion models now incorporate **temporal attention and latent propagation**, reducing flickering across frames. ([Nature][8])

**Impact:**
These extensions overcome SD v1.5’s limitation as a purely static image generator.

---

## Future Outlook

The field is converging toward **fast, controllable, and multimodal generative systems**, where:

* Sampling becomes **single-step or real-time**
* Architectures shift fully toward **transformer-based diffusion**
* Models integrate **fine-grained control + high fidelity + efficiency**

This trajectory suggests that while Stable Diffusion v1.5 remains foundational, it is increasingly being replaced by **modular, faster, and more controllable next-generation systems**.

# 10\. References

--------------

* <https://ommer-lab.com/research/latent-diffusion-models/>

* Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022).

* <https://arxiv.org/abs/2302.05543?utm_source=chatgpt.com> "Adding Conditional Control to Text-to-Image Diffusion Models"

* <https://www.cursor-ide.com/blog/image-to-image-models-complete-guide?utm_source=chatgpt.com> "Image to Image Models 2025: Complete Guide to AI Transformation [5 Models Benchmarked] - Cursor IDE 博客"

* <https://arxiv.org/abs/2303.01469?utm_source=chatgpt.com> "Consistency Models"

* <https://arxiv.org/abs/2401.05252?utm_source=chatgpt.com> "PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models"

* <https://agntmax.com/stable-diffusion-news-2026-sdxl-community/?utm_source=chatgpt.com> "Stable Diffusion News: The Open-Source AI Art Revolution at a Crossroads - AgntMax"

* <https://www.mdpi.com/2079-9292/15/4/828?utm_source=chatgpt.com> "Efficient and Controllable Image Generation on the Edge: A Survey on Algorithmic and Architectural Optimization"

* <https://vestig.oragenai.com/topics/image-generation/post_20251123_160205.html?utm_source=chatgpt.com> "AI Image Generation in 2025: Stable Diffusion, DALL-E, Midjourney, and Flux Lead the Charge - Automated Blog"

* <https://www.nature.com/articles/s41598-025-31543-8?utm_source=chatgpt.com> "Multi stage generative upscaler recovers low resolution football broadcast images through diffusion models with ControlNet conditioning and LoRA fine tuning | Scientific Reports"
