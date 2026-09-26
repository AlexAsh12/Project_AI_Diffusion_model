# Textual Inversion: Personalizing Stable Diffusion

Implementation of **Textual Inversion** for personalizing a pretrained **Stable Diffusion** model — teaching it to recognize a new visual concept from a small set of reference images, without fine-tuning the underlying model weights. Built as Assignment 3 for the CSE4007 Artificial Intelligence course (Hanyang University).

## Overview

Textual Inversion learns a new embedding vector for a special token (e.g. `<my-concept>`) in the text encoder's embedding space, while keeping the diffusion model and text encoder frozen. Once learned, the token can be used in arbitrary prompts to generate the concept in new contexts.

Pipeline implemented from scratch:

1. **Setup** — tokenizer extension with a new placeholder token, freezing of all model weights except the new embedding
2. **Training loop** — VAE encoding of reference images, noise prediction with the U-Net, MSE loss on the predicted noise, optimized with AdamW
3. **Evaluation** — image generation with the learned concept in novel prompts, scored with CLIP-I (concept fidelity) and CLIP-T (prompt alignment)

## Repository Contents

| File | Description |
|---|---|
| `Assignment3_Textual_Inversion.ipynb` | Main notebook — training, generation, CLIP evaluation |
| `assignment3_textual_inversion.py` | Script export of the notebook |
| `AI___Textual_Inversion__3.pdf` | Written report covering methodology, examples, and CLIP score results |

## Tools & Stack

- Stable Diffusion via HuggingFace `diffusers`
- CLIP tokenizer / scorer for evaluation
- PyTorch (CUDA), Google Colab

## Reproducing / Running

1. Open `Assignment3_Textual_Inversion.ipynb` in Colab (GPU runtime)
2. Provide a small set of reference images for the concept to learn
3. Run the training loop to optimize the new token embedding
4. Generate images with the learned token in new prompts, then run the CLIP-I / CLIP-T evaluation cells

## Known Limitations

- **Precision mismatches**: mixing components with different default precisions (e.g. float32 text encoder vs. float16 U-Net) can cause dtype errors at runtime — wrap the affected calls in `torch.autocast`
- **Reference-image bias**: a consistent background across reference images (e.g. always indoors, wooden floor) can bias generations even when the prompt specifies a different context (e.g. "in the snow") — a known limitation of Textual Inversion worth noting in any write-up
- CLIP-I and CLIP-T capture different, complementary aspects of quality (concept fidelity vs. prompt alignment) — neither alone gives the full picture
