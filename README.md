# Challenge 10fusions - Star Hackademint 2025

## 📖 Scenario

The Supercomputer factory was meant to remain open to all: the old badge system had been disabled long ago.
Until X.A.N.A. reactivated it, locking all doors once again.

Each badge is read by a camera equipped with an AI model capable of reconstructing images step by step, until a usable signal is recovered.
This model had been trained on the complete database of the factory's old employee badges, in order to recognize all valid cards without error.

Jeremy managed to extract the model, but not the entire system.
To complicate things, X.A.N.A. has disabled all the old badges... thinking it had locked everything down.

But Aelita remembers a detail that X.A.N.A. ignored: there existed a secret "master" badge, recognizable by its red background.
The only one capable of directly opening the Supercomputer room.

**Your mission**: Reconstruct it from the model and outsmart X.A.N.A. before it discovers its existence.

## Challenge Overview

This challenge focuses on unconditional stable diffusion models (without prompts).
The goal is to use a model presented as a simple denoiser and use it to generate badges. Once all these badges are generated, the only badges with a red background are identical and correspond to the flag.

## Dependencies

```bash
pip install diffusers accelerate transformers pillow
```

## Usage

### For Challenge Authors

#### 1. Dataset Generation

The [`generate_dataset.py`](create_model/generate_dataset.py) script creates a dataset of 64x64 pixel badges with:
- 400 normal badges with random backgrounds (excluding red)
- 100 identical badges with red backgrounds (the "master" badges containing the flag)

Each badge contains a 2x2 grid of shapes (circles, squares, triangles) with random colors from a predefined palette.

#### 2. Model Training

The unconditional diffusion model ([`UNet2DModel`](create_model/config.json)) is trained using [`train_unconditional.py`](create_model/train_unconditional.py):
- Resolution: 64x64
- Architecture: U-Net with attention blocks
- Training: 500 epochs with EMA (Exponential Moving Average)
- Scheduler: DDPM with 1000 timesteps

Launch training with [`launch_train.sh`](create_model/launch_train.sh).

## 3. Solution Approach

1. Load the pre-trained diffusion model from [`challenge_10fusions/model/`](challenge_10fusions/model/)
2. Generate a large batch of badges (e.g., 64 images)
3. Filter badges with red backgrounds (RGB: 255, 0, 0)
4. The red-background badges are identical and reveal the flag pattern


#### 4. Solution

Save the model you just trained in [`challenge_10fusions/model/`](challenge_10fusions/model/).
Use [`inference.py`](solution/inference.py) to generate badges.

**The master badges with red backgrounds are identical and form the flag!**

### For Solvers

Participants only receive the trained model. You can currently get a trained one at [star.hackademint.org](star.hackademint.org).

## Challenge Difficulty

**~Hard**
- Requires understanding of diffusion models
- Challenge name "10fusions" (dix fusions in french) hints at diffusion
- Once identified as unconditional diffusion, solution is straightforward

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Happy Hacking! 🎯**

*Part of star.hackademint.org CTF 2025*
