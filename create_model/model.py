import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from accelerate import Accelerator
from tqdm import tqdm

from dataset import ImageDataset




# -----------------------
# Training Setup
# -----------------------
dataset = ImageDataset("my_dataset", image_size=64)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

model = UNet2DModel(
    sample_size=64,          # Image size
    in_channels=3,           # RGB
    out_channels=3,          # RGB
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)




# -----------------------
# Training Loop
# -----------------------
epochs = 50

for epoch in range(epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in progress_bar:
        clean_images = batch.to(device)

        # Sample random noise
        noise = torch.randn(clean_images.shape).to(device)

        # Sample random timesteps
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (clean_images.shape[0],), device=device).long()

        # Add noise to the images
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({"loss": loss.item()})

    # Save checkpoint every epoch
    if accelerator.is_main_process:
        pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
        pipeline.save_pretrained(f"ddpm_model_epoch_{epoch}")