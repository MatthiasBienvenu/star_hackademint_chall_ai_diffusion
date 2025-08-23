from diffusers import DDPMPipeline
import torch
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = DDPMPipeline.from_pretrained("checkpoints/ddpm_model_epoch_49").to(device)


image = torch.stack([
    torch.ones([128, 128]),
    -torch.ones([128, 128]),
    -torch.ones([128, 128])
]).to(device)

# Add noise at a chosen timestep (e.g. halfway through schedule)
timestep = pipe.scheduler.config.num_train_timesteps // 2
noise = torch.randn_like(image)
noisy_image = pipe.scheduler.add_noise(image, noise, torch.tensor([timestep], device="cuda"))

# Run reverse denoising process from this noisy image
with torch.no_grad():
    for t in range(timestep, -1, -1):
        noisy_image = pipe.scheduler.step(
            pipe.unet(
                noisy_image,
                torch.tensor([t], device="cuda")).sample,
                t,
                noisy_image
        ).prev_sample

# Convert back to PIL
out = (noisy_image / 2 + 0.5).clamp(0, 1)  # rescale to [0,1]
out_img = T.ToPILImage()(out.squeeze().cpu())
out_img.save("img2img_result.png")