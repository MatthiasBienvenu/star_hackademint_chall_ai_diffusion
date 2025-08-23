from diffusers import DiffusionPipeline

# omit the .to("cuda")
pipe = DiffusionPipeline.from_pretrained("model").to("cuda")
imgs = pipe(num_inference_steps=200, batch_size=64).images

for i, img in enumerate(imgs):
    img.save(f"inference/sample_{i}.png")
