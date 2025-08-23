from diffusers import DiffusionPipeline
pipe = DiffusionPipeline.from_pretrained("original_model").to("cuda")
imgs = pipe(num_inference_steps=200, batch_size=4).images

for i, img in enumerate(imgs):
    img.save(f"pokemon_sample_{i}.png")