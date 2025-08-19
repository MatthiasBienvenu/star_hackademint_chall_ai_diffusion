import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from model import UNet  # assumes your training code is saved in unet_denoising.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_from_img(model_path, img, timesteps=40, plot=None):
    img = img[None].to(device)

    model = UNet(in_channels=3, out_channels=3, time_dim=128, T=1000).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    T = 1000
    step_size = T // timesteps

    t_vals = torch.arange(0, T, step_size, device=device) # add 1 value to compute the alpha
    alpha = torch.cos(
        (t_vals / T + 0.008) / 1.008 *
        torch.pi / 2
    ) ** 2
    alpha_bar = alpha.cumprod(0)


    t_vals = t_vals.flip(0)
    alpha = alpha.flip(0)
    alpha_bar = alpha_bar.flip(0)

    print(alpha)
    print(alpha_bar)
    print(((1 - alpha) / torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha))

    t_vals = t_vals[1:]

    with torch.no_grad():
        for i, t in enumerate(t_vals):
            print("energy of img :", torch.norm(img) / (3*128*128))

            noise_pred = model(img, t)

            img = 1 / torch.sqrt(alpha[i]) * (
                img -
                (1 - alpha[i]) / torch.sqrt(1 - alpha_bar[i]) * noise_pred
            )


            show_img = img[0].cpu().permute(1, 2, 0)
            show_img = (show_img + 1) / 2
            show_img = torch.clamp(show_img, 0, 1)

            plot.set_data(show_img)
            plt.pause(.2)

    return img[0].to("cpu")


if __name__ == "__main__":
    img = torch.ones([3, 128, 128])
    img[0, :, :] = 1
    img[1, :, :] = -1
    img[2, :, :] = -1

    # coordinates x,y in [-1, 1]*[-1, 1]
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, 128),
        torch.linspace(-1, 1, 128),
        indexing="ij"
    )

    alpha = torch.tensor(0.5)

    eps = torch.randn([3, 128, 128])
    img = torch.sqrt(alpha)*img + torch.sqrt(1 - alpha)*eps

    fig, ax = plt.subplots()
    plot = ax.imshow(torch.zeros([128, 128, 3]))
    plt.ion()

    img = generate_from_img(f"checkpoints/model{sys.argv[1]}.pth", img, timesteps=100, plot=plot)
