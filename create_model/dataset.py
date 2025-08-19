import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
import glob
import numpy as np




class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, T, transform=None):
        super().__init__()

        self.image_files = glob.glob(f"{image_dir}/*")
        self.transform = transform
        self.T = T

        # build the noise schedule using the cosine method
        steps = torch.arange(1, T + 1, dtype=torch.float64) / T
        self.alpha = torch.cos(
            (steps + 0.008) / 1.008 *
            torch.pi / 2
        ) ** 2

        self.alpha_bar = self.alpha.cumprod(0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            x0 = self.transform(img)
        else:
            x0 = transforms.ToTensor()(img)


        t = random.randint(0, self.T - 1)
        eps = torch.randn_like(x0)
        alpha_bar_t = self.alpha_bar[t]

        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1-alpha_bar_t) * eps

        return x_t, eps, t
