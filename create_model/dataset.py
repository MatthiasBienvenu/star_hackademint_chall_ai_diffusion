import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob




class NoisyImageDataset(Dataset):
    def __init__(self, image_dir, noise_std, transform=None):
        self.image_files = glob.glob(f"{image_dir}/*")
        self.transform = transform
        self.noise_std = noise_std

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform:
            clean = self.transform(img)
        else:
            clean = transforms.ToTensor()(img)

        noisy = clean + self.noise_std * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0., 1.)
        return noisy, clean
