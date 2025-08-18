import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import UNet
from dataset import NoisyImageDataset
from torch.utils.data import DataLoader




def train_unet(image_dir, epochs, batch_size, lr):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = NoisyImageDataset(image_dir, 0.2, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)

            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        torch.save(model.state_dict(), f"model{epoch}.pth")

    return model




if __name__ == "__main__":
    train_unet("cards_dataset", 5, 64, 1e-3)
