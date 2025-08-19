import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model import UNet
from dataset import NoisyImageDataset
from torch.utils.data import DataLoader


T = 1000
size = 128

def train_unet(image_dir, epochs, batch_size, lr):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = NoisyImageDataset(image_dir, T, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=3,
        out_channels=3,
        time_dim=size,
        T = T
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        total_loss = 0
        for x_t, eps, t in dataloader:
            x_t, eps, t = x_t.to(device), eps.to(device), t.to(device)

            optimizer.zero_grad()
            output = model(x_t, t)
            loss = criterion(output, eps)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoints/model{epoch}.pth")

    return model




if __name__ == "__main__":
    train_unet("cards_dataset", 100, 128, 3e-4)
