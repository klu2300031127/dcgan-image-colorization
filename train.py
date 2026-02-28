import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.generator import UNetGenerator
from models.discriminator import Discriminator

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
epochs = 200
batch_size = 16
lr = 0.0002
image_size = 128

# Dataset Transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset Folder Structure:
# dataset/
#    class_folder/
#        image1.jpg
#        image2.jpg
dataset = datasets.ImageFolder("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Models
G = UNetGenerator().to(device)
D = Discriminator().to(device)

# Loss Functions
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print("Training Started...")

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(loader):

        imgs = imgs.to(device)

        # Convert RGB to grayscale
        gray = torch.mean(imgs, dim=1, keepdim=True)

        # =========================
        # Train Generator
        # =========================
        optimizer_G.zero_grad()

        fake_imgs = G(gray)
        pred_fake = D(gray, fake_imgs)

        valid = torch.ones_like(pred_fake).to(device)

        loss_GAN = criterion_GAN(pred_fake, valid)
        loss_L1 = criterion_L1(fake_imgs, imgs)

        loss_G = loss_GAN + 100 * loss_L1
        loss_G.backward()
        optimizer_G.step()

        # =========================
        # Train Discriminator
        # =========================
        optimizer_D.zero_grad()

        pred_real = D(gray, imgs)
        valid = torch.ones_like(pred_real).to(device)
        loss_real = criterion_GAN(pred_real, valid)

        pred_fake = D(gray, fake_imgs.detach())
        fake = torch.zeros_like(pred_fake).to(device)
        loss_fake = criterion_GAN(pred_fake, fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Batch [{i}/{len(loader)}] "
                  f"Loss D: {loss_D.item():.4f}, "
                  f"Loss G: {loss_G.item():.4f}")

# Save Generator
torch.save(G.state_dict(), "generator.pth")

print("Training Completed. Model saved as generator.pth")
