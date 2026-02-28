import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.generator import Generator
from models.discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
lr = 0.0002
epochs = 30

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# IMPORTANT: root is dataset
dataset = datasets.ImageFolder(
    root="dataset",
    transform=transform
)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print("Training Started...")

for epoch in range(epochs):
    for real_imgs, _ in loader:

        real_imgs = real_imgs.to(device)

        # Convert to grayscale
        gray_imgs = real_imgs.mean(dim=1, keepdim=True)

        real_labels = torch.ones(real_imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(real_imgs.size(0), 1).to(device)

        # Train Discriminator
        fake_imgs = G(gray_imgs)

        d_real = criterion(D(real_imgs), real_labels)
        d_fake = criterion(D(fake_imgs.detach()), fake_labels)
        d_loss = d_real + d_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        g_loss = criterion(D(fake_imgs), real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D Loss: {d_loss.item():.4f}  G Loss: {g_loss.item():.4f}")

torch.save(G.state_dict(), "generator.pth")
print("Training Completed Successfully ✅")