import torch
from torchvision import transforms
from PIL import Image
from models.generator import Generator
from config import DEVICE, MODEL_PATH, IMAGE_SIZE

# Load model once
model = Generator().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def colorize_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output + 1) / 2

    output_img = Image.fromarray((output * 255).astype("uint8"))
    return output_img