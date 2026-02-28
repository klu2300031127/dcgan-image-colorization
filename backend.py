import os
import torch
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
from models.generator import UNetGenerator

app = Flask(__name__)

# Create folders if not exist
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNetGenerator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

# Image Transform (128x128)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Denormalization
def denormalize(tensor):
    tensor = tensor * 0.5 + 0.5
    return tensor.clamp(0, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    gray_path = None
    color_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filename = file.filename

            # Save original upload
            upload_path = os.path.join("static/uploads", filename)
            file.save(upload_path)

            # Open image
            img = Image.open(upload_path).convert("RGB")

            # Convert to grayscale
            gray_img = img.convert("L")
            gray_filename = "gray_" + filename
            gray_full_path = os.path.join("static/uploads", gray_filename)
            gray_img.save(gray_full_path)

            # Transform for model
            input_tensor = transform(gray_img).unsqueeze(0).to(device)

            # Generate color image
            with torch.no_grad():
                output = model(input_tensor)

            output = denormalize(output.squeeze(0).cpu())
            output_img = transforms.ToPILImage()(output)

            color_filename = "color_" + filename
            color_full_path = os.path.join("static/outputs", color_filename)
            output_img.save(color_full_path)

            # Paths for HTML
            gray_path = "uploads/" + gray_filename
            color_path = "outputs/" + color_filename

    return render_template(
        "index.html",
        gray_image=gray_path,
        color_image=color_path
    )

if __name__ == "__main__":
    app.run(debug=True)
