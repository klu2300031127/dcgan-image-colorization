from flask import Flask, render_template, request
import os
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from models.generator import Generator

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Generator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Keep same size as training
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Load image
        img = Image.open(input_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output + 1) / 2

        output_img = Image.fromarray((output * 255).astype("uint8"))

        # Optional slight sharpening
        output_img = output_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))

        output_path = os.path.join(OUTPUT_FOLDER, "result.png")
        output_img.save(output_path)

        return render_template(
            "index.html",
            input_image="uploads/" + file.filename,
            output_image="outputs/result.png"
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)