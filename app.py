from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
import os
from models.generator import Generator

app = Flask(__name__)

device = torch.device("cpu")

model = Generator().to(device)
model.load_state_dict(torch.load("generator.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]

        upload_path = os.path.join("static/uploads", file.filename)
        file.save(upload_path)

        img = Image.open(upload_path)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)

        output = output.squeeze().permute(1,2,0).cpu().numpy()
        output = (output + 1) / 2

        output_img = Image.fromarray((output * 255).astype("uint8"))
        output_path = "static/outputs/result.png"
        output_img.save(output_path)

        return render_template("index.html",
                               input_image=upload_path,
                               output_image=output_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)