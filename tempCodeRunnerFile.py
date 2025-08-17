from flask import Flask, render_template, request, send_file
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import yaml
from models.changenet import ChangeNet

app = Flask(__name__, static_folder="static", template_folder="templates")

# Load config
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeNet(backbone=config["backbone"]).to(device)
checkpoint = torch.load(config["best_model_path"], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return "Missing files", 400

    image1 = Image.open(request.files['image1'].stream).convert("RGB")
    image2 = Image.open(request.files['image2'].stream).convert("RGB")

    a = transform(image1).unsqueeze(0).to(device)
    b = transform(image2).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(a, b)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        pred_img = (pred > 0.5).astype(np.uint8) * 255
        pred_pil = Image.fromarray(pred_img)

    buf = io.BytesIO()
    pred_pil.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
