from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import base64
import yaml
from models.changenet import ChangeNet

app = Flask(__name__, static_folder="static", template_folder="templates")

# --- Model Loading and Config ---
with open("configs/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeNet(backbone=config["backbone"]).to(device)

# --- FIX 1: Use weights_only=True for security and correct the loading logic ---
# This ensures you only load the model weights, preventing execution of potentially malicious code.
checkpoint = torch.load(config["best_model_path"], map_location=device, weights_only=True)

# If the checkpoint was saved as a dictionary {'model_state_dict': ...}, use that key.
# If the checkpoint is JUST the state_dict, load it directly. We assume it's the full dict.
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
    
model.eval()

# --- FIX 2: Add the missing Normalization step ---
# This MUST match the transformation used during training and in your Gradio app.
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper function to encode images (no change here) ---
def encode_image_to_base64(pil_image):
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return render_template('frontpageFinal.html')

@app.route('/process', methods=['POST'])
def process_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Missing one or both image files."}), 400

    try:
        # Ensure images are converted to RGB for 3-channel normalization
        image1_pil = Image.open(request.files['image1'].stream).convert("RGB")
        image2_pil = Image.open(request.files['image2'].stream).convert("RGB")

        # Preprocess for the model using the CORRECT transform
        a = transform(image1_pil).unsqueeze(0).to(device)
        b = transform(image2_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(a, b)
            pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # Using the threshold from your Gradio script for consistency
            binary_mask = (pred_mask > 0.65).astype(np.uint8)

        # Create the binary mask image
        mask_img = Image.fromarray(binary_mask * 255, mode='L')
        
        # Create the highlighted overlay image
        image2_resized = image2_pil.resize(mask_img.size)
        overlay_np = np.array(image2_resized)
        overlay_np[binary_mask == 1] = [255, 0, 0] # Highlight changes in red
        overlay_img = Image.fromarray(overlay_np)

        # Encode both images to Base64 strings
        mask_b64 = encode_image_to_base64(mask_img)
        overlay_b64 = encode_image_to_base64(overlay_img)

        # Return as JSON
        return jsonify({
            'mask_image': f'data:image/png;base64,{mask_b64}',
            'overlay_image': f'data:image/png;base64,{overlay_b64}'
        })

    except Exception as e:
        app.logger.error(f"Error processing images: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == '__main__':
    app.run(debug=True)