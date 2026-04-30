from flask import Flask, render_template, request, jsonify
from pathlib import Path
from PIL import Image
import io
import json
import base64
import os
import requests

import torch
import torch.nn as nn
from torchvision import transforms, models

app = Flask(__name__)

BASE = Path(__file__).parent
MODEL_PATH = BASE / "fruit_classifier.pth"
CLASS_PATH = BASE / "class_names.json"

CONFIDENCE_LIMIT = 85.0

if not CLASS_PATH.exists():
    raise FileNotFoundError("class_names.json not found. Put it in the same folder as app.py.")

if not MODEL_PATH.exists():
    raise FileNotFoundError("fruit_classifier.pth not found. Put it in the same folder as app.py.")

with open(CLASS_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    ),
])

INFO = {
    "freshapples": {
        "status": "Fresh Apple",
        "edibility": "Safe to eat",
        "advice": "Wash before eating.",
        "storage": "Keep in a cool place or refrigerator."
    },
    "freshbanana": {
        "status": "Fresh Banana",
        "edibility": "Safe to eat",
        "advice": "Good source of potassium.",
        "storage": "Keep at room temperature."
    },
    "freshoranges": {
        "status": "Fresh Orange",
        "edibility": "Safe to eat",
        "advice": "Rich in Vitamin C.",
        "storage": "Keep refrigerated."
    },
    "freshmangoes": {
        "status": "Fresh Mango",
        "edibility": "Safe to eat",
        "advice": "Sweet and ready to eat.",
        "storage": "Keep refrigerated after ripening."
    },
    "rottenapples": {
        "status": "Rotten Apple",
        "edibility": "Not safe to eat",
        "advice": "Do not consume. Throw it away.",
        "storage": "Keep away from fresh fruits."
    },
    "rottenbanana": {
        "status": "Rotten Banana",
        "edibility": "Not safe to eat",
        "advice": "Do not consume. Dispose of it.",
        "storage": "Keep away from fresh fruits."
    },
    "rottenoranges": {
        "status": "Rotten Orange",
        "edibility": "Not safe to eat",
        "advice": "Do not consume. It may contain mold.",
        "storage": "Dispose immediately."
    },
    "rottenmangoes": {
        "status": "Rotten Mango",
        "edibility": "Not safe to eat",
        "advice": "Do not consume. Throw it away.",
        "storage": "Keep away from fresh fruits."
    }
}

def clean_label(label):
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")

def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return "data:image/jpeg;base64," + encoded

def predict_image(img):
    img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    predictions = []

    for i, prob in enumerate(probs):
        predictions.append({
            "label": CLASS_NAMES[i],
            "confidence": float(prob.item() * 100)
        })

    predictions = sorted(predictions, key=lambda x: x["confidence"], reverse=True)

    top_label = predictions[0]["label"]
    top_confidence = predictions[0]["confidence"]
    cleaned_top_label = clean_label(top_label)

    print("Top label:", top_label)
    print("Cleaned label:", cleaned_top_label)
    print("Confidence:", top_confidence)
    print("All predictions:", predictions)

    if top_confidence < CONFIDENCE_LIMIT or cleaned_top_label not in INFO:
        return {
            "thumbnail": image_to_base64(img),
            "top_label": "No fruit detected",
            "top_confidence": top_confidence,
            "predictions": predictions,
            "properties": {
                "status": "No fruit detected",
                "edibility": "Unknown",
                "advice": "Please show a clear fruit image.",
                "storage": "No storage tip available."
            }
        }

    return {
        "thumbnail": image_to_base64(img),
        "top_label": cleaned_top_label,
        "top_confidence": top_confidence,
        "predictions": predictions,
        "properties": INFO[cleaned_top_label]
    }

@app.route("/")
def home():
    return render_template("index.html", classes=CLASS_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    print("Predict route called")

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    try:
        file = request.files["image"]
        img = Image.open(file.stream)
        result = predict_image(img)
        return jsonify(result)

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)})

@app.route("/predict_url", methods=["POST"])
def predict_url():
    print("Predict URL route called")

    data = request.get_json()
    url = data.get("url", "")

    if not url:
        return jsonify({"error": "No URL provided"})

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content))
        result = predict_image(img)
        return jsonify(result)

    except Exception as e:
        print("URL prediction error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)