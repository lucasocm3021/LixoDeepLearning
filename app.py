from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torch import nn
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "reciclaai.pth"

# Rótulos e cores de exibição em português para cada classe do dataset
DISPLAY_INFO = {
    "cardboard": {"label": "Papelão", "color": "#b08968"},
    "glass": {"label": "Vidro", "color": "#2a9d8f"},
    "metal": {"label": "Metal", "color": "#6c757d"},
    "paper": {"label": "Papel", "color": "#457b9d"},
    "plastic": {"label": "Plástico", "color": "#e07a5f"},
    "trash": {"label": "Rejeito", "color": "#3d3d3d"},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

_model: nn.Module | None = None
_classes: list[str] = []

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model() -> None:
    global _model, _classes

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode 'python treino.py' primeiro."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    _classes = checkpoint["classes"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(_classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    _model = model
    print(f"Modelo carregado ({device}). Classes: {_classes}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if _model is None:
        return jsonify({"error": "Modelo não carregado no servidor."}), 500

    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "Nenhuma imagem enviada."}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Arquivo de imagem inválido."}), 400

    tensor = _transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    ranked = sorted(
        (
            {
                "class": cls,
                "label": DISPLAY_INFO.get(cls, {}).get("label", cls),
                "color": DISPLAY_INFO.get(cls, {}).get("color", "#888888"),
                "confidence": round(prob.item() * 100, 1),
            }
            for cls, prob in zip(_classes, probabilities)
        ),
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return jsonify({"prediction": ranked[0], "ranking": ranked})


if __name__ == "__main__":
    load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
