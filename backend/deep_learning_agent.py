import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from .models.cnn_model import CNNModel
from .config import MODEL_CHECKPOINT

model = CNNModel()


def _load_checkpoint_if_available():
    checkpoint_path = Path(__file__).resolve().parents[1] / MODEL_CHECKPOINT
    if not checkpoint_path.exists():
        print(f"[deep_learning_agent] Warning: checkpoint not found at {checkpoint_path}. Using untrained CNN.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


_load_checkpoint_if_available()
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def deep_learning_detection(image):
    img = Image.fromarray((image*255).astype('uint8'))
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        score = torch.sigmoid(output).item()

    return score
