from contextlib import asynccontextmanager
from pathlib import Path
import torch
import yaml
from fastapi import FastAPI
from src.model import UNet
from src.utils import load_checkpoint
import io
from fastapi import File, HTTPException, UploadFile
from PIL import Image
from fastapi.responses import Response
import numpy as np

model_state = {}

def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
    ).to(device)

    model.eval()
    model_state["model"] = model
    model_state["device"] = device
    model_state["config"] = config
    model_state["ready"] = False

    checkpoint_path = config["model_save_path"]

    if Path(checkpoint_path).exists():
        load_checkpoint(checkpoint_path, model, device=device)
        model_state["ready"] = True
        print(f"Checkpoint loaded from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Train the model first.")

    print(f"Model loaded on {device}")

    yield

    model_state.clear()
    print("Model released")

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(model_state.get("device")),
        "model_ready": model_state.get("ready"),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not model_state.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="No checkpoint loaded. Train the model first."
        )

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type: {file.content_type}. Send a JPEG or PNG."
        )

    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    model = model_state["model"]
    device = model_state["device"]
    config = model_state["config"]

    h, w = config["image_size"]
    image = image.resize((w, h))
    tensor = torch.from_numpy(
    np.array(image, dtype="float32") / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
    mask_image = Image.fromarray((mask * 255).astype("uint8"), mode="L")

    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")