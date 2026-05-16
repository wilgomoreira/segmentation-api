import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from xml.parsers.expat import model
 
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from torchvision import transforms as T
 
from src.evaluate import compute_iou, compute_dice
from src.model import UNet

logger = logging.getLogger(__name__)

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
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model_state["ready"] = True
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}. Train the model first.")

    logger.info(f"Model loaded on {device}")

    yield

    model_state.clear()
    logger.info("Model released")

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
    resize    = T.Resize((h, w))
    to_tensor = T.ToTensor()
    tensor = to_tensor(resize(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    mask = (torch.sigmoid(logits) > 0.5).squeeze().cpu().numpy()
    mask_image = Image.fromarray((mask * 255).astype("uint8"), mode="L")

    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")

@app.post("/evaluate")
async def evaluate(
    image: UploadFile = File(...),
    mask:  UploadFile = File(...),
):

    if not model_state.get("ready"):
        raise HTTPException(status_code=503, detail="Model not ready. Please wait while the model initializes.")

    # Read image (same as /predict)
    raw_image = await image.read()
    pil_image = Image.open(io.BytesIO(raw_image)).convert("RGB")

    # Read ground truth mask
    raw_mask = await mask.read()
    pil_mask = Image.open(io.BytesIO(raw_mask)).convert("L")

    model  = model_state["model"]
    device = model_state["device"]
    config = model_state["config"]

    h, w      = config["image_size"]
    resize    = T.Resize((h, w))
    to_tensor = T.ToTensor()

    # Inference
    tensor = to_tensor(resize(pil_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = (torch.sigmoid(logits) > 0.5).float()   # shape: (1, 1, H, W)

    # Ground truth: resize, convert to tensor, binarise
    gt = to_tensor(resize(pil_mask)).unsqueeze(0).to(device)
    gt = (gt > 0.5).float()           

    iou  = compute_iou(pred, gt)
    dice = compute_dice(pred, gt)

    return JSONResponse({
        "iou":  round(iou, 4),
        "dice": round(dice, 4),
    })