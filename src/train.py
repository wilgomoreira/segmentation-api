import logging
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.dataset import CarvanaDataset
from src.model import UNet
from src.evaluate import compute_iou, compute_dice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = CarvanaDataset(
        images_dir=config["images_dir"],
        masks_dir=config["masks_dir"],
        image_size=tuple(config["image_size"])
    )

    val_size = int(len(dataset) * config["val_split"])
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)

    logger.info(f"Train samples: {train_size} | Val samples: {val_size}")

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    # Best checkpoint tracking
    best_iou = 0.0
    best_checkpoint_path = config["model_save_path"].replace(".pth", "_best.pth")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("carvana-unet")

    with mlflow.start_run():

        mlflow.log_params({
            "epochs":        config["epochs"],
            "batch_size":    config["batch_size"],
            "learning_rate": config["learning_rate"],
            "val_split":     config["val_split"],
            "image_size":    str(config["image_size"]),
            "optimizer":     "Adam",
            "loss":          "BCEWithLogitsLoss",
            "device":        str(device),
        })

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            val_iou  = 0.0
            val_dice = 0.0

            with torch.no_grad():
                for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    val_loss += loss.item()
                    val_iou  += compute_iou(preds, masks)
                    val_dice += compute_dice(preds, masks)

            val_loss /= len(val_loader)
            val_iou  /= len(val_loader)
            val_dice /= len(val_loader)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_iou":    val_iou,
                "val_dice":   val_dice,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"Dice: {val_dice:.4f}"
            )

            # Save best checkpoint
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), best_checkpoint_path)
                mlflow.log_metric("best_iou", best_iou, step=epoch)
                logger.info(f"New best model saved at epoch {epoch+1} with IoU: {best_iou:.4f}")

        # Save last checkpoint and log artifacts
        torch.save(model.state_dict(), config["model_save_path"])
        mlflow.log_artifact(config["model_save_path"])
        mlflow.log_artifact(best_checkpoint_path)
        mlflow.log_artifact("configs/config.yaml")

        logger.info(f"Last checkpoint saved to {config['model_save_path']}")
        logger.info(f"Best checkpoint saved to {best_checkpoint_path} with IoU: {best_iou:.4f}")