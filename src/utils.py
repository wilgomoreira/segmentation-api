import torch
import numpy as np
import matplotlib.pyplot as plt


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]

def visualize_prediction(image, mask, pred, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred.squeeze().cpu().numpy(), cmap="gray")
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()