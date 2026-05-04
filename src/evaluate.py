import torch
from tqdm import tqdm


def evaluate(model, loader, device):
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            iou = compute_iou(preds, masks)
            dice = compute_dice(preds, masks)

            total_iou += iou
            total_dice += dice
            num_batches += 1

    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    return {"iou": avg_iou, "dice": avg_dice}


def compute_iou(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = (preds + masks - preds * masks).sum(dim=(1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def compute_dice(preds, masks):
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
    return dice.mean().item()