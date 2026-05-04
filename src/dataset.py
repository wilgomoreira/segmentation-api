import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class CarvanaDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.image_filenames = sorted(os.listdir(images_dir))

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        mask_name = img_name.replace(".jpg", "_mask.gif")

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        resize = T.Resize(self.image_size)
        to_tensor = T.ToTensor()

        image = to_tensor(resize(image))
        mask = to_tensor(resize(mask))

        mask = (mask > 0.5).float()

        return image, mask