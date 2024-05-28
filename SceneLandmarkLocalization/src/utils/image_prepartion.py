import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image

def load_and_prepare_data(image_path, depth_map_path_z1, depth_map_path_z2):
    color_image = Image.open(image_path)
    color_image = ToTensor()(color_image)  # Shape: [3, H, W]

    z1 = np.load(depth_map_path_z1)
    z1 = (z1 - np.min(z1)) / (np.max(z1) - np.min(z1))  
    z1 = torch.tensor(z1, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]

    # Load the target depth map z2
    z2 = np.load(depth_map_path_z2)
    z2 = (z2 - np.min(z2)) / (np.max(z2) - np.min(z2))  
    z2 = torch.tensor(z2, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]

    return color_image, z1, z2
