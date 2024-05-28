import torch
from torch.utils.data import Dataset

class DepthDataset(Dataset):
    def __init__(self, color_image, z1, z2):
        self.rgbd_image = torch.cat((color_image, z1), dim=0)  # Shape: [4, H, W]
        self.z2 = z2  

    def __len__(self):
        return 1  
    
    def __getitem__(self, idx):
        return self.rgbd_image, self.z2
