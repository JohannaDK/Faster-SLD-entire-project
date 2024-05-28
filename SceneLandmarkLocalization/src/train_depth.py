import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_depth import DepthDataset
from models.unet_model import UNet
from utils import load_and_prepare_data

image_path = "../data/color_image.jpg"
depth_map_path_z1 = "../data/depth_map_z1.npy"
depth_map_path_z2 = "../data/depth_map_z2.npy"

color_image, z1, z2 = load_and_prepare_data(image_path, depth_map_path_z1, depth_map_path_z2)

dataset = DepthDataset(color_image, z1, z2)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

model = UNet(n_channels=4, n_classes=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100  
for epoch in range(num_epochs):
    for X, y in dataloader:
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'unet_model.pth')
