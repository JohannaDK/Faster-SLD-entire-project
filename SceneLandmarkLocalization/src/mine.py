import numpy as np
from PIL import Image
import open3d as o3d
from transformers import pipeline
import matplotlib.pyplot as plt
import cv2

image_path = "/media/ball.jpeg" 
image = Image.open(image_path)

pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth_map = pipe(image)

depth_map = depth_map['depth']
if isinstance(depth_map, list):
    depth_map = np.array(depth_map)
elif isinstance(depth_map, Image.Image):
    depth_map = np.array(depth_map)

depth_map = depth_map.astype(np.float32)

depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)

# Display the image and depth map
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap='plasma')
plt.title("Depth Map")
plt.axis("off")

plt.show()

color_image = np.asarray(image)
depth_image = depth_map

depth_image = (depth_image * 255).astype(np.uint8)

depth_o3d = o3d.geometry.Image(depth_image)
color_o3d = o3d.geometry.Image(color_image)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d, convert_rgb_to_intensity=False)

camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)


pcd.transform([[1, 0, 0, 0],
               [0, -1, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])
