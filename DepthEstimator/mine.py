import numpy as np
from PIL import Image
from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor
import matplotlib.pyplot as plt
import cv2
import logging

logging.basicConfig(level=logging.INFO)

print("Starting the script")

image_path = "/Users/alenb/UZH/3DV/Faster-SLD/Depth-Anything/media/ethimage.jpeg"
image = Image.open(image_path)
print("Image loaded successfully")

model_path = "/Users/alenb/UZH/3DV/Faster-SLD/depth-anything-small-hf"
print("Loading model and processor from local path")
model = AutoModelForDepthEstimation.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

print("Creating pipeline")
pipe = pipeline(task="depth-estimation", model=model, feature_extractor=processor)
print("Pipeline created successfully")

depth_map = pipe(image)
print("Depth estimation completed")

depth_map = depth_map['depth']
if isinstance(depth_map, list):
    depth_map = np.array(depth_map)
elif isinstance(depth_map, Image.Image):
    depth_map = np.array(depth_map)

depth_map = depth_map.astype(np.float32)
print("Depth map converted to float32")

depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
print("Depth map normalized")

depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
print("Gaussian blur applied to depth map")

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
print("Images displayed")

color_image = np.asarray(image)
depth_image = depth_map

depth_image = (depth_image * 255).astype(np.uint8)
print("Depth image converted to uint8")

# Save the depth image as a JPEG file
depth_image_pil = Image.fromarray(depth_image)
depth_image_pil.save("/Users/alenb/UZH/3DV/Faster-SLD/Depth-Anything/media/depth_map.jpeg")
print("Depth map saved as JPEG")
