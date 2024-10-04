
import cv2
from PIL import Image
import os
import collections
import numpy as np
import struct
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import random
import pickle
from utils.colmapNeededFunctions import *


#Get the undistorted images, intrisics, camera poses and the depth maps saved baccoridng to the needed structure
dataPath="../data/"

sceneName="sceneB/"
videoFolder="input"

depthMapsFolder="colmap/dense/stereo/depth_maps/"
undistortedImagesFolder="colmap/dense/images/"

saveDenseMaps=os.path.join(dataPath,sceneName)+"/denseMaps/"
saveReconstructionData=os.path.join(dataPath,sceneName)+"/extractedData/"


dir_path_to_depth_maps=os.path.join(dataPath,sceneName,depthMapsFolder)



if(not os.path.exists(saveDenseMaps)):
    os.makedirs(saveDenseMaps)
if(not os.path.exists(saveReconstructionData)):
    os.makedirs(saveReconstructionData)

cameras_data=read_cameras_binary(os.path.join(dataPath,sceneName,"colmap/dense/sparse/cameras.bin"))
images_data=read_images_binary(os.path.join(dataPath,sceneName,"colmap/dense/sparse/images.bin"))

pickleFileData={'train':[],'train_idx':[],'test':[], 'test_idx': [],'val':[],'val_idx':[]}

train_parition=0.7
val_parition=0.15

partition_list=createPartitionList(len(images_data),train_parition,val_parition)

focal_length_list=[]
round=0

W=480
H=853
#write_images_text(images_data,"images.txt")
#write_cameras_text(cameras_data,"cameras.txt")

#save the intrinsics, pose and undistorted images
for (image_key,image),(camera_key,camera) in zip(images_data.items(),cameras_data.items()):

    if not (camera.id==image.camera_id):
        camera=findNeededCamera(cameras_data.items(),image.camera_id)


    ID_image=image.id-1
    image_path=image.name
    path_camera_intrinsics=saveReconstructionData+"image-"+"{:05}".format(round)+".intrinsics.txt"
    path_camera_pose=saveReconstructionData+"\\"+"image-"+"{:05}".format(round)+".pose.txt"

    
    if(partition_list[round]==1):
            pickleFileData['train'].append("image-"+"{:05}".format(round)+".color.jpg")
            pickleFileData['train_idx'].append(int("{:05}".format(round)))
    elif(partition_list[round]==2):
            pickleFileData['test'].append("image-"+"{:05}".format(round)+".color.jpg")
            pickleFileData['test_idx'].append(int("{:05}".format(round)))
    else:
            pickleFileData['val'].append("image-"+"{:05}".format(round)+".color.jpg")
            pickleFileData['val_idx'].append(int("{:05}".format(round)))

    if(camera.model is "SIMPLE_RADIAL"):
            focal_length, px, py, kappa=camera.params
    else:
            focal_length,fy, px, py =camera.params
            kappa=0
            assert focal_length==fy
    focal_length_list.append(focal_length)
    translation = image.tvec
    quaternion = image.qvec  
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)

    # Create the transformation matrix
    transformation_matrix_string = "\n".join(" ".join(f"{value:.6f}" for value in row) for row in createTransformationMatrix(rotation_matrix,translation))

    with open(path_camera_intrinsics, 'w') as file:
        file.write(f"{camera.width} {camera.height} {focal_length:10f} {px:10f} {py:10f} {kappa:10f} {image.name}\n")

    with open(path_camera_pose, 'w') as file:
        for i in range(3):         
            file.write(f"{rotation_matrix[i, 0]:10f} {rotation_matrix[i, 1]:10f} {rotation_matrix[i, 2]:10f} {translation[i]:10f}\n")  # Write a single line to the file
    undistorted_image_path=os.path.join(undistortedImagesFolder,image_path)
    img = cv2.imread(os.path.join(dataPath,sceneName,undistorted_image_path))
    
    cv2.imwrite(saveReconstructionData+"/"+"image-"+"{:05}".format(round)+".color.jpg", img)
    
    full_depthMap_path = os.path.join(dir_path_to_depth_maps,image_path.replace(".png",".png.geometric.bin"))
    depth_map=read_array(full_depthMap_path)

 

    depth_map_2=adjust_image_size(read_array(full_depthMap_path),W,H)
    assert np.shape(depth_map_2)==(H,W)
    np.set_printoptions(threshold=np.inf)
    min_depth, max_depth = np.percentile(
    depth_map, [5, 95])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    plt.imsave(saveDenseMaps+"dp_adjusted_"+"{:05}".format(round)+".jpg",depth_map)
    plt.imsave(saveDenseMaps+"dp_original_"+"{:05}".format(round)+".jpg",depth_map_2)
    np.save(saveDenseMaps+"image-"+"{:05}".format(round)+'.color.scaled_depth.npy', depth_map_2)
    round+=1
    

#save the pickle file with the image partition as well as separate folders with the image names for the different parititions
with open(os.path.join(dataPath,sceneName[:-1],'train_test_val.pkl'), 'wb') as f:
    pickle.dump(pickleFileData, f)

with open(os.path.join(dataPath,sceneName,sceneName[:-1]+"_train.txt"), "w") as file:
    for entry in  pickleFileData['train']:
        file.write(entry + "\n")

with open(os.path.join(dataPath,sceneName,sceneName[:-1]+"_test.txt"), "w") as file:
    for entry in  pickleFileData['test']:
        file.write(entry + "\n")

with open(os.path.join(dataPath,sceneName,sceneName[:-1]+"_val.txt"), "w") as file:
    for entry in  pickleFileData['val']:
        file.write(entry + "\n")



plt.hist(focal_length_list)
plt.title('Histogram of Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
