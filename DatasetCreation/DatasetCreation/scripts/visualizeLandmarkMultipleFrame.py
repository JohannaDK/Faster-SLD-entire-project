

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pickle
import numpy as np
import fnmatch
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
dataset_folder="../data/"
scene='sceneB/'
image_folder=os.path.join(dataset_folder,scene, 'extractedData/')
landmark_file_name='landmarks-30010'
visbility_file_name='visibility-30010'


 
landmark_file = open(os.path.join(dataset_folder,scene,'landmarks/'+landmark_file_name+'.txt'),'r')
landmarks=np.loadtxt(landmark_file,skiprows=1,usecols=(1, 2, 3))


visibility_file = open(os.path.join(dataset_folder,scene,'landmarks/'+visbility_file_name+'.txt'),'r')
visibility = np.loadtxt(visibility_file).astype(bool)

image_files_all = fnmatch.filter(os.listdir(image_folder), '*.color.jpg')
image_files_all = sorted(image_files_all)



indoor6_images = pickle.load(open(os.path.join(dataset_folder,scene, 'train_test_val.pkl'), 'rb'))

indoor6_imagename_to_index = {}
for field in ['train']:
    for i, f in enumerate(indoor6_images[field]):
        image_name = open(os.path.join(dataset_folder, 
                                    scene, 'extractedData', 
                                    f.replace('color.jpg', 
                                                'intrinsics.txt'))).readline().split(' ')[-1][:-1]
        indoor6_imagename_to_index[image_name] = indoor6_images['train_idx'][i]



numofImages=4
images_list=[]
landmarks_list=[]
for land_id,land in enumerate(landmarks):
    images_list=[]
    landmarks_list=[]
    for image_id,im in enumerate(image_files_all):
        if(im in indoor6_images['train']):

            C_T_G = np.loadtxt(os.path.join(image_folder, im.replace('color.jpg', 'pose.txt')))
            color_img = Image.open(os.path.join(image_folder, im))
            intrinsics = open(os.path.join(image_folder,im.replace('color.jpg', 'intrinsics.txt')))
            intrinsics = intrinsics.readline().split()
            index_1=indoor6_images['train'].index(im)
            imageName=intrinsics[6]
            index_2=indoor6_imagename_to_index[imageName]   


            image_downsampled=1
            W = int(intrinsics[0]) // (image_downsampled * 32) * 32
            H = int(intrinsics[1]) // (image_downsampled * 32) * 32

            scale_factor_x = W / float(intrinsics[0])
            scale_factor_y = H / float(intrinsics[1])

                                      
            fx =float(intrinsics[2]) * scale_factor_x
            fy =float(intrinsics[2]) * scale_factor_y

            cx = float(intrinsics[3])
            cy = float(intrinsics[4])

            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]], dtype=float)

            K_inv = np.linalg.inv(K)


            proj = K @ (C_T_G[:3, :3] @ land.reshape((3, 1)) + C_T_G[:3, 3:])
 
            landmark2d = proj / proj[2:]

            inside_patch = (landmark2d[0] < W) *  (landmark2d[0] >= 0) * (landmark2d[1] < H) * (landmark2d[1] >= 0)  # L vector

            _mask1 = visibility[land_id, index_2] * inside_patch


            if(_mask1):
                coordinate = (landmark2d[0], landmark2d[1])
                image = np.array(color_img)

                
                images_list.append(image)
                landmarks_list.append([landmark2d[0], landmark2d[1]])

                #plt.imshow(image)  # Display the image
                #plt.scatter([landmark2d[0]], [landmark2d[1]], c='red', s=40)
                #plt.show()
                # Display the image
        



            if((len(images_list)%numofImages==0) and len(images_list)!=0):
                # Calculate the number of rows needed
                num_rows = 2  # Add 2 to n to ensure proper rounding up

                # Create a figure with the appropriate size
                plt.figure(figsize=(10, 5 * num_rows))

                # Loop through the images and plot each one
                for i in range(numofImages):
                    plt.subplot(num_rows, 2, i + 1)  # num_rows rows, 3 columns, index i+1
                    plt.imshow(images_list[i])  # Display the image
                    plt.scatter([landmarks_list[i][0]], [landmarks_list[i][1]], c='red', s=120)
                    plt.axis('off')  # Turn off axis
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                plt.tight_layout(pad=0.5)  # Adjust the layout to make images closer
                plt.show()
                break


