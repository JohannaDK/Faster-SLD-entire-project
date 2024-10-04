
import argparse
import numpy as np
import os
import pickle
from utils.read_write_models import qvec2rotmat, read_model,write_images_text
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.

    Parameters:
    q (tuple or list): Quaternion (q_w, q_x, q_y, q_z)

    Returns:
    numpy.ndarray: 3x3 rotation matrix
    """
    q_w, q_x, q_y, q_z = q

    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
    ])

    return R

dataPath="../data/"

sceneName="sceneC/"

path = os.path.join(dataPath,sceneName,'colmap/dense/sparse/')
print(path)
cameras, images, points3D = read_model(path, ext='.bin')

#write_images_text(images,'../data/scene1/imagestext.txt')
print(len(images))

path_to_data=os.path.join(dataPath,sceneName,'extractedData/')

for i, k in enumerate(tqdm(points3D)):            
        pointInGlobal = points3D[k].xyz
        print(pointInGlobal)
        #print(pointInGlobal)
        image_ids = points3D[k].image_ids
        point_id=points3D[k].id

        trackLength = len(image_ids)
        print(trackLength)
        if(trackLength>25):
            for l in image_ids:
                image_name=images[l].name
                print(image_name)
                print("image id",l)
                path_to_image=os.path.join(dataPath,sceneName,'colmap/dense/images/')+image_name
                indexes=images[l].point3D_ids
                for index,v in enumerate(indexes):
                    
                    if(v==point_id):
                        coordinates=images[l].xys[index]

                formated_l = "{:05d}".format(l)
                C_T_G = np.loadtxt(os.path.join(path_to_data,'image-'+formated_l+'.pose.txt'))
                intrinsics = open(os.path.join(path_to_data,'image-'+formated_l+'.intrinsics.txt'))

                intrinsics = intrinsics.readline().split()
                image_downsampled=1
                W = int(intrinsics[0])
                H = int(intrinsics[1]) 

                scale_factor_x = W / float(intrinsics[0])
                scale_factor_y = H / float(intrinsics[1])

                                  
                fx =float(intrinsics[2]) 
                fy =float(intrinsics[2]) 

                cx = float(intrinsics[3])
                cy = float(intrinsics[4])

                K = np.array([[fx, 0., cx],
                              [0., fy, cy],
                              [0., 0., 1.]], dtype=float)



                proj = K @ (C_T_G[:3, :3] @ pointInGlobal.reshape((3, 1)) + C_T_G[:3, 3:])
                landmark2d = proj / proj[2:]


                print("t",images[l].tvec)
                print("t2",C_T_G[:3, 3:])

                print("\nR",quaternion_to_rotation_matrix(images[l].qvec))
                print("\nR2",C_T_G[:3, :3])
                print("\ndiff",C_T_G[:3, :3]-quaternion_to_rotation_matrix(images[l].qvec))


                print('landmark2d',landmark2d)
                print('coordinate',coordinates)
               

                image = mpimg.imread(path_to_image)
                plt.imshow(image)
                plt.scatter([coordinates[0]], [coordinates[1]], c='red', s=40)
                plt.show()

        