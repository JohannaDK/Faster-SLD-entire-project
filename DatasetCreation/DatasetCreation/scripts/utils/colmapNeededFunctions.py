import argparse
import numpy as np
import os
import struct
import pylab as plt
import matplotlib.image as mpimg
import numpy as np
import pickle
import collections
import random
class Image:
    def __init__(self, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.id = id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

class Camera:
    def __init__(self, id, model, width, height, params):
        self.id = id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)
def read_array(path):

    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")

    return np.transpose(array, (1, 0, 2)).squeeze()

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras



def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        print("images: "+str(num_reg_images))
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images



def createTransformationMatrix(R,t):
    transformation_matrix = np.eye(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = R  # Insert the rotation matrix
    transformation_matrix[:3, 3] = t       # Insert the translation vector
    first_three_rows = transformation_matrix[:-1, :]
    return first_three_rows


def createPartitionList(numberOfItems,partition1,partition2):
    num_ones = int(numberOfItems * partition1)  # 70% of size
    num_twos = int(numberOfItems * partition2)  # 15% of size
    num_threes = numberOfItems - num_ones - num_twos
    result_list = [1] * num_ones + [2] * num_twos + [3] * num_threes
    random.shuffle(result_list)
    return result_list


def quaternion_to_rotation_matrix(q):
    q_w, q_x, q_y, q_z = q
    # Compute the elements of the rotation matrix
    R = np.array([
        [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x * q_y - q_z * q_w), 2 * (q_x * q_z + q_y * q_w)],
        [2 * (q_x * q_y + q_z * q_w), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y * q_z - q_x * q_w)],
        [2 * (q_x * q_z - q_y * q_w), 2 * (q_y * q_z + q_x * q_w), 1 - 2 * (q_x**2 + q_y**2)]
    ])
    return R


def findNeededCamera(Cameras_list,ID):
    print("camera not found")
    for camera_key,camera in Cameras_list:
        if(camera.id==ID):
            return camera
    assert False



def write_cameras_text(cameras, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_images_text(images, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum(
            (len(img.point3D_ids) for _, img in images.items())
        ) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(
            len(images), mean_observations
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [
                img.id,
                *img.qvec,
                *img.tvec,
                img.camera_id,
                img.name,
            ]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def adjust_image_size(image_np,W,H):
    

    width_diff = W - image_np.shape[1]
    height_diff = H - image_np.shape[0]

    # Padding
    if width_diff > 0 or height_diff > 0:
        pad_width = ((max(0, height_diff // 2), max(0, (height_diff + 1) // 2)),
                     (max(0, width_diff // 2), max(0, (width_diff + 1) // 2)))  # Only pad the first two dimensions (height and width)
        image_np = np.pad(image_np, pad_width, mode='edge')

    # Cropping
    if width_diff < 0 or height_diff < 0:
        start_x = max(0, -width_diff // 2)
        start_y = max(0, -height_diff // 2)
        end_x = start_x + W
        end_y = start_y + H
        image_np = image_np[start_y:end_y, start_x:end_x]

    # Ensure the final size is exactly target_size
    image_np = image_np[:H, :W]
    return image_np