# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import os
import collections
import numpy as np
import struct
import argparse
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


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


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


def read_cameras_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id,
                    model=model,
                    width=width,
                    height=height,
                    params=params,
                )
    return cameras


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


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


def read_images_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [
                        tuple(map(float, elems[0::3])),
                        tuple(map(float, elems[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
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
import struct

class CameraParams:
    def __init__(self, width, height, focal_length, px, py, kappa,model):
        self.w = width
        self.h = height
        self.focal_length = focal_length
        self.px = px
        self.py = py
        self.kappa = kappa
        self.model=model

def load_cameras_and_points(camera_filename):
    num_params = [3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12]
    
    print(f"Loading Cameras from [{camera_filename}] ...")
    
    cam_data = {}
    with open(camera_filename, 'rb') as fc:
        buffer = fc.read(8)
        ncams = struct.unpack('i', buffer[:4])[0]
        print(f" found {ncams} cameras")
        
        for i in range(ncams):
            buffer = fc.read(24)
            camera_id, model_id, width, height = struct.unpack('iiii', buffer[:16])
            np = num_params[model_id]
            #print(f"[{i + 1} out of {ncams}] cameraID = {camera_id}, modelID = {model_id}, width = {width}, height = {height}, np = {np}")
            buffer = fc.read(8 * np)
            if model_id == 2:
                focal_length, px, py, kappa = struct.unpack('dddd', buffer[:32])
                #print(f"f = {focal_length}, cx= {px}, cy= {py}, kappa= {kappa}")
                cp = CameraParams(width, height, focal_length, px, py, kappa,model_id)
                cam_data[camera_id] = cp
            elif model_id == 0:
                focal_length, px, py = struct.unpack('ddd', buffer[:24])
                cp = CameraParams(width, height, focal_length, px, py, 0,model_id)
                cam_data[camera_id] = cp
            else:
                print("modelID is neither 0 nor 2; not implemented yet!")
                assert Fail
                break

    return cam_data

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


def load_image_info(image_filename):
    print(f"Loading Image info from [{image_filename}] ...")
    buffer = bytearray(64)  # Assuming 64 bytes is the maximum needed based on your C++ code snippet
    rotations=[]
    translations=[]
    with open(image_filename, 'rb') as fi:
        #fi.readinto(buffer[0:8])
        #nimgs, = struct.unpack('i', buffer[0:4])
        num_reg_images = read_next_bytes(fi, 8, "Q")[0]

        print(f" found {num_reg_images} images")

        for i in range(num_reg_images):
            fi.readinto(buffer)
            imgID, = struct.unpack('i', buffer[0:4])
            qw = struct.unpack('d', buffer[4:12])[0]
            qx = struct.unpack('d', buffer[12:20])[0]
            qy = struct.unpack('d', buffer[20:28])[0]
            qz = struct.unpack('d', buffer[28:36])[0]
            tx = struct.unpack('d', buffer[36:44])[0]
            ty = struct.unpack('d', buffer[44:52])[0]
            tz = struct.unpack('d', buffer[52:60])[0]
            camID, = struct.unpack('i', buffer[60:64])

            # Using numpy to handle quaternion and rotation matrix
            qvec = np.array([qw, qx, qy, qz])
            #print(qvec)
            R = quat2mat(qvec)  # Convert quaternion to rotation matrix
            tvec = np.array([tx, ty, tz])
            # You can process R and tvec as needed
            rotations.append(R)
            translations.append(tvec)

    return rotations,translations

def quat2mat(q):
    """ Convert a quaternion into a 3x3 rotation matrix. """
    q = np.array(q, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.eye(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3], q[1, 2]-q[3, 0], q[1, 3]+q[2, 0]],
        [q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3], q[2, 3]-q[1, 0]],
        [q[1, 3]-q[2, 0], q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]
    ])





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


def write_images_binary(images, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def read_points3D_text(path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def write_points3D_text(points3D, path):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum(
            (len(pt.image_ids) for _, pt in points3D.items())
        ) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(
            len(points3D), mean_track_length
        )
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def detect_model_format(path, ext):
    if (
        os.path.isfile(os.path.join(path, "cameras" + ext))
        and os.path.isfile(os.path.join(path, "images" + ext))
        and os.path.isfile(os.path.join(path, "points3D" + ext))
    ):
        print("Detected model format: '" + ext + "'")
        return True

    return False


def read_model(path, ext=""):
    # try to detect the extension automatically
    if ext == "":
        if detect_model_format(path, ".bin"):
            ext = ".bin"
        elif detect_model_format(path, ".txt"):
            ext = ".txt"
        else:
            print("Provide model format: '.bin' or '.txt'")
            return

    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def write_model(cameras, images, points3D, path, ext=".bin"):
    if ext == ".txt":
        write_cameras_text(cameras, os.path.join(path, "cameras" + ext))
        write_images_text(images, os.path.join(path, "images" + ext))
        write_points3D_text(points3D, os.path.join(path, "points3D") + ext)
    else:
        write_cameras_binary(cameras, os.path.join(path, "cameras" + ext))
        write_images_binary(images, os.path.join(path, "images" + ext))
        write_points3D_binary(points3D, os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def main():
    parser = argparse.ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="outut model format",
        default=".txt",
    )
    args = parser.parse_args()

    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format
    )

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))

    if args.output_model is not None:
        write_model(
            cameras,
            images,
            points3D,
            path=args.output_model,
            ext=args.output_format,
        )

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



if __name__ == "__main__":
    #main()
    dir_path="..\\data\\scene1\\colmap\\images"
    path_to_save="..\\dense\\images\\01"

    camera_intrinsics=read_cameras_binary("..\\data\\scene1\\colmap\\sparse\\0\\cameras.bin")
    #print(camera_intrinsics)
    camera_poses=read_images_binary("..\\data\\scene1\\colmap\\sparse\\0\\images.bin")
    #print(camera_poses[0])

    cemra_intrisics_meth2=load_cameras_and_points("..\\data\\scene1\\colmap\\sparse\\0\\cameras.bin")
    count=0
    count2=0
    fs=[]
    kappas=[]


    #Rotations,translations=load_image_info("..\\data\\scene1\\colmap\\sparse\\0\\images.bin")

    for (image_id, image),(image_id_2,camera),(image_id_3,camera_meth2) in zip(camera_poses.items(),camera_intrinsics.items(),cemra_intrisics_meth2.items()):
        image_path = image.name
        #print(image_path)
        path=os.path.join(dir_path, os.path.dirname(image_path))
        #print(path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        #print("image_name "+image_name)
        path_camera_intrinsics=path_to_save+"\\"+image_name+".intrinsics.txt"
        print(path_camera_intrinsics)
        path_camera_pose=path_to_save+"\\"+image_name+".pose.txt"
        #print(path_camera_intrinsics)
        #print(path_camera_pose)
        # print(camera.model)
        # print(camera.width)
        # print(camera.height)
        # print(camera.params)
        
        if(camera.model is "SIMPLE_RADIAL"):
            count+=1
        else:
            count2+1


        fs.append(camera_meth2.focal_length)
        kappas.append(camera_meth2.kappa)

        # print("w: "+str(camera_meth2.w))
        # print("h: "+str(camera_meth2.h))
        # print("f: "+str(camera_meth2.focal_length))
        # print("px"+str(camera_meth2.px))
        # print("py: "+str(camera_meth2.py))
        # print("kappa: "+str(camera_meth2.kappa))





        # Define the translation vector
        translation = image.tvec

        # Define the quaternion (w, x, y, z format)
        quaternion = image.qvec  
        # Create a rotation matrix from the quaternion
        rotation_matrix = R.from_quat(quaternion).as_matrix()

        # Create the transformation matrix
        transformation_matrix = np.eye(4)  # Start with an identity matrix
        transformation_matrix[:3, :3] = rotation_matrix  # Insert the rotation matrix
        transformation_matrix[:3, 3] = translation       # Insert the translation vector
        first_three_rows = transformation_matrix[:-1, :]
        matrix_string = "\n".join(" ".join(f"{value:.6f}" for value in row) for row in first_three_rows)
        


        # print(translation)
        # print(rotation_matrix)
        # print("\n")
        # print(translations[image_id])
        # print(Rotations[image_id])
        

        with open(path_camera_intrinsics, 'w') as file:
             print(image.name)
             file.write(f"{camera.width} {camera.height} {camera_meth2.focal_length:10f} {camera_meth2.px:10f} {camera_meth2.py:10f} {camera_meth2.kappa:10f} {os.path.basename(image.name)}\n")

        with open(path_camera_pose, 'w') as file:
            for i in range(3):         
                file.write(f"{rotation_matrix[i, 0]:10f} {rotation_matrix[i, 1]:10f} {rotation_matrix[i, 2]:10f} {translation[i]:10f}\n")  # Write a single line to the file
        print("round done")
    print(count)
    print(count2)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first histogram
    axs[0].hist(kappas, bins=30, color='blue', alpha=0.7)
    axs[0].set_title('kappas Distribution')
    axs[0].set_xlabel('Values')
    axs[0].set_ylabel('kappa')

    # Plot the second histogram
    axs[1].hist(fs, bins=30, color='green', alpha=0.7)
    axs[1].set_title('f Distribution')
    axs[1].set_xlabel('Values')
    axs[1].set_ylabel('focal length')

    # Display the plots
    plt.tight_layout()
    plt.show()