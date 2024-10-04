import os


if __name__ == '__main__':

    dataset_folder = "../data/"
    scene_id="sceneB/"

    num_landmarks=300
    output_format=10
    cmd = 'python ./landmark_selection.py '
    cmd += ' --dataset_folder %s' % dataset_folder
    cmd += ' --scene_id %s' % scene_id
    cmd += ' --num_landmarks %d' % num_landmarks
    cmd += ' --output_format %d' % output_format
 

    # Launch training
    os.system(cmd)
