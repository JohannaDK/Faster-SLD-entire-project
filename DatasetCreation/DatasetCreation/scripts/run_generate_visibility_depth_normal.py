import os
if __name__ == '__main__':
    scene_id='sceneB/'
    mode='all'
    dataset_folder="../data/"
    landmark_config="landmarks/landmarks-30010"
    visibility_config="landmarks/visibility-30010"
    output_folder="refinedData/"

    
    cmd = 'python ./generate_visibility_depth_normal.py '
    cmd += ' --dataset_folder %s' % dataset_folder
    cmd += ' --output_folder %s' % output_folder
    cmd += ' --scene_id %s' % scene_id
    cmd += ' --landmark_config %s' % landmark_config
    cmd += ' --visibility_config %s' % visibility_config
   

    # Launch training
    os.system(cmd)