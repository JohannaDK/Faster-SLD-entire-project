import os
import random


# This script is used to divide the videos into frames and run colmap for the sparse recondstruction

dataPath="../data/"

#changed based on sceneName
sceneName="sceneB"
videoFolder="input"


videosPath=os.path.join(dataPath,sceneName,videoFolder)

files = os.listdir(videosPath)

#need to be specified
test=0.2
train=0.8

#Create a split of the video between training and testing
if not("train_files.txt"in files and "test_files.txt" in files):
    print("hi")
    files = [file for file in files if file.lower().endswith('.mov')]
    count_videos=sum(1 for file in files if file.lower().endswith('.mov'))
    random.shuffle(files)
    split_index=int(train*count_videos)
    train_files = files[:split_index]
    test_files = files[split_index:]
    # Save the train file names
    with open(os.path.join(videosPath,'train_files.txt'), 'w') as f:
        for file in train_files:
            if(file.lower().endswith('.mov')):
                f.write(f"{file}\n")

    # Save the test file names
    with open(os.path.join(videosPath,'test_files.txt'), 'w') as f:
        for file in test_files:
            if(file.lower().endswith('.mov')):
                f.write(f"{file}\n")

os.system('utils\\extractFrames.bat '+sceneName)
os.system('utils\\runCOLMAP.bat '+sceneName)
os.makedirs(os.path.join(dataPath,sceneName+"/","colmap/dense"))
