Code Needed to Generate a Dataset with Landmarks:

For each of the Python scripts, the scene has to be chosen.
1.Create a scene folder in the data directory.
2.Add the videos taken for a scene to the input folder.
3.Run SparseReconstruction.py in the utils folder after specifying the scene name in it. This will generate the frames and the COLMAP sparse reconstruction.
4.Run the dense reconstruction manually through the COLMAP interface.
5.Run extractData.py to get the camera poses, intrinsics, undistorted images, and dense maps saved in the required data folder.
6.Run the landmarks generation file run_landmark_selection.py.
7.Run the adjusted visibility generation file run_generate_visibility_depth_normal.py.

Visualization:

visualizeLandmarkMultipleFrame.py can be used to see landmarks in different extracted images.
