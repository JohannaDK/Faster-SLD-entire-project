@echo off

REM Check that an argument was provided.
if "%1"=="" (
    echo    Usage: extractFrames.bat [Scene]
    exit /b
)

REM Before running the script, copy all the mapping videos (train split) into ..\data\%SCENE%\input\ and name them as 00.ext, 01.ext, 02.ext, etc., where ext = video file extension.
REM The script will run ffmpeg on each video and generate subfolders called %SCENE%\images\00, %SCENE%\images\01, %SCENE%\images\02, etc. and will dump the video frames as png images into those subfolders.  

REM Set the frame-rate here.
set FPS=4

REM The name of the folder for a specific scene.
set SCENE=%1

set INPUT_DIR=..\data\%SCENE%\input\
set COLMAP_DIR=..\data\%SCENE%\colmap\
set FRAMES_DIR=%COLMAP_DIR%\images\
set MARKERS_DIR=%COLMAP_DIR%\markers\

mkdir %COLMAP_DIR%
mkdir %FRAMES_DIR%

for %%i in (%INPUT_DIR%\*.MOV) do (
	echo %%i
	mkdir %FRAMES_DIR%\%%~ni
	mkdir %MARKERS_DIR%\%%~ni
	..\ffmpeg-4.4-full_build\bin\ffmpeg.exe -i %%i -vf fps=%FPS% %FRAMES_DIR%\%%~ni\frame%%06d.png
)
	