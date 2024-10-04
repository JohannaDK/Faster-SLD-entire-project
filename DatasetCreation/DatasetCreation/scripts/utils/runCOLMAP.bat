@echo off

REM Check that an argument was provided.
if "%1"=="" (
    echo    Usage: runCOLMAP.bat [Scene]
    exit /b
)

set DATASET_DIR=..\data
set DATASET=%DATASET_DIR%\%1\colmap\

call ..\colmap\COLMAP feature_extractor --database_path %DATASET%\database.db --image_path %DATASET%\images
call ..\colmap\COLMAP vocab_tree_matcher --database_path %DATASET%\database.db --VocabTreeMatching.vocab_tree_path ..\colmap\voc\vocab_tree_flickr100K_words256K.bin

md %DATASET%\sparse

call ..\colmap\COLMAP mapper --database_path %DATASET%\database.db --image_path %DATASET%\images --output_path %DATASET%\sparse --Mapper.ba_global_images_ratio 1.3 --Mapper.ba_global_points_ratio 1.3 --Mapper.ba_global_max_num_iterations 20 --Mapper.ba_global_max_refinements 2
