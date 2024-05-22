#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=3dv
#SBATCH --output=%j.out

. /etc/profile.d/modules.sh
module add cuda/12.1
nvcc --version