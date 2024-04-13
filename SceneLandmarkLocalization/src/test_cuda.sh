#!/bin/bash

#SBATCH --account=3dv
#SBATCH --output=%j.out
#SBATCH --gres=gpu:nvidia_geforce_gtx_1080ti:4

. /etc/profile.d/modules.sh
module add cuda/11.8
srun python3 test.py