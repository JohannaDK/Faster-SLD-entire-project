#!/bin/bash

#SBATCH --time=00:10
#SBATCH --account=3dv
#SBATCH --output=%j.out

. /etc/profile.d/modules.sh
module add cuda/11.8
srun python3 test.py