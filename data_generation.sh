#!/bin/bash

#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=16g
#SBATCH -J "Im Creation"
#SBATCH -A rbe549
#SBATCH -p academic
#SBATCH -t 23:59:59
#SBATCH --error=SLURM_OUTPUT/data_gen%A_4.err
#SBATCH --output=SLURM_OUTPUT/data_gen_%A_4.out
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

module load py-pip/24.0 ffmpeg libxxf86vm/1.1.5/o24nj2h

source ./viovenv/bin/activate
pip install -r requirements.txt

export LD_LIBRARY_PATH=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-13.2.0/libxxf86vm-1.1.5-o24nj2he2v2kxnmovhk57d6ix2stwpkq/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/cm/shared/spack/opt/spack/linux-ubuntu20.04-x86_64/gcc-12.1.0/mesa-23.2.1-rwofodz7vknmfcno2bagsh5bzradaqt2/lib:$LD_LIBRARY_PATH

paths=('Straight_Line' 'Circle' 'Sinusoid' 'Figure_Eight' 'Hyperbolic_Paraboloid')

#python -u Code/Phase2/data_generation.py --Path ${paths[$SLURM_ARRAY_TASK_ID]}
python -u Code/Phase2/data_generation.py --Path 'Hyperbolic_Paraboloid'
