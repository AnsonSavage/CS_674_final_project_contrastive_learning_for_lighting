#!/bin/bash

#SBATCH --time=10:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8192M   # memory per CPU core
#SBATCH -J "generate_contrastive_learning_for_lighting_training_data"   # job name
#SBATCH --mail-user=ansonsav@byu.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=dw87


export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
lscpu | grep "^CPU"

echo "(Number of CPUs)"

grep -c ^processor /proc/cpuinfo

./blender-4.3.0-linux-x64/blender ./input_scenes/Blender_3.blend --background --python ./src/training_data_generation/main.py -- --start_seed=10000 --hdri_dir=./hdris/ --output_dir=./training_data/test_1/