#!/bin/sh
#SBATCH --chdir=/scratch/nas/1/{username}/gitt/Accelerate-scientific-applications-using-GPGPU-computing/00_firstProgram/
#SBATCH --output=/scratch/nas/1/{username}/git/Accelerate-scientific-applications-using-GPGPU-computing/00_firstProgram/sortida-%j.out
#SBATCH --error=/scratch/nas/1/{username}/git/Accelerate-scientific-applications-using-GPGPU-computing/00_firstProgram/error-%j.out
#SBATCH --job-name="cuda"
#SBATCH -A cuda
#SBATCH -p cuda
#SBATCH --gres=gpu:1
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
./hello_cuda
