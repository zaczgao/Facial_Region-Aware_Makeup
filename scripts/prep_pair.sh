#!/bin/bash
#SBATCH --qos=normal                # priority (highest,normal)
#SBATCH --partition=andrena
#SBATCH --nodes=1
#SBATCH --gres=gpu:1            # number of GPUs per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=240:0:0              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x.%j # output file name
#SBATCH --error=%x.%j  # error file name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=z.gao@qmul.ac.uk


#SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)
#echo "Script directory ${SCRIPT_DIR}"
NOW=$(date +"%Y%m%d_%H%M%S")

source ~/.bashrc

export HF_HOME="/gpfs/scratch/eey123/huggingface"

#module load cuda/11.8.0-gcc-12.2.0
module load cudnn/8.9.7.29-11-cuda-11.8.0-gcc-12.2.0
module load miniforge
conda activate torch-2


EDIT_MODEL="kontext"  # kontext, flux2
T2I_MODEL="qwen"
OUT_DIR="/gpfs/scratch/eey123"


# read -p "Avaliable CUDA_VISIBLE_DEVICES [0,1,2,3...]: " DEVICES


# ffhq 40000x5x1 get makeup images, filter makeup images, get makeup pair (makeup_mix)
START_IDX=0
END_IDX=5000
python -u ./utils/create_makeup_pair.py \
  --edit_method=${EDIT_MODEL} --t2i_method=${T2I_MODEL} \
  --data_makeup_path="./assets/makeup_gpto3.json" \
  --data_id_path="./assets/ffhq_id.txt" \
  --data_face_dir="/gpfs/scratch/eey123/ffhq" \
  --process="edit" \
  --out_dir=${OUT_DIR}/makeup_pair_ffhq_${EDIT_MODEL} \
  --start_idx=${START_IDX} --end_idx=${END_IDX} \
  --num_id=40000 \
  --num_makeup=5 \
  --num_img_makeup=1 \
  --img_size=512 --min_h=250 --min_w=250 \
  2>&1 | tee ./makeup_pair_ffhq-${NOW}.txt

python -u ./utils/create_makeup_pair.py \
  --edit_method=${EDIT_MODEL} --t2i_method=${T2I_MODEL} \
  --data_makeup_path="./assets/makeup_gpto3.json" \
  --data_id_path="./assets/ffhq_id.txt" \
  --data_face_dir="/gpfs/scratch/eey123/ffhq" \
  --process="mix" \
  --out_dir=${OUT_DIR}/makeup_pair_ffhq_${EDIT_MODEL} \
  --start_idx=${START_IDX} --end_idx=${END_IDX} \
  --num_id=40000 \
  --num_makeup=5 \
  --num_img_makeup=1 \
  --img_size=512 --min_h=250 --min_w=250 \
  2>&1 | tee ./makeup_pair_ffhq-${NOW}.txt

# get face data (mask, 3d mesh) from id
python -u ./utils/prep_face.py \
  --data_dir=${OUT_DIR}/makeup_pair_ffhq_${EDIT_MODEL}/id \
  --out_dir=${OUT_DIR}/makeup_pair_ffhq_${EDIT_MODEL}-face \
  --exp_ratio=-1 --use_square=0 --min_h=250 --min_w=250 \
  --use_face_data=1 \
  2>&1 | tee ./makeup_pair_ffhq_id_face-${NOW}.txt

