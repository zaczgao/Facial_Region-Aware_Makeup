#!/bin/bash
#SBATCH --qos=normal                # priority (highest,normal)
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # number of GPUs per node
#SBATCH --ntasks-per-node=2
#SBATCH --time=240:0:0              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x.%j # output file name
#SBATCH --error=%x.%j  # error file name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=z.gao@qmul.ac.uk


#SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)
#echo "Script directory ${SCRIPT_DIR}"
NOW=$(date +"%Y%m%d_%H%M%S")


source ~/.bashrc

SERVER="qmul"

if [[ $SERVER == "qmul" ]]; then
  export HF_HOME="/gpfs/scratch/eey123/huggingface"

  #module load cmake/3.27.9-gcc-12.2.0
  #module load cuda/11.8.0-gcc-12.2.0
  module load cudnn/8.9.7.29-11-cuda-11.8.0-gcc-12.2.0
  module load miniforge
elif [[ $SERVER == "huawei" ]]; then
  module load cuda/12.4
else
  echo "Unknown SERVER: $SERVER" >&2
  exit 1
fi

conda activate torch-2


OUT_DIR="./output"

DATA_ROOT="/gpfs/scratch/eey123/makeup_pair_ffhq_kontext"
DATA_TRAIN=${DATA_ROOT}/makeup
DATA_VAL=${DATA_ROOT}/makeup
ANNO_TRAIN=${DATA_ROOT}/makeup_pair_ffhq_kontext-train.csv
ANNO_VAL=${DATA_ROOT}/makeup_pair_ffhq_kontext-val.csv

VAL_DATA_ROOT="/gpfs/scratch/eey123/makeup_pair_qwen_kontext"
VAL_DATA_TRAIN=${VAL_DATA_ROOT}/makeup
VAL_DATA_VAL=${VAL_DATA_ROOT}/makeup
VAL_ANNO_TRAIN=${VAL_DATA_ROOT}/makeup_pair_qwen_kontext-train.csv
VAL_ANNO_VAL=${VAL_DATA_ROOT}/makeup_pair_qwen_kontext-val.csv


#MASTER_PORT=$(( ( RANDOM % 40000 ) + 20000 ))
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) +  ${SLURM_ARRAY_TASK_ID:-0})
echo "MASTER_PORT=$MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


## train encoder layer
#torchrun --nproc_per_node 2 --master_port=$MASTER_PORT -- ./train_style_clip.py \
#  --train-data=${DATA_TRAIN} --train-anno=${ANNO_TRAIN} \
#  --val-data=${DATA_VAL} --val-anno=${ANNO_VAL} \
#  --memory-data=${DATA_TRAIN} --memory-anno=${ANNO_TRAIN} \
#  --dataset-type="style" \
#  --model="vit_large" --model_text="vit_giant" --use-bn-sync --name="vit_style_clip" \
#  --lock-image --lock-image-unlocked-groups=1 --lock-image-freeze-bn-stats \
#  --warmup=1000 --batch-size=512 --epochs=50 --accum-freq=1 \
#  --lr=1e-05 --lr-head=2e-05 --wd=0.2 --lr-scheduler=cosine \
#  --lambda_ssl=1. --lambda_sup=0. --lambda_text=0.1 \
#  --precision=amp \
#  --workers=8 \
#  --report-to=tensorboard --save-frequency=25 --log-every-n-steps=20 \
#  --logs=${OUT_DIR} \
#  2>&1 | tee ./style-clip-train-${NOW}.txt
#
## test on synthetic data
#torchrun --nproc_per_node 2 --master_port=$MASTER_PORT -- ./train_style_clip.py \
#  --val-data=${VAL_DATA_VAL} --val-anno=${VAL_ANNO_VAL} \
#  --memory-data=${VAL_DATA_TRAIN} --memory-anno=${VAL_ANNO_TRAIN} \
#  --dataset-type="style" \
#  --model="vit_large" --model_text="vit_giant" --use-bn-sync --name="vit_style_clip-test" \
#  --lock-image --lock-image-unlocked-groups=1 --lock-image-freeze-bn-stats \
#  --warmup=1000 --batch-size=512 --epochs=50 --accum-freq=1 \
#  --lr=1e-05 --lr-head=2e-05 --wd=0.2 --lr-scheduler=cosine \
#  --precision=amp \
#  --logs=${OUT_DIR} \
#  --resume=${OUT_DIR}/vit_style_clip/checkpoints/epoch_50.pt \
#  2>&1 | tee ./style-clip-test-${NOW}.txt


# train lora
#torchrun --nproc_per_node 2 --master_port=$MASTER_PORT -- ./train_style_clip.py \
srun --cpu_bind=v --accel-bind=gn python -u ./train_style_clip.py \
  --train-data=${DATA_TRAIN} --train-anno=${ANNO_TRAIN} \
  --val-data=${DATA_VAL} --val-anno=${ANNO_VAL} \
  --memory-data=${DATA_TRAIN} --memory-anno=${ANNO_TRAIN} \
  --dataset-type="style" \
  --model="vit_large" --model_text="vit_giant" --use-bn-sync --name="vit_style_clip" \
  --lock-image-unlocked-groups=1 --lock-image-freeze-bn-stats \
  --warmup=1000 --batch-size=256 --epochs=50 --accum-freq=2 \
  --lr=1e-05 --lr-head=2e-05 --wd=0.2 --lr-scheduler=cosine \
  --lambda_ssl=1. --lambda_sup=0. --lambda_text=0.1 \
  --precision=amp \
  --workers=8 \
  --report-to=tensorboard --save-frequency=25 --log-every-n-steps=20 \
  --logs=${OUT_DIR} \
  2>&1 | tee ./style-clip-train-${NOW}.txt

# test on synthetic data
#torchrun --nproc_per_node 2 --master_port=$MASTER_PORT -- ./train_style_clip.py \
srun --cpu_bind=v --accel-bind=gn python -u ./train_style_clip.py \
  --val-data=${VAL_DATA_VAL} --val-anno=${VAL_ANNO_VAL} \
  --memory-data=${VAL_DATA_TRAIN} --memory-anno=${VAL_ANNO_TRAIN} \
  --dataset-type="style" \
  --model="vit_large" --model_text="vit_giant" --use-bn-sync --name="vit_style_clip-test" \
  --lock-image-unlocked-groups=1 --lock-image-freeze-bn-stats \
  --warmup=1000 --batch-size=512 --epochs=50 --accum-freq=1 \
  --lr=1e-05 --lr-head=2e-05 --wd=0.2 --lr-scheduler=cosine \
  --precision=amp \
  --logs=${OUT_DIR} \
  --resume=${OUT_DIR}/vit_style_clip/checkpoints/epoch_50.pt \
  2>&1 | tee ./style-clip-test-${NOW}.txt

# tensorboard --logdir=logs/tensorboard/ --port=7777