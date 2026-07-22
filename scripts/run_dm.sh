#!/bin/bash
#SBATCH --qos=normal                # priority (high,normal)
#SBATCH --partition=gpu    # gpu, andrena
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2            # number of GPUs per node
#SBATCH --ntasks-per-node=1
#SBATCH --time=18:0:0              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x.%j # output file name
#SBATCH --error=%x.%j  # error file name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=z.gao@qmul.ac.uk

# ~/miniforge3/bin/conda init bash
# conda create -n torch-2 python=3.10

#SCRIPT_DIR=$(cd "$(dirname "$0")";pwd)
#echo "Script directory ${SCRIPT_DIR}"
NOW=$(date +"%Y%m%d_%H%M%S")

source ~/.bashrc

export HF_HUB_OFFLINE=1

SERVER="huawei"

if [[ $SERVER == "qmul" ]]; then
  export HF_HOME="/gpfs/scratch/eey123/huggingface"

  #module load cmake/3.27.9-gcc-12.2.0
  #module load cuda/11.8.0-gcc-12.2.0
  module load cudnn/8.9.7.29-11-cuda-11.8.0-gcc-12.2.0
  module load miniforge

  #TRAIN_DATA_ROOT="/gpfs/scratch/eey123/makeup_pair-face/makeup_pair_ffhq_kontext-face" # 45590/31590
  #TRAIN_DATA_ROOT="/gpfs/scratch/eey123/makeup_pair-face/makeup_pair_qwen_flux2-face" # 51880
  #TRAIN_DATA_ROOT="/gpfs/scratch/eey123/makeup_pair-face/makeup_pair_qwen_kontext-face" # 17062/9056
  TRAIN_DATA_ROOT="/gpfs/scratch/eey123/makeup_pair-face"
elif [[ $SERVER == "huawei" ]]; then
  module load cuda/12.8

  TRAIN_DATA_ROOT="/mnt/data-alpha-sg-01/team-camera/home/z84401400/data/makeup_pair-face"
else
  echo "Unknown SERVER: $SERVER" >&2
  exit 1
fi

#read -p "Avaliable CUDA_VISIBLE_DEVICES [0,1,2,3...]: " DEVICES
#export CUDA_VISIBLE_DEVICES=${DEVICES}

conda activate torch-2

DM_CKPT="stabilityai/stable-diffusion-2-1-base"
STYLE_CLIP_CKPT="./output/vit_style_clip/checkpoints/epoch_50.pt"
PLACEHOLDER="<part>"
#PROMPT="a person with <part> makeup"
#PROMPT="a <part> person"
PROMPT="a person with makeup"
#PROMPT="a person"
GEO_MODE="3d"

CLIP_LORA=1
#CLIP_HIDDEN="3,6,12,24"
CLIP_HIDDEN="6,12,24"

ATTN_SIZE="32,64"
NUM_PARTS=4
SKIP_BG=0
USE_IPA=1
USE_TEXT_INV=0
SD_LORA=1
SD_LORA_RANK=16
SD_LORA_ALPHA=16
USE_EMA=0
STAGE1_OUT_DIR="./output/dm-stage1"
OUT_DIR="./output/dm"

BENCHMARK_ROOT="/mnt/data-alpha-sg-01/team-camera/home/z84401400/data/makeup/benchmark"
VAL_DATA_ROOT_LIST=(
  "${BENCHMARK_ROOT}/MWild-MT-pair"
  "${BENCHMARK_ROOT}/MWild-MWild-pair"
  "${BENCHMARK_ROOT}/MWild-CPM-pair"
  "${BENCHMARK_ROOT}/ffhq-MT-pair"
  "${BENCHMARK_ROOT}/ffhq-MWild-pair"
  "${BENCHMARK_ROOT}/ffhq-CPM-pair"
)
VAL_DATA_ROOT="${VAL_DATA_ROOT_LIST[0]}"
VAL_ANNO_PATH="${VAL_DATA_ROOT}/pair.txt"
DET_FACE=0


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4) +  ${SLURM_ARRAY_TASK_ID:-0})
echo "MASTER_PORT=$MASTER_PORT"
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


# stage1 pretrain id controlnet -> STAGE1_OUT_DIR
##accelerate launch --main_process_port=0 --mixed_precision=fp16 --multi_gpu \
#srun -u --cpu_bind=v --accel-bind=gn accelerate launch --main_process_port=${MASTER_PORT} --mixed_precision=fp16 --multi_gpu \
#  ./train_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --allow_tf32 \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --dataset_name="makeup" --train_data_dir=${TRAIN_DATA_ROOT} \
#  --resolution=512 --dataloader_num_workers=8 \
#  --placeholder_token=${PLACEHOLDER} --vector_shuffle \
#  --use_templates --swap_pair_rate=0. --drop_p_text=0.05 --drop_p_style=0.05 --drop_p_all=0.05 \
#  --attn_size=${ATTN_SIZE} --num_parts=${NUM_PARTS} --skip_background=${SKIP_BG} \
#  --use_ipa=0 --use_text_inv=0 \
#  --use_lora=0 --rank=${SD_LORA_RANK} --lora_alpha=${SD_LORA_ALPHA} \
#  --use_ema=0 \
#  --weight_mask=0.0 --weight_attn=0.0 \
#  --geo_mode=${GEO_MODE} \
#  --train_batch_size=8 --gradient_accumulation_steps=1 \
#  --max_train_steps=50000 --learning_rate=1e-5 --lr_adapter=1e-5 --adam_weight_decay=0.01 --lr_scheduler="constant" --lr_warmup_steps=0 \
#  --checkpointing_steps=10000 --checkpoints_total_limit=1 \
#  --val_data_root=${VAL_DATA_ROOT} --val_anno_path=${VAL_ANNO_PATH} --validation_prompt="${PROMPT}" --num_validation_images=1 \
#  --output_dir=${STAGE1_OUT_DIR} --log_frequency=100 --report_to="tensorboard" \
#  2>&1 | tee ./dm-train-stage1-${NOW}.txt


# stage2 train makeup -> OUT_DIR
##accelerate launch --main_process_port=0 --mixed_precision=fp16 --multi_gpu \
srun -u --cpu_bind=v --accel-bind=gn accelerate launch --main_process_port=${MASTER_PORT} --mixed_precision=fp16 --multi_gpu \
  ./train_dm.py \
  --pretrained_model_name_or_path=${DM_CKPT} \
  --stage1_pretrain_dir=${STAGE1_OUT_DIR} \
  --allow_tf32 \
  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
  --dataset_name="makeup" --train_data_dir=${TRAIN_DATA_ROOT} \
  --resolution=512 --dataloader_num_workers=8 \
  --placeholder_token=${PLACEHOLDER} --vector_shuffle \
  --swap_pair_rate=0. --drop_p_text=0.05 --drop_p_style=0.05 --drop_p_all=0.05 \
  --attn_size=${ATTN_SIZE} --num_parts=${NUM_PARTS} --skip_background=${SKIP_BG} \
  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} \
  --use_lora=${SD_LORA} --rank=${SD_LORA_RANK} --lora_alpha=${SD_LORA_ALPHA} \
  --use_ema=${USE_EMA} \
  --weight_mask=0.9 --weight_attn=0.1 \
  --geo_mode=${GEO_MODE} \
  --train_batch_size=8 --gradient_accumulation_steps=1 \
  --max_train_steps=100000 --learning_rate=1e-5 --lr_adapter=1e-5 --adam_weight_decay=0.01 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=10000 --checkpoints_total_limit=5 \
  --val_data_root=${VAL_DATA_ROOT} --val_anno_path=${VAL_ANNO_PATH} --validation_prompt="${PROMPT}" --num_validation_images=1 \
  --output_dir=${OUT_DIR} --log_frequency=100 --report_to="tensorboard" \
  2>&1 | tee ./dm-train-stage2-${NOW}.txt


# tensorboard --logdir path/to/logs --port 6006 --host localhost


#for i in "${!VAL_DATA_ROOT_LIST[@]}"; do
#  VAL_DATA_ROOT="${VAL_DATA_ROOT_LIST[i]}"
#  VAL_ANNO_PATH="${VAL_DATA_ROOT}/pair.txt"
#  echo "Running: ${VAL_DATA_ROOT}"
#
#  python -u ./test_dm.py \
#    --pretrained_model_name_or_path=${DM_CKPT} \
#    --ckpt_dir=${OUT_DIR} \
#    --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#    --placeholder_token=${PLACEHOLDER} \
#    --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} \
#    --num_parts=${NUM_PARTS} --skip_background=${SKIP_BG} \
#    --use_lora=${SD_LORA} --use_ema=${USE_EMA} \
#    --geo_mode=${GEO_MODE} \
#    --data_root=${VAL_DATA_ROOT} --anno_path=${VAL_ANNO_PATH} --validation_prompt="${PROMPT}" \
#    --detect_face=${DET_FACE} \
#    --out_dir="./result" \
#    2>&1 | tee ./dm-test-${NOW}.txt
#done


# single image pair
#python -u ./test_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --ckpt_dir=${OUT_DIR} \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --placeholder_token=${PLACEHOLDER} \
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} \
#  --num_parts=${NUM_PARTS} --skip_background=${SKIP_BG} \
#  --use_lora=${SD_LORA} --use_ema=${USE_EMA} \
#  --geo_mode=${GEO_MODE} \
#  --data_id_path="${BENCHMARK_ROOT}/custom/id/1698.png" \
#  --data_makeup_path="${BENCHMARK_ROOT}/custom/makeup/2407.png" \
#  --validation_prompt="${PROMPT}" \
#  --guidance_scale=5.5 --ipa_scale=1.0 \
#  --detect_face=1 --exp_ratio=-1 --use_square=1 \
#  --vis_all=1 --vis_attn=1 \
#  --out_dir="./result"


# region face, eyes, mouth
#python -u ./test_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --ckpt_dir=${OUT_DIR} \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --placeholder_token=${PLACEHOLDER} \
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} \
#  --num_parts=${NUM_PARTS} --skip_background=${SKIP_BG} \
#  --use_lora=${SD_LORA} --use_ema=${USE_EMA} \
#  --geo_mode=${GEO_MODE} \
#  --data_id_path="${BENCHMARK_ROOT}/custom/id/stablemakeup-2.jpg" \
#  --data_makeup_path="${BENCHMARK_ROOT}/custom/makeup/vRX31.png;${BENCHMARK_ROOT}/custom/makeup/157.png;${BENCHMARK_ROOT}/custom/makeup/126.png" \
#  --validation_prompt="${PROMPT}" \
#  --guidance_scale=5.5 --ipa_scale=1.0 \
#  --detect_face=1 --exp_ratio=-1 --use_square=1 \
#  --vis_all=1 --vis_attn=1 \
#  --out_dir="./result"
