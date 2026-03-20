#!/bin/bash
#SBATCH --qos=normal                # priority (highest,normal)
#SBATCH --partition=camera-xlong    # camera-xlong, andrena
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:2            # number of GPUs per node
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:0:0              # maximum execution time (HH:MM:SS)
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

SERVER="huawei"

export HF_HUB_OFFLINE=1

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
  module load cuda/12.4

  TRAIN_DATA_ROOT="../data/makeup_pair-face"
else
  echo "Unknown SERVER: $SERVER" >&2
  exit 1
fi

conda activate torch-2

#read -p "Avaliable CUDA_VISIBLE_DEVICES [0,1,2,3...]: " DEVICES
#CUDA_VISIBLE_DEVICES=${DEVICES}

DM_CKPT="stabilityai/stable-diffusion-2-1-base"
STYLE_CLIP_CKPT="./output/vit_style_clip/checkpoints/epoch_50.pt"
PLACEHOLDER="<part>"
#PROMPT="a person with <part> makeup"
#PROMPT="a <part> person"
PROMPT="a person with makeup"
#PROMPT="a person"

CLIP_LORA=1
CLIP_HIDDEN="3,6,12,24"

NUM_PARTS=4
USE_IPA=1
USE_TEXT_INV=0
SD_LORA=1
SD_LORA_RANK=8
SD_LORA_ALPHA=16
USE_3D=1
OUT_DIR="./output/dm"
STAGE1_OUT_DIR="./output/dm-stage1"

BENCHMARK_ROOT="../data/makeup/benchmark"
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


# pretrain id controlnet -> STAGE1_OUT_DIR
#srun --cpu_bind=v --accel-bind=gn accelerate launch --main_process_port=${MASTER_PORT} --mixed_precision=fp16 --multi_gpu \
#  ./train_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --allow_tf32 \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --dataset_name="makeup" --train_data_dir=${TRAIN_DATA_ROOT} \
#  --resolution=512 --dataloader_num_workers=8 \
#  --placeholder_token=${PLACEHOLDER} \
#  --vector_shuffle --swap_pair_rate=0. --drop_text_rate=0.05 --drop_style_rate=0.05 --drop_all_rate=0.05 \
#  --attn_size="32,64" --num_parts=${NUM_PARTS} --skip_background \
#  --use_ipa=0 --use_text_inv=0 --use_3d=${USE_3D} \
#  --use_lora=0 --rank=${SD_LORA_RANK} --lora_alpha=${SD_LORA_ALPHA} \
#  --weight_attn=0.1 \
#  --train_batch_size=32 --gradient_accumulation_steps=1 \
#  --max_train_steps=50000 --num_train_epochs=1000 --learning_rate=1e-05 --lr_adapter=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
#  --checkpointing_steps=10000 --checkpoints_total_limit=1 \
#  --val_data_root=${VAL_DATA_ROOT} --val_anno_path=${VAL_ANNO_PATH} --validation_prompt="${PROMPT}" --num_validation_images=1 \
#  --output_dir=${STAGE1_OUT_DIR} --log_frequency=100 --report_to="tensorboard" \
#  2>&1 | tee ./dm-train-stage1-${NOW}.txt


srun --cpu_bind=v --accel-bind=gn accelerate launch --main_process_port=${MASTER_PORT} --mixed_precision=fp16 --multi_gpu \
  ./train_dm.py \
  --pretrained_model_name_or_path=${DM_CKPT} \
  --stage1_pretrain_dir=${STAGE1_OUT_DIR} \
  --allow_tf32 \
  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
  --dataset_name="makeup" --train_data_dir=${TRAIN_DATA_ROOT} \
  --resolution=512 --dataloader_num_workers=8 \
  --placeholder_token=${PLACEHOLDER} \
  --vector_shuffle --swap_pair_rate=0. --drop_text_rate=0.05 --drop_style_rate=0.05 --drop_all_rate=0.05 \
  --attn_size="32,64" --num_parts=${NUM_PARTS} --skip_background \
  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
  --use_lora=${SD_LORA} --rank=${SD_LORA_RANK} --lora_alpha=${SD_LORA_ALPHA} \
  --weight_attn=0.1 \
  --train_batch_size=32 --gradient_accumulation_steps=1 \
  --max_train_steps=150000 --num_train_epochs=1000 --learning_rate=2e-05 --lr_adapter=2e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --checkpointing_steps=20000 --checkpoints_total_limit=5 \
  --val_data_root=${VAL_DATA_ROOT} --val_anno_path=${VAL_ANNO_PATH} --validation_prompt="${PROMPT}" --num_validation_images=1 \
  --output_dir=${OUT_DIR} --log_frequency=100 --report_to="tensorboard" \
  2>&1 | tee ./dm-train-${NOW}.txt


# tensorboard --logdir path/to/logs --port 6006 --host 0.0.0.0


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
#    --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
#    --num_parts=${NUM_PARTS} --use_lora=${SD_LORA} \
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
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
#  --num_parts=${NUM_PARTS} --use_lora=${SD_LORA} \
#  --data_id_path="../data/makeup/benchmark/mine/id/1698.png" \
#  --data_makeup_path="../data/makeup/benchmark/mine/makeup/2407.png" \
#  --validation_prompt="${PROMPT}" \
#  --guidance_scale=7.5 --ipa_scale=1.0 \
#  --detect_face=1 --exp_ratio=-1 --use_square=1 \
#  --vis_cat=1 --vis_attn=1 \
#  --out_dir="./result"


# "an Asian girl with makeup, short hair", "a man with makeup, wearing hat"
# text-to-image
#python -u ./test_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --ckpt_dir=${OUT_DIR} \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --placeholder_token=${PLACEHOLDER} \
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
#  --num_parts=${NUM_PARTS} --use_lora=${SD_LORA} \
#  --data_id_path="" \
#  --data_makeup_path="../data/makeup/benchmark/mine/makeup/1080.png" \
#  --validation_prompt="an Asian girl with makeup, short hair" \
#  --detect_face=1 \
#  --vis_cat=1 \
#  --out_dir="./result"


# region
#python -u ./test_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --ckpt_dir=${OUT_DIR} \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --placeholder_token=${PLACEHOLDER} \
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
#  --num_parts=${NUM_PARTS} --use_lora=${SD_LORA} \
#  --data_id_path="../data/makeup/benchmark/mine/id/00059.png" \
#  --data_makeup_path="../data/makeup/benchmark/mine/makeup/1878.png" \
#  --validation_prompt="${PROMPT}" \
#  --detect_face=1 --exp_ratio=-1 --use_square=1 \
#  --token_idx="0" \
#  --vis_cat=1 \
#  --out_dir="./result"


# region face, eyes, mouth
#python -u ./test_dm.py \
#  --pretrained_model_name_or_path=${DM_CKPT} \
#  --ckpt_dir=${OUT_DIR} \
#  --style_clip_ckpt=${STYLE_CLIP_CKPT} --use_clip_lora=${CLIP_LORA} --clip_hidden=${CLIP_HIDDEN} \
#  --placeholder_token=${PLACEHOLDER} \
#  --use_ipa=${USE_IPA} --use_text_inv=${USE_TEXT_INV} --use_3d=${USE_3D} \
#  --num_parts=${NUM_PARTS} --use_lora=${SD_LORA} \
#  --data_id_path="../data/makeup/benchmark/mine/id/stablemakeup-2.jpg" \
#  --data_makeup_path="../data/makeup/benchmark/mine/makeup/vRX31.png;../data/makeup/benchmark/mine/makeup/157.png;../data/makeup/benchmark/mine/makeup/126.png" \
#  --validation_prompt="${PROMPT}" \
#  --guidance_scale=7.5 --ipa_scale=1.0 \
#  --detect_face=1 --exp_ratio=-1 --use_square=1 \
#  --vis_cat=1 --vis_attn=1 \
#  --out_dir="./result"
