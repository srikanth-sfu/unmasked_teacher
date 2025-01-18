export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

MODEL_PATH='/home/ens/smuralidharan/checkpoints/umt/pretrain/baseline_b16_arid/checkpoint-49.pth'
JOB_NAME='baseline_b16_ktarid_f8_res224'
OUTPUT_DIR="/home/ens/smuralidharan/checkpoints/umt/src_finetune/$JOB_NAME"
PREFIX="${SLURM_TMPDIR}/data/ucf_hmdb/"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='video_splits/'

python -m torch.distributed.launch --nproc_per_node 4 run_class_finetuning.py \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --nb_classes 12 \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 7 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_iterations 4000 \
        --iterations 20000 \
        --tubelet_size 1 \
        --lr 2.5e-5 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --enable_deepspeed \
        --test_best \
        --data_set ucf_hmdb \
        --video_ext .mp4 \
        --split ',' \
        --mixup 0.0 \
        --cutmix 0.0 \
        --train_split_src kinetics600_dailyda_train.csv \
        --val_split_src kinetics600_dailyda_val.csv
