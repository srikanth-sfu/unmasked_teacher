export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

MODEL_PATH='/home/ens/smuralidharan/checkpoints/umt/src_finetune/baseline_b16_ucf_hmdb_f8_res224/'
JOB_NAME='tubeletumt_s3_b16_ucf_hmdb_f8_res224'
OUTPUT_DIR="/home/ens/smuralidharan/checkpoints/umt_tubelet/colabtrainonly/$JOB_NAME"
PREFIX="/storage/smuralidharan/data/ucf_hmdb/"
LOG_DIR="./logs/${JOB_NAME}"
DATA_PATH='video_splits/'

python -m torch.distributed.launch --nproc_per_node 4 run_collaborative_tuning.py \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --prefix ${PREFIX} \
        --nb_classes 12 \
        --finetune ${MODEL_PATH} \
        --finetune-tag "checkpoint-best" \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 5 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 8 \
        --num_workers 12 \
        --warmup_iterations 4000 \
        --iterations 20000 \
        --tubelet_size 1 \
        --lr 1e-5 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --test_num_segment 4 \
        --test_num_crop 3 \
        --dist_eval \
        --test_best \
        --data_set ucf_hmdb \
        --data_set_target hmdb_ucf \
        --video_ext .mp4 \
        --split ',' \
        --mixup 0.0 \
        --cutmix 0.0 \
        --train_anno_path ucf101_train_hmdb_ucf.csv \
        --train_anno_path_target hmdb51_train_hmdb_ucf.csv \
        --test_anno_path hmdb51_val_hmdb_ucf.csv 
