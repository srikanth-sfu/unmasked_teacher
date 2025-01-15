import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from datasets.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_dataset_colab
from engines.engine_for_collabtraining import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
from utils import LabelSmoothingCrossEntropyNoReduction
from models import *
import utils
from tubelets import build_transform
from moco import MoCo

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--iterations', default=-1, type=int)
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')
    parser.add_argument('--use_learnable_pos_emb', action='store_true')
    parser.set_defaults(use_learnable_pos_emb=False)

    parser.add_argument('--fc_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')
    parser.add_argument('--video_ext', default=".mp4", help='video ext')
    

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--layer_decay', type=float, default=0.75)

    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_iterations', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--num_sample', type=int, default=2,
                        help='Repeated_aug (default: 2)')
    parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--finetune-tag', default='', help='finetune from checkpoint')
    parser.add_argument('--delete_head', action='store_true', help='whether delete head')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument('--checkpoint_num', default=0, type=int,
                        help='number of layers for using checkpoint')
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')


    # CLIP decpder parameters
    parser.add_argument('--clip_teacher', default='clip_b16', type=str,
                        help='Name of CLIP teacher')
    parser.add_argument('--clip_input_resolution', default=224, type=int,
                        help='input resolution of CLIP decoder')
    parser.add_argument('--clip_loss_ratio', default=1., type=float,
                        help='ratio for CLIP loss, pixel_loss + RATIO * clip_loss')
    parser.add_argument('--clip_loss_type', default='l2', type=str,
                        help='type of CLIP loss')
    parser.add_argument('--clip_decoder_type', default='SA_Decoder', type=str,
                        help='type of CLIP decoder')
    parser.add_argument('--clip_decoder_embed_dim', default=512, type=int,
                        help='embedding dimension of CLIP decoder')
    parser.add_argument('--clip_output_dim', default=768, type=int,
                        help='output dimension of CLIP decoder')
    parser.add_argument('--clip_norm_type', default='l2', type=str,
                        help='type of feature normalization')
    parser.add_argument('--clip_return_attn', default=True, type=bool,
                        help='whether return CLIP attention')
    parser.add_argument('--clip_return_layer', default=1, type=int,
                        help='number of CLIP return layers')
    parser.add_argument('--clip_return_interval', default=1, type=float,
                        help='interval of CLIP teacher return layers')
    parser.add_argument('--clip_student_return_interval', default=1, type=float,
                        help='interval of CLIP student return layers')
    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube', 'attention'],
                        type=str, help='masked strategy of video tokens/patches')
    parser.add_argument('--mask_ratio', default=0.8, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    # Dataset parameters
    parser.add_argument('--prefix', default='', type=str, help='prefix for data')
    parser.add_argument('--split', default=' ', type=str, help='split for metadata')
    parser.add_argument('--data_path', default='you_data_path', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=400, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--use_decord', default=True,
                        help='whether use decord to load video, otherwise load image')
    parser.add_argument('--data_set', default='Kinetics', choices=[
        'Kinetics', 'Kinetics_sparse', 
        'SSV2', 'UCF101', 'HMDB51', 'image_folder',
        'mitv1_sparse', 'ucf_hmdb'
        ], type=str, help='dataset')
    parser.add_argument('--data_set_target', default='Kinetics', choices=[
        'Kinetics', 'Kinetics_sparse', 
        'mitv1_sparse', 'ucf_hmdb', 'hmdb_ucf'
        ], type=str, help='dataset')
    parser.add_argument("--tubelet-config", default="", type=str)

    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--test_best', action='store_true',
                        help='Whether test the best model')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    
    utils.init_distributed_mode_new(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    tubelet_params = [
                dict(
                    type='GroupToTensor',
                    switch_rgb_channels=False,
                    div255=True,
                    mean=None,
                    std=None
                )
    ]
    if not args.tubelet_config:
        print("Using default tubelet setting")
        tubelet_params1=[
                dict(
                    type='Tubelets',
                    region_sampler=dict(
                        scales=[32, 48, 56, 64, 96, 128],
                        ratios=[0.5, 0.67, 0.75, 1.0, 1.33, 1.50, 2.0],
                        scale_jitter=0.18,
                        num_rois=2,
                    ),
                    key_frame_probs=[0.5, 0.3, 0.2],
                    loc_velocity=5,
                    rot_velocity=6,
                    shear_velocity=0.066,
                    size_velocity=0.0001,
                    label_prob=1.0,
                    motion_type='gaussian',
                    patch_transformation='rotation',
                )
            ]
    else:
        with(open(args.tubelet_config, 'r')) as tconfig:
            tubelet_params1 = [json.load(tconfig)]
        if args.model_root.endswith("/"):
            args.model_root = args.model_root[:-1]
        args.model_root += ("_" + os.path.basename(args.tubelet_config.split(".")[0])) + '/'
    
    tubelet_params = tubelet_params1 + tubelet_params
    print(json.dumps(tubelet_params, indent=4))
    tubelet_transform = build_transform(tubelet_params)


    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    teacher_model = eval(args.clip_teacher)(
        clip_norm_type=args.clip_norm_type,
        input_resolution=args.clip_input_resolution,
        return_attn=args.clip_return_attn,
        clip_return_layer=args.clip_return_layer,
        clip_return_interval=args.clip_return_interval
    )

    dataset_train, args.nb_classes = build_dataset_colab(is_train=True, test_mode=False, args=args)
    
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val_src, _ = build_dataset_colab(is_train=False, test_mode=False, target=False, args=args)
        dataset_val_tgt, _ = build_dataset_colab(is_train=False, test_mode=False, target=True, args=args)
    #dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    dataset_test = None

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    
    print("Sampler_train = %s" % str(sampler_train))
    
    if args.dist_eval:
        if len(dataset_val_src) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val_src = torch.utils.data.DistributedSampler(
            dataset_val_src, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        sampler_val_tgt = torch.utils.data.DistributedSampler(
            dataset_val_tgt, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # sampler_test = torch.utils.data.DistributedSampler(
        #     dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val_src = torch.utils.data.SequentialSampler(dataset_val_src)
        sampler_val_tgt = torch.utils.data.SequentialSampler(dataset_val_tgt)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True
    )

    if dataset_val_src is not None:
        data_loader_val_src = torch.utils.data.DataLoader(
            dataset_val_src, sampler=sampler_val_src,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
        data_loader_val_tgt = torch.utils.data.DataLoader(
            dataset_val_tgt, sampler=sampler_val_tgt,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(args.num_frames * args.num_segments,args.init_scale, args.use_checkpoint, args.checkpoint_num)
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )
    moco_model = create_model(
        args.model,
        pretrained=False,
        num_classes=0,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        use_learnable_pos_emb=args.use_learnable_pos_emb,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )
    
    moco = MoCo(moco_model, args.clip_output_dim)	
    model.add_module("moco", moco)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size


    model.to(device)
    teacher_model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if args.iterations > 0:
        args.epochs = args.iterations // num_training_steps_per_epoch
        args.epochs += 1
        args.warmup_epochs = args.warmup_iterations // num_training_steps_per_epoch
        args.warmup_epochs += 1
    args.lr = args.lr * total_batch_size * args.num_sample / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_sample / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_sample / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Repeated sample = %d" % args.num_sample)
    print("Update frequent = %d" % args.update_freq)
    print("Number of source training examples = %d" % len(dataset_train))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model_engine, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )
        utils.load_model_colab(model_engine=model_engine, model=model, output_dir=args.finetune, model_name=args.finetune_tag)
        model = model_engine

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            print("Loading model outside deepspeed")
            utils.load_model_colab(model_engine=None, model=model, output_dir=args.finetune, model_name=None, deepspeed=False)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
            teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=False)


        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, start_warmup_value=args.warmup_lr, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
        criterion_target = SoftTargetCrossEntropy(reduction="none")
        
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        criterion_target = LabelSmoothingCrossEntropyNoReduction(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        criterion_target = torch.nn.CrossEntropyLoss(reduction="none")

    print("criterion = %s" % str(criterion))
    args.max_accuracy_src, args.max_accuracy_tgt = 0, 0
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = final_test(data_loader_test, model, device, preds_file)
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
            print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
            log_stats = {'Final top-1': final_top1,
                        'Final Top-5': final_top5}
            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)
        

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy_src, max_accuracy_tgt = args.max_accuracy_src, args.max_accuracy_tgt
    print("Max tgt accuracy", max_accuracy_tgt)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            teacher_model=teacher_model, clip_input_resolution=args.clip_input_resolution,
            clip_loss_ratio=args.clip_loss_ratio, mask_ratio=args.mask_ratio,
            clip_label_embedding=args.clip_label_embedding, criterion_target=criterion_target,
            len_iterable=data_loader_train, tubelet_params=tubelet_transform
        )
        if data_loader_val_src is not None:
            test_stats_src = validation_one_epoch(data_loader_val_src, model, device)
            test_stats_tgt = validation_one_epoch(data_loader_val_tgt, model, device)
            timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"[{timestep}] Accuracy of the network on the {len(dataset_val_src)} (source) val videos: {test_stats_src['acc1']:.1f}%")
            print(f"[{timestep}] Accuracy of the network on the {len(dataset_val_tgt)} (target) val videos: {test_stats_tgt['acc1']:.1f}%")
            
            if max_accuracy_src < test_stats_src["acc1"]:
                max_accuracy_src = test_stats_src["acc1"]
            if max_accuracy_tgt < test_stats_tgt["acc1"]:
                max_accuracy_tgt = test_stats_tgt["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_latest_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, model_name='best', model_ema=model_ema,
                        max_accuracy_src=max_accuracy_src, max_accuracy_tgt=max_accuracy_tgt)
            print(f'Max accuracy -- src val: {max_accuracy_src:.2f}%')
            print(f'Max accuracy -- tgt val: {max_accuracy_tgt:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1_src=test_stats_src['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5_src=test_stats_src['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss_src=test_stats_src['loss'], head="perf", step=epoch)
                log_writer.update(val_acc1_tgt=test_stats_tgt['acc1'], head="perf", step=epoch)
                log_writer.update(val_acc5_tgt=test_stats_tgt['acc5'], head="perf", step=epoch)
                log_writer.update(val_loss_tgt=test_stats_tgt['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}_src': v for k, v in test_stats_src.items()},
                         **{f'val_{k}_tgt': v for k, v in test_stats_tgt.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema, 
                    max_accuracy_src=max_accuracy_src, max_accuracy_tgt=max_accuracy_tgt)
            utils.save_latest_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, model_name='latest', model_ema=model_ema, 
                max_accuracy_src=max_accuracy_src, max_accuracy_tgt=max_accuracy_tgt)
        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    if args.test_best:
        utils.auto_load_model(
            args=args, model=model, model_without_ddp=model_without_ddp,
            optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)
    test_stats = final_test(data_loader_val_tgt, model, device, preds_file)
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        log_stats = {'Final top-1': final_top1,
                    'Final Top-5': final_top5}
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
