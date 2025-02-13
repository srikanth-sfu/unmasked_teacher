import math
import time
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
import copy
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets.video_transforms import Compose, Normalize 
import clip
from PIL import Image

def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
        device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
        log_writer=None, lr_scheduler=None, start_steps=None,
        lr_schedule_values=None, wd_schedule_values=None, 
        teacher_model=None, clip_input_resolution=224,
        clip_loss_type='l2', clip_loss_ratio=0.5,
        mask_type='tube', mask_ratio=0., moco=None, tubelet_params=None
    ):
    model.train()
    moco.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 8

    if clip_loss_type == 'mse':
        loss_func_clip = nn.MSELoss()
    elif clip_loss_type == 'smooth_l1':
        loss_func_clip = nn.SmoothL1Loss()
    acc_dbg, total_dbg = 0, 0 
    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos, videos_raw, targets = batch
        
        text_embed = torch.from_numpy(np.load("video_splits/dailyda_classnames.npy"))
        model_dbg, preprocess = clip.load("ViT-B/32", "cpu")
        model_dbg.to(device)
        videos_clip = videos_raw[:,0,:,0]
        videos_clip = torch.permute(videos_clip, (0,2,3,1)).cpu().numpy().astype('uint8')
        videos_clip = [preprocess(Image.fromarray(videos_clip[i])) for i in range(videos_clip.shape[0])]
        videos_clip = np.stack(videos_clip)
        videos_clip = torch.from_numpy(videos_clip).to(device)
        out_dbg = model_dbg.encode_image(videos_clip).cpu()
        out_dbg /= out_dbg.norm(dim=-1, keepdim=True)
        preds_dbg = (100.0 * out_dbg @ text_embed.type(torch.float32).T).softmax(dim=-1)
        print(out_dbg.shape, preds_dbg.shape)
        _, preds_dbg = preds_dbg[0].topk(1)
        cur = (preds_dbg.cpu().numpy() == targets[:,0].numpy()).sum()
        print(cur, preds_dbg.cpu().numpy().shape)
        metric_logger.update(lr=10)
        metric_logger.update(min_lr=10)
        acc_dbg, total_dbg = acc_dbg+cur.item(), total_dbg+preds_dbg.shape[0] 
        print("ACC", 100*acc_dbg/total_dbg)
        continue
        num_rows = 7
        indices = torch.randint(0, videos_raw.size(0), (num_rows,))
        videos_raw = videos_raw[indices]
        feat_src_np, feat_tgt_np = torch.split(videos_raw, split_size_or_sections=1, dim=1)
        feat_src_np, feat_tgt_np = feat_src_np.squeeze(1).numpy(), feat_tgt_np.squeeze(1).numpy()

        np.random.shuffle(feat_tgt_np)
        src_tubelet, tgt_tubelet = utils.transform_tubelet(feat_src_np, feat_tgt_np, tubelet_params)
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        src_tubelet.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        tgt_tubelet.sub_(mean[None, :, None, None, None]).div_(std[None, :, None, None, None])
        videos = videos.to(device, non_blocking=True)
        src_tubelet = src_tubelet.to(device, non_blocking=True)
        tgt_tubelet = tgt_tubelet.to(device, non_blocking=True)

        if mask_type in ['attention']:
            bool_masked_pos = None
        else:
            bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        with torch.no_grad():
            # calculate the predicted CLIP features
            B, C, T, H, W = videos.shape
            if H != clip_input_resolution:
                clip_videos = torch.nn.functional.interpolate(
                    videos.view(B, C*T, H, W), 
                    size=(clip_input_resolution, clip_input_resolution), 
                    mode='bicubic', align_corners=False
                )
                clip_videos = clip_videos.view(B, C, T, clip_input_resolution, clip_input_resolution)
            else:
                clip_videos = videos
            
            with torch.cuda.amp.autocast():
                if bool_masked_pos is None:
                    norm_clip, attn = teacher_model(clip_videos)
                else:
                    norm_clip = teacher_model(clip_videos)

            BT, N = attn.shape
            N_vis = N - int(N * mask_ratio)
            if mask_type == 'attention':
                importance = torch.multinomial(attn, N)
                bool_masked_pos = torch.ones((BT, N))
                pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
                pos2 = importance[:, :N_vis]
                bool_masked_pos[pos1, pos2] = 0
                bool_masked_pos = bool_masked_pos.view(B, -1).to(torch.bool)
                    
            C_CLIP = norm_clip.shape[-1]
            if len(norm_clip.shape) == 4:
                K = norm_clip.shape[0]
                clip_bool_masked_pos = bool_masked_pos.unsqueeze(0).repeat(K, 1, 1)
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(K, B, -1, C_CLIP)
            else:
                clip_bool_masked_pos = bool_masked_pos
                targets_clip_vis = norm_clip[~clip_bool_masked_pos].reshape(B, -1, C_CLIP)
            targets_clip = targets_clip_vis

        with torch.cuda.amp.autocast():
            unmasked = torch.zeros((src_tubelet.shape[0], bool_masked_pos.shape[-1])).type(torch.bool).to(device)
            src_tubelet = model(src_tubelet, unmasked)
            moco_loss = moco(model.module, src_tubelet, tgt_tubelet, unmasked)["nce_loss"].mean()

        loss = (0.001*moco_loss)
        loss_value = loss.item()
        loss_pixel = torch.tensor(0.)
        loss_clip = torch.tensor(0.)


        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=list(model.parameters())+list(moco.parameters()), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(moco_loss=moco_loss.item())
        metric_logger.update(loss_pixel=loss_pixel.item())
        metric_logger.update(loss_clip=loss_clip.item())
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_pixel=loss_pixel.item(), head="loss_pixel")
            log_writer.update(loss_clip=loss_clip, head="loss_clip")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestep}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
