import os
import time
import numpy as np
import math
import sys
from typing import Iterable, Optional
import torch
from datasets.mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
import torch.nn as nn
import torch.utils.checkpoint as checkpoint


def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale

def combine_labels(label1, conf1, label2, conf2, threshold):
    """
    Combines two labels based on their confidence scores and specified logic.

    Args:
        label_1 (str): First label.
        conf_1 (float): Confidence score for label_1.
        label_2 (str): Second label.
        conf_2 (float): Confidence score for label_2.
        threshold (float): Confidence threshold.

    Returns:
        str: Combined label or -1 if no conditions are met.
    """
    assert label1.shape == label2.shape == conf1.shape == conf2.shape, "Inputs must have the same shape"
    
    batch_size = label1.shape[0]
    combined_labels = torch.full((batch_size,), -1, dtype=label1.dtype).to(label1.device)  # Initialize with -1
    
    # Condition 1: Same label
    same_label_mask = (label1 == label2)
    combined_labels[same_label_mask] = label1[same_label_mask]

    # Condition 2: Label 1 confidence > threshold and Label 2 confidence <= threshold
    label1_high_mask = (conf1 > threshold) & (conf2 <= threshold)
    combined_labels[label1_high_mask] = label1[label1_high_mask]

    # Condition 3: Label 2 confidence > threshold and Label 1 confidence <= threshold
    label2_high_mask = (conf2 > threshold) & (conf1 <= threshold)
    combined_labels[label2_high_mask] = label2[label2_high_mask]

    return combined_labels, (combined_labels != -1).type(label1.dtype), conf2

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    teacher_model=None, clip_input_resolution=224, criterion_target=None,
                    clip_loss_ratio=0.5, mask_ratio=0., clip_label_embedding=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _, _, ds_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples_tgt = samples[ds_id==1]
        samples, targets = samples[ds_id==0], targets[ds_id==0]
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        samples_tgt = samples_tgt.to(device, non_blocking=True)
        clip_label_embedding = torch.from_numpy(clip_label_embedding).to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion)


        with torch.no_grad():
            # calculate the predicted CLIP features
            B, C, T, H, W = samples_tgt.shape
            clip_input_resolution = 224
            if H != clip_input_resolution:
                clip_videos = torch.nn.functional.interpolate(
                    samples_tgt.view(B, C*T, H, W), 
                    size=(clip_input_resolution, clip_input_resolution), 
                    mode='bicubic', align_corners=False
                )
                clip_videos = clip_videos.view(B, C, T, clip_input_resolution, clip_input_resolution)
            else:
                clip_videos = samples_tgt
            
            with torch.cuda.amp.autocast():
                norm_clip, attn = teacher_model(clip_videos)
                norm_clip = norm_clip.reshape(B,-1,norm_clip.shape[-1]).mean(dim=1)
                clip_output = (norm_clip @ clip_label_embedding.T)#.reshape(B,-1).mean(dim=-1).squeeze(0)
                clip_label_conf = nn.functional.softmax(clip_output, dim=-1)
                clip_label_conf, clip_labels = clip_label_conf.max(-1)
                src_output = model(samples_tgt)
                src_encoder_labels_conf = nn.functional.softmax(src_output,dim=-1)
                src_encoder_labels_conf, src_encoder_labels = src_encoder_labels_conf.max(-1)
                target_labels, target_mask, target_conf = combine_labels(clip_labels, clip_label_conf, src_encoder_labels, src_encoder_labels_conf, threshold=0.1)
        
            
            BT, N = attn.shape
            N_vis = N - int(N * mask_ratio)
            importance = torch.multinomial(attn, N)
            bool_masked_pos = torch.ones((BT, N))
            pos1 = torch.arange(BT).view(-1, 1).repeat(1, N_vis)
            pos2 = importance[:, :N_vis]
            bool_masked_pos[pos1, pos2] = 0
            bool_masked_pos = bool_masked_pos.view(B, -1).to(torch.bool)

        with torch.cuda.amp.autocast():
            x = model.patch_embed(clip_videos)
            B, _, _ = x.size()

            if model.pos_embed is not None:
                x = x + model.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            B, _, C = x.shape
            x = x[~bool_masked_pos].reshape(B, -1, C) # ~mask means visible
        
            x = model.pos_drop(x)

            for idx, blk in enumerate(model.blocks):
                if model.use_checkpoint and idx < model.checkpoint_num:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

            x = model.norm(x)
            if model.fc_norm is not None:
                x = model.fc_norm(x.mean(1))
            else:
                x = x[:, 0]
            
            outputs_clip = model.head(model.fc_dropout(x))


            loss_target = criterion_target(outputs_clip, target_labels)
            loss_target = (loss_target * target_mask * target_conf).mean()

        loss += loss_target
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
            class_acc_target = ((outputs_clip.max(-1)[-1] == target_labels)*target_mask).float().mean()
            
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_target=loss_target.item())
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(class_acc_target=class_acc_target)
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
            log_writer.update(loss_target=loss_target.item(), head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(class_acc_target=class_acc_target, head="loss")

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, fp32=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
