# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from mixin import TrainStepMixin
import copy
import torch.utils.checkpoint as checkpoint


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MoCo(nn.Module, TrainStepMixin):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722

    The code is mainly ported from the official repo:
    https://github.com/facebookresearch/moco

    """

    def __init__(self,
                 model,
                 in_channels: int,
                 queue_size: int = 384,
                 momentum: float = 0.999,
                 temperature: float = 0.07):
        super(MoCo, self).__init__()
        self.K = queue_size
        self.m = momentum
        self.T = temperature

        self.register_buffer("queue", torch.randn(128, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.key_encoder = model
        self.fc = nn.TransformerEncoderLayer(in_channels, 2, in_channels)
        self.key_fc = nn.TransformerEncoderLayer(in_channels, 2, in_channels)
        self.positional_encoding = nn.Parameter(torch.randn(1568, in_channels))
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def key_encoder_forward(self,clip_videos,mask):
        return self.key_encoder(clip_videos,mask)

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _momentum_update_key_encoder(self, backbone):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(backbone.parameters(),
                                    self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.fc.parameters(),
                                    self.key_fc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr


    def forward(self, model, q, k_in, mask):
        with(torch.cuda.amp.autocast()):
            NS, B, _, _ = q.shape
            q = q.reshape(-1,q.shape[-2], q.shape[-1])
            q = q + self.positional_encoding
            q = q.transpose(0,1)
            q = self.fc(q)
            q = q.transpose(0,1)
            q = q.mean(dim=1)
            q = self.fc(q)
            q = nn.functional.normalize(q, dim=1)

            # # compute key features
            with torch.no_grad():
                self._momentum_update_key_encoder(backbone=model)
                im_k, idx_unshuffle = self._batch_shuffle_ddp(k_in)
                tgt_tubelet = self.key_encoder_forward(im_k, mask)
                k = tgt_tubelet.reshape(-1,tgt_tubelet.shape[-2], tgt_tubelet.shape[-1])
                k = k + self.positional_encoding
                k = k.transpose(0,1)
                k = self.key_fc(k)
                k = k.transpose(0,1)
                k = k.mean(dim=1)
                k = nn.functional.normalize(k, dim=1).reshape(NS, B, -1)
                k = torch.transpose(k, 1, 0)
                k = k.contiguous()
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
                

            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            k = torch.transpose(k, 1, 0).reshape(B*NS, -1)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            nce_loss = nn.functional.cross_entropy(logits, labels)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

        return dict(nce_loss=nce_loss)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
