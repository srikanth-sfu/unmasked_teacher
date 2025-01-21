import clip
import pandas as pd
import torch
import numpy as np
import decord
import os
from PIL import Image

text = torch.from_numpy(np.load("video_splits/dailyda_classnames.npy")).to('cuda').type(torch.float16)
split = pd.read_csv("video_splits/arid_val.csv")
samples, labels = list(split.iloc[:,0]), list(split.iloc[:,1])

model, preprocess = clip.load('ViT-B/16', 'cuda')
video_prefix = '/storage/smuralidharan/data/'


def load(video_loc):
    vid = decord.VideoReader(video_loc, ctx=decord.cpu(0))
    converted_len = 32 
    seg_len = len(vid)
    all_index = []
    if seg_len <= converted_len:
        index = np.linspace(0, seg_len, num=seg_len // 4) 
        index = np.concatenate((index, np.ones(8 - seg_len // 4) * seg_len))
        index = np.clip(index, 0, seg_len - 1).astype(np.int64)
    else:
        end_idx = (seg_len - converted_len) // 2
        str_idx = end_idx - converted_len
        index = np.linspace(str_idx, end_idx, num=8)
        index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
    all_index.extend(list(index))
    vid.seek(0)
    buffer = vid.get_batch(all_index).asnumpy()
    return buffer
    

acc, total = 0, len(labels)
for sample, label in zip(samples,labels):
    video_loc = os.path.join(video_prefix, sample)
    vid = load(video_loc)
    vid = [Image.fromarray(vid[x]) for x in range(8)]
    vid = [preprocess(x).unsqueeze(0) for x in vid]
    vid = torch.cat(vid).to('cuda')
    vis = model.encode_image(vid)
    vis = vis/vis.norm(dim=-1, keepdim=True)
    similarity = (100.0 * vis @ text.T).softmax(dim=-1).mean(dim=0)
    values, indices = similarity.topk(1)
    acc += (indices.cpu().numpy().tolist()[0] == label)
    print(acc, )
print(acc,total)
