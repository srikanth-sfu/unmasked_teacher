import os
import clip
model, preprocess = clip.load("ViT-B/16", device="cuda")
import decord
import numpy as np
import pandas as pd
from PIL import Image
import torch
from models import *


clip_teacher = "clip_b16"
teacher_model = eval(clip_teacher)(
        clip_norm_type="l2",
        input_resolution=224,
        return_attn=True,
        clip_return_layer=1,
        clip_return_interval=1
    )

def loadvideo_decord(sample, prefix=None, video_ext='.avi', sample_rate_scale=1, chunk_nb=0):
    """Load video content using Decord"""
    if not sample.endswith(video_ext):
        sample += video_ext
    fname = sample
    fname = os.path.join(prefix, fname)

    vr = decord.VideoReader(fname, width=224, height=224,
                    num_threads=1, ctx=decord.cpu(0))

    # handle temporal segments
    clip_len, frame_sample_rate, num_segment, mode = 16, 4, 1, "validation" 
    converted_len = int(clip_len * frame_sample_rate)
    seg_len = len(vr) // num_segment


    all_index = []
    
    for i in range(num_segment):
        if seg_len <= converted_len:
            index = np.linspace(0, seg_len, num=seg_len // frame_sample_rate)
            index = np.concatenate((index, np.ones(clip_len - seg_len // frame_sample_rate) * seg_len))
            index = np.clip(index, 0, seg_len - 1).astype(np.int64)
        else:
            if mode == 'validation':
                end_idx = (seg_len - converted_len) // 2
            else:
                end_idx = np.random.randint(converted_len, seg_len)
            str_idx = end_idx - converted_len
            index = np.linspace(str_idx, end_idx, num=clip_len)
            index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
        index = index + i*seg_len
        all_index.extend(list(index))

    all_index = all_index[::int(sample_rate_scale)]
    vr.seek(0)
    buffer = vr.get_batch(all_index).asnumpy()
    return buffer

def classify(vid, label_texts):
    
    vid = [preprocess(Image.fromarray(vid[x])).to("cuda") for x in range(vid.shape[0])]
    text = clip.tokenize(label_texts).to("cuda")
    with torch.no_grad():
        text_features = model.encode_text(text)
        frame_probs = []
        frame_probs1 = []
        #for image in vid:
            #image_features = model.encode_image(image.unsqueeze(0))
        if(True):
            image_features, _ = clip_teacher(vid)
            print(image_features.shape)
            os._exit(1)
            text_features = text_features/text_features.norm(dim=1, keepdim=True)
            image_features = image_features/image_features.norm(dim=1, keepdim=True)
            
            logits_per_image = 100. * image_features @ text_features.t()
            logits_per_image1, _ = model(image.unsqueeze(0), text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs1 = logits_per_image1.softmax(dim=-1).cpu().numpy()
            frame_probs.append(probs.squeeze(0))
            frame_probs1.append(probs1.squeeze(0))
    frame_probs = np.stack(frame_probs)
    frame_probs = frame_probs.mean(axis=0)
        
    frame_probs1 = np.stack(frame_probs1).mean(axis=0)

    pred, pred_conf = frame_probs.argmax(), frame_probs.max()
    pred1, pred1_conf = frame_probs1.argmax(), frame_probs1.max()
    return pred, pred_conf, pred1, pred1_conf

if __name__ == "__main__":
    import random
    import os

    val_filelist_name = "video_splits/hmdb51_val_hmdb_ucf.csv"
    val_files = pd.read_csv(val_filelist_name)
    val_files, val_labels = list(val_files[val_files.columns[0]]), list(val_files[val_files.columns[1]])
    label_texts = open("video_splits/ucf_hmdb_classnames.txt").read().split("\n")
    files_to_sample = [random.randint(0, len(val_files)) for _ in range(10)]
    prefix = os.path.join(os.getenv("SLURM_TMPDIR"), "data/ucf_hmdb/")
    for file_id in files_to_sample:
        fn, label = val_files[file_id], val_labels[file_id]
        vid = loadvideo_decord(fn, prefix=prefix)
        pred, pred_conf, pred1, pred1_conf = classify(vid, label_texts)
        print(label, pred, pred_conf, pred1, pred1_conf)
