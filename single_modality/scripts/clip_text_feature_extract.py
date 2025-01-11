#!/usr/bin/env python
import os
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
import clip
from PIL import Image


MODEL_PATH = '/project/def-mpederso/smuralid/checkpoints/umt/clip_visual_encoder'
_MODELS = {
    # extracted from OpenAI, see extract_clip
    "ViT-B/16": os.path.join(MODEL_PATH, "vit_b16.pth"),
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14.pth"),
    "ViT-L/14_336": os.path.join(MODEL_PATH, "vit_l14_336.pth"),
}

def extract_clip_features(classnames, output_file):

    # Load the pre-trained CLIP model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    # Tokenize class labels
    text_inputs = clip.tokenize(classnames).to(device)

    # Extract features
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)

    # Normalize features
    feats = text_features / text_features.norm(dim=-1, keepdim=True)
    feats = feats.cpu().numpy()
    print(feats.shape)
    np.save(output_file, feats)

if __name__ == '__main__':
    import sys
    classnames = open(sys.argv[1]).read().split("\n")
    output_file = sys.argv[1].replace(".txt", ".npy")
    extract_clip_features(classnames, output_file)