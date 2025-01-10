import os
from decord import VideoReader, cpu
import numpy as np
import pandas as pd

def loadvideo_decord(sample, prefix=None, video_ext='.avi', sample_rate_scale=1, chunk_nb=0):
    """Load video content using Decord"""
    if not sample.endswith(video_ext):
        sample += video_ext
    fname = sample
    fname = os.path.join(prefix, fname)

    try:
        vr = VideoReader(fname, width=224, height=224,
                        num_threads=1, ctx=cpu(0))

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
    except:
        print("Colab file, video cannot be loaded by decord: ", fname)
        os._exit(1)
        return []

if __name__ == "__main__":
    import random
    import os

    val_filelist_name = "video_splits/hmdb51_val_hmdb_ucf.csv"
    val_files = pd.read_csv(val_filelist_name)
    val_files, val_labels = list(val_files[val_files.columns[0]]), list(val_files[val_files.columns[1]])

    files_to_sample = [random.randint(len(val_files)) for _ in range(10)]
    val_files, val_labels = val_files[files_to_sample], val_labels[files_to_sample]
    prefix = os.path.join(os.getenv("SLURM_TMPDIR"), "/data/ucf_hmdb/")
    for filename in val_files:
        vid = loadvideo_decord(filename, prefix=prefix)