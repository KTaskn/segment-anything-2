import torch
from tqdm import tqdm
import argparse
from models import WrapperI3D
import pandas as pd
from milforvideo.video import Extractor
import cv2
import numpy as np
import itertools
from PIL import Image
from masking import compute_masking

import warnings
warnings.filterwarnings('ignore')

def compute_background(frames: np.ndarray):
    return np.median(frames, axis=0).astype(np.uint8)

tmp_num = 0
F = 16
N_BATCHES = 5
N_WORKERS = 5
N_CLUSTERS = 5
ATTENTION_SIZE = 5
N_SAMPLE_BACKGROUND = 10000


def open_frames(paths, resize=1.0):
    return np.array([cv2.resize(cv2.imread(path), dsize=None, fx=resize, fy=resize) for path in paths])

def img2tensor_forvideo(paths, idx, img_background, n_clusters, resize=1.0, blur=False, combination=False):
    frames = open_frames(paths, resize)
    for masked_frames, _ in compute_masking(frames, img_background, n_clusters, combination=combination, blur=blur):
        yield masked_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pathlist", help="Path to the dataset", type=str)
    parser.add_argument("output_path", help="Path to the output file", type=str)
    parser.add_argument("--gpu", action='store_true', help="Use GPU")
    parser.add_argument("--is_getable_images", action='store_true', help="Save Images")
    parser.add_argument("--resize", default=1.0, help="resize", type=float)
    parser.add_argument("--blur", action='store_true', help="Use blur")
    parser.add_argument("--combination", action='store_true', help="Use combination")
    
    args = parser.parse_args()
    print(f"pathlist: {args.pathlist}")
    print(f"output_path: {args.output_path}")
    print(f"gpu: {args.gpu}")
    print(f"is_getable_images: {args.is_getable_images}")
    print(f"resize: {args.resize}")
    print(f"blur: {args.blur}")
    print(f"combination: {args.combination}")
    
    # You can change the model here
    net = WrapperI3D()
    
    # Get the image path and label from a input file
    with open(args.pathlist) as f:
        grp_path_and_label = [
            row.split(" ")
            for row in f.read().split("\n")
            if row
        ]
        df = pd.DataFrame({
            "grp": [int(grp) for grp, _, _ in grp_path_and_label],
            "path": [path for _, path, _ in grp_path_and_label],
            "label": [int(label) for _, _, label in grp_path_and_label],
        })
        
    outputs = []
    for idx, df_grp in tqdm(df.groupby('grp')):
        # when compute background, sample frame If the number of frames is over than N_SAMPLE_BACKGROUND
        paths_for_background = df_grp["path"].tolist() if len(df_grp["path"].tolist()) < N_SAMPLE_BACKGROUND else df_grp["path"].sample(N_SAMPLE_BACKGROUND).tolist()
        frames = open_frames(paths_for_background, args.resize)
        img_background = compute_background(frames)
        parser = lambda paths, idx: img2tensor_forvideo(paths, idx, img_background, args.n_clusters, resize=args.resize, blur=args.blur, combination=args.combination)
        extractor = Extractor(
            df_grp["path"].tolist(), 
            df_grp["label"].tolist(), 
            net, parser,
            F=F,
            n_batches=N_BATCHES,
            n_workers=N_WORKERS,
            aggregate=lambda labels: [max(labels)],
            cuda=args.gpu)
        features = extractor.extract(is_getable_images=args.is_getable_images)         
        features.img_background = Image.fromarray(cv2.cvtColor(img_background, cv2.COLOR_BGR2RGB))
        outputs.append(features)
        
    print("faetures_size: ", outputs[0].features.size())
    torch.save(outputs, args.output_path)
