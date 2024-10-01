import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import os
import argparse

import cv2
from utils import colormap
from glob import glob

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

BLUR_SIZE = 9
BLUR_SIGMA = 0
    
def open_frames(paths, resize=1.0):
    return np.array([cv2.resize(cv2.imread(path), dsize=None, fx=resize, fy=resize) for path in paths])

def compute_background(frames: np.ndarray):
    return np.median(frames, axis=0).astype(np.uint8)

# obj_idが動的に変わるので、最大のobj_idを保持するようにする
MAX_OBJ_ID = 50
def compute_masking(
    paths: np.ndarray,
    F: int = 16,
    combination: bool = False,
    blur: bool = False,
    visualize: bool = False,
    output_path: str = "outputs",
):
    frames = open_frames(paths)
    img_background = compute_background(frames)
    
    n, w, h, c = frames.shape
    # points_per_side is made to be half of the smallest side of the image
    points_per_side = w // 2 if w > h else h // 2

    sam2 = build_sam2(
        model_cfg, checkpoint, device="cuda", apply_postprocessing=False
    )
    # make the initial mask
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2, points_per_side=points_per_side
    )
    
    
    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        for start_frame_idx in range(0, frames.shape[0], F):
            image = frames[start_frame_idx]
            masks = mask_generator.generate(image)
            # Fごとに初期化
            predictor = build_sam2_video_predictor(model_cfg, checkpoint)
            state = predictor.init_state(os.path.dirname(paths[0]))
            for idx, mask in enumerate(masks):
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=start_frame_idx,
                    obj_id=idx,
                    mask=mask["segmentation"],
                )
                
                # obj_idがMAX_OBJ_IDを超えた場合は停止
                assert idx < MAX_OBJ_ID
                    

            # propagate the prompts to get masklets throughout the video
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=start_frame_idx, max_frame_num_to_track=F):
                masks = masks.cpu().numpy()
                target_frame = frames[frame_idx]
                # coloring the each mask
                for obj_id, mask in zip(object_ids, masks):
                    w, h, c = target_frame.shape
                    mask = np.stack([mask[0]] * 3, axis=-1)
                    if blur:
                        output = np.where(mask > 0, target_frame, cv2.GaussianBlur(target_frame, (BLUR_SIZE, BLUR_SIZE), BLUR_SIGMA))
                    else:
                        output = np.where(mask > 0, target_frame, img_background)
                    Image.fromarray(output).save(
                        os.path.join(output_path, f"{obj_id:02}_{frame_idx:05}.png")
                    )
                    
                not_exists_obj_id = list(set(range(MAX_OBJ_ID)) - set(object_ids))
                for obj_id in not_exists_obj_id:
                    Image.fromarray(img_background).save(
                        os.path.join(output_path, f"{obj_id:02}_{frame_idx:05}.png")
                    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--blur", action="store_true")
    args = parser.parse_args()
    paths = sorted(glob(os.path.join(args.path, "*")))
    print(args.path, args.output)
    compute_masking(paths, blur=args.blur, output_path=args.output)