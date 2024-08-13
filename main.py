import torch
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import os
from PIL import Image
from glob import glob
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from utils import colormap
import argparse

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# VIDEOPATH = "/datasets/ucf-crime/jpgs/Anomaly/Robbery102_x264"
VIDEOPATH = "/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test001/"

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("path_video", type=str, default=VIDEOPATH)
    parse.add_argument("path_output", type=str, default="outputs")
    args = parse.parse_args()

    path_video = args.path_video
    path_output = args.path_output

    paths_image = sorted(glob(os.path.join(path_video, "*")))

    with torch.inference_mode(), torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        image = Image.open(paths_image[0])
        image = np.array(image.convert("RGB"))

        # points_per_side is made to be half of the smallest side of the image
        w, h = image.shape[1], image.shape[0]
        points_per_side = w // 2 if w > h else h // 2

        sam2 = build_sam2(
            model_cfg, checkpoint, device="cuda", apply_postprocessing=False
        )

        # make the initial mask
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2, points_per_side=points_per_side
        )
        masks = mask_generator.generate(image)

        state = predictor.init_state(VIDEOPATH)

        for idx, mask in enumerate(masks):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=idx,
                mask=mask["segmentation"],
            )

        # propagate the prompts to get masklets throughout the video
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            masks = masks.cpu().numpy()
            image = Image.open(paths_image[frame_idx])
            image = np.array(image.convert("RGB"))
            # coloring the each mask
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0]
                mask = mask > 0
                image[mask] = colormap[obj_id]
            Image.fromarray(image).save(f"{path_output}/{frame_idx:05}.png")
