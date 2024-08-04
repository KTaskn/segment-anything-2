import torch
from sam2.build_sam import build_sam2_video_predictor
import numpy as np
import os
from PIL import Image
from glob import glob
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

def colormap(rgb=True):
	color_list = np.array(
		[
			0.000, 0.000, 0.000,
			1.000, 1.000, 1.000,
			1.000, 0.498, 0.313,
			0.392, 0.581, 0.929,
			0.000, 0.447, 0.741,
			0.850, 0.325, 0.098,
			0.929, 0.694, 0.125,
			0.494, 0.184, 0.556,
			0.466, 0.674, 0.188,
			0.301, 0.745, 0.933,
			0.635, 0.078, 0.184,
			0.300, 0.300, 0.300,
			0.600, 0.600, 0.600,
			1.000, 0.000, 0.000,
			1.000, 0.500, 0.000,
			0.749, 0.749, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 1.000,
			0.667, 0.000, 1.000,
			0.333, 0.333, 0.000,
			0.333, 0.667, 0.000,
			0.333, 1.000, 0.000,
			0.667, 0.333, 0.000,
			0.667, 0.667, 0.000,
			0.667, 1.000, 0.000,
			1.000, 0.333, 0.000,
			1.000, 0.667, 0.000,
			1.000, 1.000, 0.000,
			0.000, 0.333, 0.500,
			0.000, 0.667, 0.500,
			0.000, 1.000, 0.500,
			0.333, 0.000, 0.500,
			0.333, 0.333, 0.500,
			0.333, 0.667, 0.500,
			0.333, 1.000, 0.500,
			0.667, 0.000, 0.500,
			0.667, 0.333, 0.500,
			0.667, 0.667, 0.500,
			0.667, 1.000, 0.500,
			1.000, 0.000, 0.500,
			1.000, 0.333, 0.500,
			1.000, 0.667, 0.500,
			1.000, 1.000, 0.500,
			0.000, 0.333, 1.000,
			0.000, 0.667, 1.000,
			0.000, 1.000, 1.000,
			0.333, 0.000, 1.000,
			0.333, 0.333, 1.000,
			0.333, 0.667, 1.000,
			0.333, 1.000, 1.000,
			0.667, 0.000, 1.000,
			0.667, 0.333, 1.000,
			0.667, 0.667, 1.000,
			0.667, 1.000, 1.000,
			1.000, 0.000, 1.000,
			1.000, 0.333, 1.000,
			1.000, 0.667, 1.000,
			0.167, 0.000, 0.000,
			0.333, 0.000, 0.000,
			0.500, 0.000, 0.000,
			0.667, 0.000, 0.000,
			0.833, 0.000, 0.000,
			1.000, 0.000, 0.000,
			0.000, 0.167, 0.000,
			0.000, 0.333, 0.000,
			0.000, 0.500, 0.000,
			0.000, 0.667, 0.000,
			0.000, 0.833, 0.000,
			0.000, 1.000, 0.000,
			0.000, 0.000, 0.167,
			0.000, 0.000, 0.333,
			0.000, 0.000, 0.500,
			0.000, 0.000, 0.667,
			0.000, 0.000, 0.833,
			0.000, 0.000, 1.000,
			0.143, 0.143, 0.143,
			0.286, 0.286, 0.286,
			0.429, 0.429, 0.429,
			0.571, 0.571, 0.571,
			0.714, 0.714, 0.714,
			0.857, 0.857, 0.857
		]
	).astype(np.float32)
	color_list = color_list.reshape((-1, 3)) * 255
	if not rgb:
		color_list = color_list[:, ::-1]
	return color_list

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# VIDEOPATH = "/datasets/ucf-crime/jpgs/Anomaly/Robbery102_x264"
VIDEOPATH = "/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/Test001/"

colors = colormap()[3:]

paths_image = sorted(glob(os.path.join(VIDEOPATH, "*")))
with torch.inference_mode(), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    image = Image.open(paths_image[0])
    image = np.array(image.convert("RGB"))

    sam2 = build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
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
        for obj_id, mask in zip(object_ids, masks):
            mask = mask[0]
            mask = (mask > 0)
            image[mask] = colors[obj_id]
        Image.fromarray(image).save(f"outputs/{frame_idx:05}.png")
            