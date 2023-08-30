# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:15:30 2023

@author: siva
"""

import os
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import supervision as sv
import cv2

DEVICE = "cpu"
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_l_0b3195.pth")
SAM_ENCODER_VERSION = "vit_l"


def load_model():
    return sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)


def predict(image, sam_model, detections):
    sam_predictor = SamPredictor(sam_model)
    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor, image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), xyxy=detections.xyxy
    )
    return detections


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def annotate_image(image, detections, CLASSES):
    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [f"{CLASSES[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
    return annotated_image
