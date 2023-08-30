# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:15:30 2023

@author: siva
"""

import glob
from groundingdino.util.inference import Model
import os
from typing import List
import supervision as sv

model_path = "models"

# wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
GROUNDING_DINO_CONFIG_PATH = os.path.join("models", "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("weights", "groundingdino_swint_ogc.pth")


def load_model():
    return Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device="cpu"
    )


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


def predict(image, grounding_dino_model, CLASSES):
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=CLASSES),
        box_threshold=0.35,
        text_threshold=0.25,
    )
    return detections


def annotate_image(image, detections, CLASSES):
    box_annotator = sv.BoxAnnotator()
    labels = [f"{CLASSES[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
    return annotated_frame
