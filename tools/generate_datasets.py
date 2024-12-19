from typing import List, Optional
from omegaconf import OmegaConf
import os
import time
import json
import wandb
import logging
import argparse
import torch
import random
import cv2
import numpy as np
import torch.nn.functional as F
from pathlib import Path
import pyiqa
import shutil

if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lq_dir", type=str, default="/mnt/e/Output/cam5_test")
    parser.add_argument("--output_dir", type=str, default="/mnt/f/DataSet/lora/waymo")
    parser.add_argument("--waymo_dir", type=str, default="/mnt/f/DataSet/waymo/processed/training")
    parser.add_argument("--iqa_threshold", type=float, default=70)
    args = parser.parse_args()

    lq_dir = Path(args.lq_dir)
    output_dir = Path(args.output_dir)
    waymo_dir = Path(args.waymo_dir)

    if not lq_dir.exists():
        print(f"Error: {lq_dir} does not exist")
        exit(1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iqa_metric = pyiqa.create_metric('brisque', device=device)

    if not (output_dir / "lq").exists():
        (output_dir / "lq").mkdir(parents=True, exist_ok=True)
    if not (output_dir / "gt").exists():
        (output_dir / "gt").mkdir(parents=True, exist_ok=True)
    if not (output_dir / "low_quality").exists():
        (output_dir / "low_quality").mkdir(parents=True, exist_ok=True)

    for scene_dir in lq_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        lq_img_dir = lq_dir / scene_dir.name / "test_images"
        img_list = os.listdir(lq_img_dir)
        print(f"Processing {scene_dir.name} with {len(img_list)} images")

        for img_name in img_list:
            if not img_name.endswith(".png"):
                continue
            name = img_name.split(".")[0]
            scene_id, frame_id, cam_id, iters = name.split("_")
            img_path = lq_img_dir / img_name
            lq = cv2.imread(img_path)

            torch_img = torch.from_numpy(lq).permute(2, 0, 1).unsqueeze(0).to(device)
            torch_img = torch_img / 255.0
            quality_score = iqa_metric(torch_img)

            if quality_score > args.iqa_threshold:
                print(f"Skipping {img_name} with quality score {quality_score}")
                cv2.imwrite(str(output_dir / "low_quality"/ img_name), lq)
                continue
            
            gt_img_path = waymo_dir / scene_id / "images"/ f"{frame_id}_{cam_id}.jpg"
            gt = cv2.imread(gt_img_path)
            # resize gt to h / 2, w / 2
            gt = cv2.resize(gt, (gt.shape[1] // 2, gt.shape[0] // 2))

            assert lq.shape == gt.shape
            cv2.imwrite(str(output_dir / "lq"/ img_name), lq)
            cv2.imwrite(str(output_dir / "gt"/ img_name), gt)



