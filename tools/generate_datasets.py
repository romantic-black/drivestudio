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

from utils.visualization import to8b
from utils.mytools import split_trajectory
from datasets.driving_dataset import DrivingDataset
from datasets.my_dataset import MyDataset
from utils.misc import import_str
from models.trainers import BasicTrainer
from models.video_utils import (
    render_images,
    save_videos,
    render_novel_views
)


if __name__ == '__main__':
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/e/Output/cam5")
    parser.add_argument("--target_dir", type=str, default="/mnt/f/DataSet/lora/waymo")
    parser.add_argument("--dataset_root", type=str, default="/mnt/f/DataSet/waymo/processed/training")
    parser.add_argument("--iters", type=str, default=[200, 300, 400, 500])
    parser.add_argument("--scene_name", type=str, default="014")
    args = parser.parse_args()

    scene_id_list = os.listdir(args.data_root)
    gt_dir = os.path.join(args.target_dir, "gt")
    os.makedirs(gt_dir, exist_ok=True)
    input_dir = os.path.join(args.target_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    mask_dir = os.path.join(args.target_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    scene_name = args.scene_name
    scene_dir = os.path.join(args.data_root, scene_name)
    dataset_dir = os.path.join(args.dataset_root, scene_name)

    # Ensure subdirectories exist
    os.makedirs(os.path.join(input_dir, scene_name), exist_ok=True)
    os.makedirs(os.path.join(mask_dir, scene_name), exist_ok=True)
    os.makedirs(os.path.join(gt_dir, scene_name), exist_ok=True)

    config_path = os.path.join(scene_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)

    dataset = MyDataset(cfg.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )

    for iter in args.iters:
        if isinstance(iter, str):
            iter = int(iter)
        ckpt_path = os.path.join(scene_dir, f"checkpoint_{iter:05d}.pth")

        trainer.resume_from_checkpoint(
            ckpt_path=ckpt_path,
            load_only_model=True
        )

        traj = dataset.get_novel_render_traj(["front_center_interp"], dataset.frame_num)["front_center_interp"]
        segments, ranges = split_trajectory(traj, min_length=3)
        indices = []
        ground_truth_paths = []

        for seg in segments:
            frame_id = random.choice(seg)
            for cam in range(5):
                indices.append(5 * frame_id + cam)
                gt_path = os.path.join(dataset_dir, "images", f"{frame_id:03d}_{cam}.jpg")
                dynamic_masks_path = os.path.join(dataset_dir, "fine_dynamic_masks", "all", f"{frame_id:03d}_{cam}.png")
                sky_masks_path = os.path.join(dataset_dir, "sky_masks", f"{frame_id:03d}_{cam}.png")
                ground_truth_paths.append({
                    "gt": gt_path,
                    "dynamic_mask": dynamic_masks_path,
                    "sky_mask": sky_masks_path
                })

        results = render_images(trainer, dataset.full_image_set, vis_indices=indices)

        for i, img in enumerate(results["rgbs"]):
            img = to8b(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_id, cam_id = indices[i] // 5, indices[i] % 5
            name = f"{scene_name}_{frame_id:03d}_{cam_id}_{iter:05d}"

            cv2.imwrite(os.path.join(input_dir, scene_name, f"{name}.png"), img)

            gt = cv2.imread(ground_truth_paths[i]["gt"])
            dynamic_mask = cv2.imread(ground_truth_paths[i]["dynamic_mask"])
            sky_mask = cv2.imread(ground_truth_paths[i]["sky_mask"])

            # Ensure mask is boolean before conversion
            mask = (dynamic_mask == [255, 255, 255]).all(axis=2) | (sky_mask == [255, 255, 255]).all(axis=2)
            cv2.imwrite(os.path.join(mask_dir, scene_name, f"{name}.png"), mask.astype(np.uint8) * 255)
            cv2.imwrite(os.path.join(gt_dir, scene_name, f"{name}.png"), gt)

            print(f"Processed {scene_name} {frame_id:03d}_{cam_id}_{iter:05d}")
        del results["rgbs"]

            