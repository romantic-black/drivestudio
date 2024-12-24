from omegaconf import OmegaConf
from datasets.driving_dataset import DrivingDataset
from utils.mytools import split_trajectory
from typing import Dict
from torch import Tensor
import torch
import numpy as np
import random
from numba import njit
from utils.mytools import Grid2d,Grid2dNumba,farthest_point_sampling, random_point_sampling


class MyDataset(DrivingDataset):

    def __init__(self, data_cfg:OmegaConf,):
        super().__init__(data_cfg)
        self.data_cfg = data_cfg
        self.fake_image_infos = []
        self.fake_cam_infos = []

    def load_fake_gt(self, fake_image_infos, fake_cam_infos, remove_old=False):
        assert len(fake_image_infos) == len(fake_cam_infos)
        if remove_old:
            self.fake_image_infos = []
            self.fake_cam_infos = []
        self.fake_image_infos.extend(fake_image_infos)
        self.fake_cam_infos.extend(fake_cam_infos)

    def fake_gt_next(self):
        index = random.randint(0, len(self.fake_image_infos) - 1)
        return self.fake_image_infos[index], self.fake_cam_infos[index]

    def split_train_test(self):
        if self.data_cfg.my_config.split_key_frame:
            traj = self.get_novel_render_traj(["front_center_interp"], self.frame_num)["front_center_interp"]
            segments, ranges = split_trajectory(traj, min_length=3)
            kf_num = len(segments)
            # segments：list[list[int]]
            if kf_num < 2:
                raise RuntimeError("segments num less than 2, no need to restruct.")
            elif 3 >= kf_num >= 2:  # 选择segments最大list的当训练, 剩下的当测试
                sorted_segments = sorted(segments, key=len, reverse=True)
                train_segments = sorted_segments[:1]
                test_segments = sorted_segments[1:]
            elif 5 >= kf_num > 3:   # 选择segments最大的两个list当训练，剩下的当测试
                sorted_segments = sorted(segments, key=len, reverse=True)
                train_segments = sorted_segments[:2]
                test_segments = sorted_segments[2:]
            elif 8 >= kf_num > 5:
                sorted_segments = sorted(segments, key=len, reverse=True)
                train_segments = sorted_segments[:3]
                test_segments = sorted_segments[3:]
            else:   # 随机选 60 当训练，40 当测试
                random.shuffle(segments)
                split_index = int(0.6 * kf_num) + 1
                train_segments = segments[:split_index]
                test_segments = segments[split_index:]

            # 将segmets合并为一个数组
            train_timesteps = np.concatenate(train_segments)
            test_timesteps = np.concatenate(test_segments)

            train_timesteps = np.sort(train_timesteps)
            test_timesteps = np.sort(test_timesteps)

            # propagate the train and test timesteps to the train and test indices
            train_indices, test_indices = [], []
            for t in range(self.num_img_timesteps):
                if t in train_timesteps:
                    for cam in range(self.pixel_source.num_cams):
                        train_indices.append(t * self.pixel_source.num_cams + cam)
                elif t in test_timesteps:
                    for cam in range(self.pixel_source.num_cams):
                        test_indices.append(t * self.pixel_source.num_cams + cam)

            # Again, training and testing indices are indices into the full dataset
            # train_indices are img indices, so the length is num_cams * num_train_timesteps
            # but train_timesteps are timesteps, so the length is num_train_timesteps (len(unique_train_timestamps))
            return train_timesteps, test_timesteps, train_indices, test_indices
        else:
            return super().split_train_test()
    
def get_fake_gt_samples(dataset: MyDataset,
                        num_points=200,
                        min_coverage=0.6,
                        max_coverage=0.8,
                        cam_width=960,
                        cam_height=640,
                        radius=6,
                        grid_size=0.5,
                        angle_resolution=36,
                        farthest_sample=True
                        ):
    source = dataset.lidar_source
    points = source.origins + source.directions * source.ranges
    grounds = source.grounds
    flow_class = source.flow_classes
    timesteps = source.timesteps

    cameras = dataset.pixel_source.camera_data

    grid = Grid2d(points, grounds, flow_class, timesteps, cameras, grid_size=grid_size, angle_resolution=angle_resolution)
    grid_numba = Grid2dNumba(grid, radius=radius)
    area_coverage = grid_numba.get_area_coverage()

    mask = (grid_numba.area_coverage > min_coverage) & (grid_numba.area_coverage < max_coverage)
    chosen_indices_ = grid_numba.indices[mask]
    if farthest_sample:
        _, chosen_indices = farthest_point_sampling(Tensor(chosen_indices_), num_points)
    else:
        _, chosen_indices = random_point_sampling(Tensor(chosen_indices_), num_points)
    cam2worlds, intrinsics, norm_times, step_times = grid.to_camera_pose(chosen_indices)

    obstacles = points[flow_class <= 0]
    depth_maps = []

    # depth map
    for idx in range(len(chosen_indices)):
        c2w, intrinsic, norm_time, step_time = cam2worlds[idx], intrinsics[idx], norm_times[idx], step_times[idx]
        current_flow = flow_class[timesteps == step_time]
        current_obstacles = points[timesteps == step_time]
        current_objects = current_obstacles[current_flow > 0]
        
        current_points = np.concatenate([current_objects, obstacles])
        intrinsic_4x4 = torch.nn.functional.pad(
            intrinsic, (0, 1, 0, 1)
        )
        intrinsic_4x4[3, 3] = 1.0
        lidar2img = intrinsic_4x4 @ c2w.inverse() 
        current_points = (
            lidar2img[:3, :3] @ current_points.T + lidar2img[:3, 3:4]
        ).T # (num_pts, 3)  

        depth = current_points[:, 2]
        cam_points = current_points[:, :2] / (depth.unsqueeze(-1) + 1e-6) # (num_pts, 2)
        valid_mask = (
            (cam_points[:, 0] >= 0)
            & (cam_points[:, 0] < cam_width)
            & (cam_points[:, 1] >= 0)
            & (cam_points[:, 1] < cam_height)
            & (depth > 0)
        ) # (num_pts, )
        
        cam_points = cam_points[valid_mask].cpu().numpy()
        depth = depth[valid_mask].cpu().numpy()
        depth_map = np.zeros((cam_height, cam_width))
        depth_map = z_buffer(depth_map, cam_points, depth)
        depth_maps.append(torch.Tensor(depth_map))
        
    return cam2worlds, intrinsics, norm_times, step_times, depth_maps


@njit
def z_buffer(depth_map, cam_points, depth):
    for idx, point in enumerate(cam_points):
        z = depth[idx]
        x, y = point
        x, y = int(round(x)), int(round(y))
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            if depth_map[y, x] == 0 or depth_map[y, x] > z:
                depth_map[y, x] = z
    return depth_map


if __name__ == '__main__':
    import os
    data_root = "/mnt/e/Output/background/023_test"

    cfg = OmegaConf.load(os.path.join(data_root, "config.yaml"))
    cfg.data.data_root = "/home/a/drivestudio/data/waymo/processed/training"

    dataset = MyDataset(cfg.data)
    cam2worlds, intrinsics, norm_times, step_times, depth_maps = get_fake_gt_samples(dataset, min_coverage=0.6,
                                                                                     max_coverage=0.8, num_points=100)