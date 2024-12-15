from oauthlib.uri_validate import segment

from datasets.driving_dataset import DrivingDataset
from utils.mytools import split_trajectory
from typing import Dict
from torch import Tensor
import numpy as np
import random

class MyDataset(DrivingDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    #     traj = self.get_novel_render_traj(["front_center_interp"], self.frame_num)["front_center_interp"]
    #     self.segments, self.ranges = split_trajectory(traj)
    #     # monkey patch
    #     self.pixel_source.propose_training_image = self.propose_training_image

    
    # def get_segment(self, idx):
    #     return self.segments[idx]
    
    # def get_range(self, idx):
    #     return self.ranges[idx]

    # def propose_training_image(
    #     self,
    #     candidate_indices: Tensor = None,
    # ) -> Dict[str, Tensor]:
    #     while True:
    #         selected_segment = random.choice(self.segments)
    #         selected_frame = random.choice(selected_segment)
    #         selected_cam = random.choice(range(self.num_cams))
    #         selected_idx = selected_cam + selected_frame * self.num_cams
    #         if selected_idx in candidate_indices:
    #             break
    #     return selected_idx

    def split_train_test(self):
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
    
