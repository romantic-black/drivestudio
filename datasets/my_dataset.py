
from datasets.driving_dataset import DrivingDataset
from utils.mytools import split_trajectory
from typing import Dict
from torch import Tensor
import random

class MyDataset(DrivingDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        traj = self.get_novel_render_traj(["front_center_interp"], self.frame_num)["front_center_interp"]
        self.segments, self.ranges = split_trajectory(traj)
        # monkey patch
        self.pixel_source.propose_training_image = self.propose_training_image

    
    def get_segment(self, idx):
        return self.segments[idx]
    
    def get_range(self, idx):
        return self.ranges[idx]

    def propose_training_image(
        self,
        candidate_indices: Tensor = None,
    ) -> Dict[str, Tensor]:
        while True:
            selected_segment = random.choice(self.segments)
            selected_frame = random.choice(selected_segment)
            selected_cam = random.choice(range(self.num_cams))
            selected_idx = selected_cam + selected_frame * self.num_cams
            if selected_idx in candidate_indices:
                break
        return selected_idx
    
