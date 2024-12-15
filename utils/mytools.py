import torch
import numpy as np

def split_trajectory(trajectory, num_splits=0, min_count=1, min_length=0):
    """
    Split trajectory into segments.
    Args:
        trajectory (torch.Tensor): Trajectory tensor of shape [frame_num, 4, 4].
        num_splits (int): Number of splits. If 0, the function will automatically determine the number of splits.
        min_count (int): Minimum number of occurence of each split.
        min_length (float): Minimum length of each split.
    Returns:
        segments (list): List of segments, each segment is a list of frame indices.
        ranges (torch.Tensor): Tensor of shape [num_splits, 2], each row is a range [start, end].
    """

    positions = trajectory[:, :3, 3].cpu()

    delta_positions = positions[1:] - positions[:-1]  # 相邻帧的位置差，形状为[frame_num - 1, 3]
    distances = torch.norm(delta_positions, dim=1)    # 相邻帧之间的距离，形状为[frame_num - 1]
    cumulative_distances = torch.cat([torch.tensor([0.0], device=distances.device), torch.cumsum(distances, dim=0)])  # 累积距离，形状为[frame_num]
    total_distance = cumulative_distances[-1]

    # 初始化区段数量为帧数
    frame_num = positions.shape[0]
    max_segments = frame_num

    # 自适应计算最大可行的区段数量
    if num_splits == 0:
        for n in range(max_segments, 0, -1):
            # 计算每个区段的边界距离
            segment_boundaries = torch.linspace(0, total_distance, steps=n + 1)

            # 使用bucketize函数确定每帧所属的区段索引
            segment_indices = torch.bucketize(cumulative_distances, segment_boundaries, right=False) - 1
            segment_indices = torch.clamp(segment_indices, min=0, max=n - 1)  # 确保索引在有效范围内

            # 统计每个区段的帧数
            counts = torch.bincount(segment_indices, minlength=n)

            # 检查是否所有区段都有至少一帧且长度满足最小长度
            segment_lengths = segment_boundaries[1:] - segment_boundaries[:-1]
            if torch.all(counts >= min_count) and torch.all(segment_lengths >= min_length):
                # 找到了最大的n，使得每个区段至少有一帧且长度满足最小长度
                num_splits = n
                break

    segment_length = total_distance / num_splits

    segment_indices = (cumulative_distances / segment_length).long()
    segment_indices = torch.clamp(segment_indices, max=num_splits-1)
    segment_indices = segment_indices
    segments = [[] for _ in range(num_splits)]
    boundaries = torch.linspace(0, total_distance, steps=num_splits + 1)
    start, end = boundaries[:-1], boundaries[1:]
    ranges = torch.stack([start, end], dim=1)
    for i in range(num_splits):

        indices = torch.where(segment_indices == i)[0].tolist()
        segments[i] = indices

    return segments, ranges
 

