import torch
import numpy as np
from itertools import product

from sympy.physics.quantum.gate import normalized

from utils.grid_numba import Grid2dNumba
import requests

def load_osediff(port):
    url = f"http://127.0.0.1:{port}/load_model"
    response = requests.get(url, timeout=60)
    return response.json()

def process_osediff(port, input_folder, output_folder):
    url = f"http://127.0.0.1:{port}/process_folder"
    data = {
        "input_folder": input_folder,
        "output_folder": output_folder
    }
    response = requests.post(url, data=data, timeout=3600)
    return response.json()

def clear_osediff(port):
    url = f"http://127.0.0.1:{port}/clear_model"
    response = requests.get(url)
    return response.json()


def farthest_point_sampling(points, K, angle_resolution=36, pos_range=10):
    """
    使用最远点采样从点集中采样 K 个点的索引和子集。
    points: [N, 3] 的张量, 列为 (x,y,a)，其中 a 是 [0,35] 的离散角度索引
    K:      需要采样的点数

    返回:
    farthest_indices: [K] 的 LongTensor，为选中点的索引
    sampled_points:   [K, 3] 的张量，为选中点的集合
    """
    device = points.device
    N = points.shape[0]
    if K >= N:
        return torch.arange(N, device=device), points

    # 提取 x, y, a
    x = points[:, 0] / pos_range
    y = points[:, 1] / pos_range
    a = points[:, 2]

    # 将 a 转换为弧度
    # a 的取值 0~35, 对应角度为 a * 10 度
    # 弧度 = 角度 * π/180 = (a*10)*π/180 = a * (π/18)
    angle_rad = a * (2 * torch.pi / angle_resolution)
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)

    # 构造用于距离计算的特征空间 (x, y, cos_a, sin_a)
    coords = torch.stack([x, y, cos_a, sin_a], dim=1)  # [N,4]

    # 初始化：随机选择一个点作为第一个中心点
    farthest_indices = torch.zeros(K, dtype=torch.long, device=device)
    farthest_indices[0] = torch.randint(0, N, (1,), device=device)

    # dist 记录每个点与已选中心点集合中最近的中心点的距离
    dist = torch.full((N,), float('inf'), device=device)

    # 更新距离：计算每个点到第一个已选中点的距离
    initial_center = coords[farthest_indices[0]].unsqueeze(0)  # [1,4]
    dist = torch.sum((coords - initial_center) ** 2, dim=-1)  # [N]

    for i in range(1, K):
        # 选择距离最远的点作为下一个中心点
        farthest_indices[i] = torch.argmax(dist)
        new_center = coords[farthest_indices[i]].unsqueeze(0)  # [1,4]
        new_dist = torch.sum((coords - new_center) ** 2, dim=-1)
        dist = torch.minimum(dist, new_dist)

    sampled_points = points[farthest_indices]
    return farthest_indices, sampled_points


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





class Grid2d:
    def __init__(self, lidar_points, ground_mask, flow_class, timesteps, gt_cameras, angle_resolution=36, grid_size=0.5):
        assert lidar_points.shape[1] == 3
        self.lidar_points = lidar_points
        self.ground_mask = ground_mask
        self.flow_class = flow_class
        self.timesteps = timesteps
        self.angle_resolution = angle_resolution
        p = self.lidar_points[~self.ground_mask]
        c = self.flow_class[~self.ground_mask]
        p = p[c <= 0]
        self.obstacles = p
        self.grounds = self.lidar_points[self.ground_mask]

        self.grid_size = grid_size
        self.voxel_grid_2d, self.coord_range = create_voxel_grid_2d(p, self.grid_size)
        self.grid_range = (0, self.voxel_grid_2d.shape[0], 0, self.voxel_grid_2d.shape[1])
        self.voxel_dict = {
            (i, j): {
                "coord":self.grid_to_coord(i, j), 
                "count":self.voxel_grid_2d[i, j],
                "hit_angles": [],
                }
            for i in range(self.voxel_grid_2d.shape[0])
            for j in range(self.voxel_grid_2d.shape[1])
            if self.voxel_grid_2d[i, j]
        }
        cam2worlds = []
        intrinsics = []
        normalized_time = []
        step_time = []

        for cam in gt_cameras.values():
            cam2worlds.append(cam.cam_to_worlds)
            intrinsics.append(cam.intrinsics)
            normalized_time.append(cam.normalized_time)

            step_time.append(torch.Tensor(list(range(cam.start_timestep, cam.end_timestep))))

        self.step_time = torch.cat(step_time, dim=0)
        self.cam2worlds = torch.cat(cam2worlds, dim=0)
        self.intrinsics = torch.cat(intrinsics, dim=0)
        self.normalized_time = torch.cat(normalized_time, dim=0)
        self.gt_camera_set = CameraSet(self.cam2worlds, self.intrinsics)
        self.ray_casting_set(self.gt_camera_set)
        self.fov = self.gt_camera_set.fovs[0]
        self.gt_cameras = gt_cameras


    def angle_to_theta(self, angle):
        angle = angle % (2 * np.pi)
        theta = int(angle / (2 * np.pi / self.angle_resolution))
        return theta
    
    def theta_to_angle(self, theta):
        angle = theta * (2 * np.pi / self.angle_resolution)
        angle = angle % (2 * np.pi)
        return angle


    def get_hot_map(self):
        hot_map = torch.zeros_like(self.voxel_grid_2d)
        hot_map[self.voxel_grid_2d > 0] = 1
        hit = [[*indice] for indice in self.voxel_dict if len(self.voxel_dict[indice]["hit_angles"]) > 0]
        if hit:  # 确保 hit 列表非空
            hit_indices = torch.tensor(hit).long()  # 将hit转换为tensor
            hot_map[hit_indices[:, 0], hit_indices[:, 1]] = 2  # 对应位置设为2
            
        return hot_map

    def is_in_grid(self, i, j):
        (i_min, i_max, j_min, j_max) = self.grid_range
        return i_min <= i < i_max and j_min <= j < j_max
    
    def is_in_coord(self, x, y):
        (x_min, x_max, y_min, y_max) = self.coord_range
        return x_min <= x < x_max and y_min <= y < y_max


    def grid_to_coord(self, i, j, center=True):
        (x_min, x_max, y_min, y_max) = self.coord_range
        if center:
            return (x_min + i * self.grid_size + self.grid_size / 2, y_min + j * self.grid_size + self.grid_size / 2)
        else:
            return (x_min + i * self.grid_size, y_min + j * self.grid_size)
    
    def coord_to_grid(self, x, y):
        (x_min, x_max, y_min, y_max) = self.coord_range
        i = int(torch.round((x - x_min) / self.grid_size))
        j = int(torch.round((y - y_min) / self.grid_size))
        return (i, j)
    
    def to_camera_pose(self, indices):
        normalized_time = self.normalized_time
        step_time = self.step_time

        gt_xy = self.gt_camera_set.camera_to_worlds[:, :2, 3]

        cam2worlds = []
        intrinsics = []
        norm_times = []
        step_times = []

        for idx, (i, j, theta) in enumerate(indices):
            x, y = self.grid_to_coord(i, j)
            angle = self.theta_to_angle(theta)

            # 取最近的gt, 使用torch.norm
            gt_indice = torch.argmin(torch.norm(gt_xy - torch.tensor([x, y]), dim=1))
            base_cam2world = self.gt_camera_set.camera_to_worlds[gt_indice]
            base_intrinsic = self.gt_camera_set.intrinsics[0]   # 使用主相机内参
            base_angle = self.gt_camera_set.angles[gt_indice]
            norm_t = normalized_time[gt_indice]
            step = step_time[gt_indice]

            R_base_to_0 = torch.tensor([
                [torch.cos(-base_angle), -torch.sin(-base_angle), 0],
                [torch.sin(-base_angle), torch.cos(-base_angle),  0],
                [0,                     0,                       1]
            ], device=base_cam2world.device, dtype=base_cam2world.dtype)

            cam2world = base_cam2world.clone()

            # 计算绕 Z 轴的新旋转矩阵 R_z
            R_z = torch.tensor([
                [torch.cos(angle), -torch.sin(angle), 0],
                [torch.sin(angle), torch.cos(angle),  0],
                [0,                0,                 1]
            ], device=cam2world.device, dtype=cam2world.dtype)

            # 更新 cam2world 的旋转部分和位移部分
            cam2world[:3, :3] = R_z @ R_base_to_0 @ base_cam2world[:3, :3]
            cam2world[0, 3] = x      # 设置平移部分 x
            cam2world[1, 3] = y      # 设置平移部分 y

            intrinsic = base_intrinsic.clone()

            cam2worlds.append(cam2world)
            intrinsics.append(intrinsic)
            norm_times.append(norm_t)
            step_times.append(step)

        cam2worlds = torch.stack(cam2worlds, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        norm_times = torch.stack(norm_times, dim=0)
        step_times = torch.stack(step_times, dim=0)
        return cam2worlds, intrinsics, norm_times, step_times
    

    def ray_casting(self, camera, num_rays=20, max_distance=50, save_hit=True, get_coverage=False):
        hit_points = []
        coverages = []
        cx, cy, angle = camera.cx, camera.cy, camera.angle
        i1, j1 = self.coord_to_grid(cx, cy)
        cx, cy = self.grid_to_coord(i1, j1)
        assert self.is_in_grid(i1, j1)

        start_angle = angle - camera.fov / 2
        end_angle = angle + camera.fov / 2

        angles = torch.linspace(start_angle, end_angle, steps=num_rays)


        for ang in angles:
            # 计算方向向量
            dx = torch.cos(ang).item()
            dy = torch.sin(ang).item()

            # 计算终点（以最大距离为界）
            x2 = cx + max_distance * dx
            y2 = cy + max_distance * dy
            i2, j2 = self.coord_to_grid(x2, y2)

            line_points = self.bresenham_line(i1, j1, i2, j2)
            hit_cell = None

            for i, j in line_points:
                if (i, j) in self.voxel_dict:
                    hit_cell = (i, j)
                    if save_hit:
                        self.voxel_dict[(i, j)]["hit_angles"].append(ang)
                    if get_coverage:
                        coverage = self.get_voxel_coverage(i, j, ang)
                        coverages.append(coverage)
                    break
            
            if hit_cell:
                hit_points.append(hit_cell)

            if not hit_cell and get_coverage:
                coverages.append(0)
        if get_coverage:
            return hit_points, coverages
        else:
            return hit_points

    def ray_casting_set(self, camera_set):
        for camera in camera_set:
            self.ray_casting(camera)

    def get_voxel_coverage(self, i, j, angle, sigma=np.pi/180):
        if (i, j) not in self.voxel_dict:
            return 0
        if not self.voxel_dict[(i, j)]["hit_angles"]:
            return 0
        
        hit_angles = self.voxel_dict[(i, j)]["hit_angles"]
        hit_angles = np.array(hit_angles)

        diff = np.abs(hit_angles - angle.item())
        delta = np.minimum(diff, np.pi * 2 - diff)
        delta = delta[delta < np.pi / 2]
        if delta.size == 0:
            return 0
        coverage = np.sum(np.exp(- (delta**2) / (2 * sigma**2)))

        # 限制coverage在0到1之间
        coverage = np.clip(coverage, 0, 1)
        return coverage
    
    def get_camera_coverage(self, camera):
        hit_points, coverages = self.ray_casting(camera, save_hit=False, get_coverage=True)
        return np.mean(coverages) if len(coverages) else 0


    def get_camera_set_coverage(self, camera_set):
        coverages = []
        for camera in camera_set:
            coverages.append(self.get_camera_coverage(camera))
        return coverages


    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        # 当以 x 为主轴遍历
        if dx >= dy:
            err = dx / 2
            while x != x1:
                if not self.is_in_grid(x, y):
                    break
                points.append((x, y))
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
        else:
            # 当以 y 为主轴遍历
            err = dy / 2
            while y != y1:
                if not self.is_in_grid(x, y):
                    break
                points.append((x, y))
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
        return points

class Camera:
    def __init__(self, camera_to_world=None, intrinsic=None, cx=None, cy=None, angle=None, fov=None):
        if cx is None:
            assert camera_to_world.shape == (4, 4)
            assert intrinsic.shape == (3, 3)
            
            self.camera_to_world = camera_to_world
            self.intrinsic = intrinsic
            self.fov = 2 * torch.atan2(intrinsic[0, 2], intrinsic[0, 0])
            self.cx = camera_to_world[0, 3]
            self.cy = camera_to_world[1, 3]
            self.angle = torch.atan2(camera_to_world[1, 0], camera_to_world[0, 0])
        else:
            self.cx = cx
            self.cy = cy
            self.angle = angle
            self.fov = fov


    def get_attr(self):
        return self.cx, self.cy, self.angle, self.fov


class CameraSet:
    def __init__(self, camera_to_worlds, intrinsics):
        self.camera_to_worlds = camera_to_worlds
        self.intrinsics = intrinsics
        # 向量化
        self.fovs = 2 * torch.atan2(intrinsics[:, 0, 2], intrinsics[:, 0, 0])
        self.cxs = camera_to_worlds[:, 0, 3]
        self.cys = camera_to_worlds[:, 1, 3]
        self.angles = torch.atan2(camera_to_worlds[:, 1, 0], camera_to_worlds[:, 0, 0])
    
    def __len__(self):
        return self.fovs.shape[0]
    
    def __getitem__(self, index):
        return Camera(self.camera_to_worlds[index], self.intrinsics[index])



def create_voxel_grid_2d(lidar_points, voxel_size=0.5):
    z = lidar_points[:, 2]
    lidar_points = lidar_points[(-1 <= z) & (z <= 3)]
    x_min, x_max = lidar_points[:, 0].min(), lidar_points[:, 0].max()
    y_min, y_max = lidar_points[:, 1].min(), lidar_points[:, 1].max()
    
    grid_size_x = int((x_max - x_min) / voxel_size) + 1
    grid_size_y = int((y_max - y_min) / voxel_size) + 1
    
    voxel_grid = torch.zeros((grid_size_x, grid_size_y), dtype=torch.int32)
    
    # 计算每个点的体素索引
    voxel_indices_x = ((lidar_points[:, 0] - x_min) / voxel_size).long()
    voxel_indices_y = ((lidar_points[:, 1] - y_min) / voxel_size).long()
    
    # 将二维索引展平成一维索引
    flat_indices = voxel_indices_x * grid_size_y + voxel_indices_y
    
    # 使用 torch.bincount 统计每个体素的点数量
    counts = torch.bincount(flat_indices, minlength=grid_size_x * grid_size_y)
    
    # 将一维结果重塑为二维网格
    voxel_grid = counts.view(grid_size_x, grid_size_y)
    
    return voxel_grid, (x_min, x_max, y_min, y_max)

if __name__ == '__main__':

    points = torch.load("/home/a/drivestudio/notebook/data/points.pth")
    grounds = torch.load("/home/a/drivestudio/notebook/data//grounds.pth")
    flow_class = torch.load("/home/a/drivestudio/notebook/data//flow_class.pth")
    cameras = torch.load("/home/a/drivestudio/notebook/data//cameras.pth")

    grid = Grid2d(points, grounds, flow_class, timesteps, cameras)

    grid_numba = Grid2dNumba(grid, radius=8)
    coverage = grid_numba.get_area_coverage()
    print(coverage)