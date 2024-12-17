import torch
import numpy as np
from itertools import product
from numba import njit, prange


class Grid2dNumba:
    def __init__(self, grid, radius=8):
        self.voxel_grid_2d = grid.voxel_grid_2d.numpy()
        (x_min, x_max, y_min, y_max) = grid.coord_range

        self.coord_range = x_min.item(), x_max.item(), y_min.item(), y_max.item()
        self.voxel_dict = grid.voxel_dict
        self.angle_resolution = grid.angle_resolution
        self.grid_size = grid.grid_size
        self.poses = grid.cam2worlds[:, :3, 3].numpy()
        self.fov = grid.fov.numpy()
        self.radius = radius

        self.grid_range = (0, self.voxel_grid_2d.shape[0], 0, self.voxel_grid_2d.shape[1])
        self.voxel_indices = np.array(list(self.voxel_dict.keys()))

        # 将 voxel_dict 中的["hit_angles"] 转换为 numpy 数组, 形状为 [voxel_num, max_hit_angles_num]
        hit_angles_list = [voxel["hit_angles"] for voxel in self.voxel_dict.values()]
        max_hit_angles_num = max(len(angles) for angles in hit_angles_list)
        self.hit_angles = np.full((len(hit_angles_list), max_hit_angles_num), np.nan, dtype=np.float32)

        for idx, angles in enumerate(hit_angles_list):
            self.hit_angles[idx, :len(angles)] = angles

        # 取voxel_grid_2d中每个voxel的中心点
        indices = np.meshgrid(np.arange(self.voxel_grid_2d.shape[0]), np.arange(self.voxel_grid_2d.shape[1]),
                              indexing='ij')
        indices = np.stack(indices, axis=-1).reshape(-1, 2)  # 将 indices 转换为二维数组

        # 取在pose半径8m内的点
        theta = np.arange(0, self.angle_resolution, dtype=int)  # 确保 angle_resolution 是整数
        self.area_indices = np.empty((0, 2))
        x, y = self.coord_to_grid(self.poses[:, :2])
        for i in range(x.shape[0]):
            # 计算每个pose到所有indices的距离
            distances = np.linalg.norm(indices - np.array([x[i], y[i]]), axis=1)  # 确保 distances 是一维数组
            mask = distances <= self.radius  # 直接使用布尔数组
            self.area_indices = np.concatenate([self.area_indices, indices[mask]], axis=0)

        # 确保使用正确的 indices 变量
        self.indices = np.array([(x, y, angle) for x, y in self.area_indices for angle in theta]).astype(int)
        self.indices = np.unique(self.indices, axis=0)

    def coord_to_grid(self, poses):
        x, y = poses[:, 0], poses[:, 1]
        (x_min, x_max, y_min, y_max) = self.coord_range
        i = (x - x_min) / self.grid_size
        j = (y - y_min) / self.grid_size
        # 四舍五入
        i = np.round(i).astype(int)
        j = np.round(j).astype(int)
        return i, j

    def get_area_coverage(self):
        return compute_area_coverage(self.voxel_indices, self.hit_angles, np.array(self.coord_range),
                                     np.array(self.grid_range), self.indices, self.fov, self.grid_size)


@njit
def create_voxel_map(voxel_indices, voxel_hit_angles, grid_range):
    """
    假设:
    voxel_indices: (N,2)
    voxel_hit_angles: (N,) 与 voxel_indices 对应
    grid_range: (i_min, i_max, j_min, j_max)

    返回:
    voxel_map: 一个 2D 数组，大小为 (i_max, j_max, 2)
               voxel_map[i,j,0] = start_idx
               voxel_map[i,j,1] = count

    注意：这里假设 i,j 索引非负且在范围内。
    """
    i_min, i_max, j_min, j_max = grid_range
    # 初始化统计个数
    counts = np.zeros((i_max, j_max), dtype=np.int32)
    for n in range(voxel_indices.shape[0]):
        i_ = voxel_indices[n, 0]
        j_ = voxel_indices[n, 1]
        if i_min <= i_ < i_max and j_min <= j_ < j_max:
            counts[i_, j_] += 1

    # 前缀和分配存储空间
    starts = np.zeros((i_max, j_max), dtype=np.int32)
    cumulative = 0
    for i_ in range(i_min, i_max):
        for j_ in range(j_min, j_max):
            cnt = counts[i_, j_]
            starts[i_, j_] = cumulative
            cumulative += cnt

    # 创建索引数组
    voxel_map_indices = np.full((voxel_indices.shape[0],), -1, dtype=np.int32)
    current_counts = np.zeros((i_max, j_max), dtype=np.int32)
    for n in range(voxel_indices.shape[0]):
        i_ = voxel_indices[n, 0]
        j_ = voxel_indices[n, 1]
        if i_min <= i_ < i_max and j_min <= j_ < j_max:
            idx = starts[i_, j_] + current_counts[i_, j_]
            voxel_map_indices[idx] = n
            current_counts[i_, j_] += 1

    return starts, counts, voxel_map_indices


@njit
def compute_area_coverage(voxel_indices, voxel_hit_angles, coord_range, grid_range, area_indices, fov, grid_size,
                          num_rays=100, max_distance=50):
    area_coverage = np.zeros(len(area_indices))
    for idx, (i, j, theta) in enumerate(area_indices):
        start_angle = theta - fov / 2
        end_angle = theta + fov / 2

        (x_min, x_max, y_min, y_max) = coord_range
        cx, cy = (x_min + i * grid_size + grid_size / 2, y_min + j * grid_size + grid_size / 2)
        angles = np.linspace(start_angle, end_angle, num=num_rays)
        coverage = np.zeros_like(angles)
        for idx, ang in enumerate(angles):
            dx = np.cos(ang)
            dy = np.sin(ang)
            x2 = cx + max_distance * dx
            y2 = cy + max_distance * dy

            i2 = round((x2 - x_min) / grid_size)
            j2 = round((y2 - y_min) / grid_size)

            line_points = bresenham_line(i, j, i2, j2, grid_range)
            for i_, j_ in line_points:
                i_ = int(i_)
                j_ = int(j_)
                mask = (voxel_indices[:, 0] == i_) & (voxel_indices[:, 1] == j_)
                voxel_index = np.where(mask)[0]
                if len(voxel_index) == 0:
                    continue
                else:
                    voxel_coverage = compute_voxel_coverage(ang,
                                                            voxel_hit_angles=voxel_hit_angles[voxel_index].reshape(-1))
                    coverage[idx] = voxel_coverage
                    break

        area_coverage[idx] = coverage.mean()
    return area_coverage


@njit
def compute_voxel_coverage(ang, voxel_hit_angles, sigma=np.pi / 20):
    # 去除 nan
    voxel_hit_angles = voxel_hit_angles[~np.isnan(voxel_hit_angles)]
    diff = np.abs(voxel_hit_angles - ang)
    delta = np.minimum(diff, np.pi * 2 - diff)
    delta = delta[delta < np.pi / 2]
    if delta.size == 0:
        return 0
    coverage = np.sum(np.exp(- (delta ** 2) / (2 * sigma ** 2)))
    if coverage > 1.0:
        coverage = 1.0
    elif coverage < 0.0:
        coverage = 0.0
    return coverage


@njit
def bresenham_line(x0, y0, x1, y1, grid_range):
    (i_min, i_max, j_min, j_max) = grid_range
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
            if not (i_min <= x < i_max or j_min <= y < j_max):
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
            if not (i_min <= x < i_max or j_min <= y < j_max):
                break
            points.append((x, y))
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
    return points
