import torch
import numpy as np
from numba import njit, prange


class Grid2dNumba:
    def __init__(self, grid, radius=8, step=0.5, angle_step=1):
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
        
        self.hit_angles = np.concatenate(hit_angles_list, axis=0)

        hit_angles_num = np.array([len(angles) for angles in hit_angles_list])
        
        i_min, i_max, j_min, j_max = self.grid_range
        starts = np.zeros((i_max, j_max), dtype=np.int32)
        counts = np.zeros((i_max, j_max), dtype=np.int32)
        start = 0
        for idx, (i, j) in enumerate(self.voxel_indices):
            counts[i, j] = hit_angles_num[idx]
            starts[i, j] = start
            start += counts[i, j]
        
        self.starts = starts
        self.counts = counts

        self.step = step
        self.angle_step = angle_step
        step = self.step / self.grid_size

        # 取voxel_grid_2d中每个voxel的中心点
        indices = np.meshgrid(np.arange(self.voxel_grid_2d.shape[0], dtype=int, step=step),
                              np.arange(self.voxel_grid_2d.shape[1], dtype=int, step=step),
                              indexing='ij')
        indices = np.stack(indices, axis=-1).reshape(-1, 2)  # 将 indices 转换为二维数组

        # 取在pose半径8m内的点
        theta = np.arange(0, self.angle_resolution, dtype=int, step=self.angle_step)  # 确保 angle_resolution 是整数
        self.area_indices = np.empty((0, 2))
        x, y = self.coord_to_grid(self.poses[:, :2])
        for i in range(x.shape[0]):
            # 计算每个pose到所有indices的距离
            distances = np.linalg.norm(indices - np.array([x[i], y[i]]), axis=1)  # 确保 distances 是一维数组
            mask = distances <= self.radius / self.grid_size  # 直接使用布尔数组
            self.area_indices = np.concatenate([self.area_indices, indices[mask]], axis=0)

        # area_indices 中 不能包含 voxel_indices 的点
        self.area_indices = np.setdiff1d(self.area_indices, self.voxel_indices)

        # 确保使用正确的 indices 变量
        self.indices = np.array([(x, y, angle) for x, y in self.area_indices for angle in theta]).astype(int)
        self.indices = np.unique(self.indices, axis=0)

        self.hit_angles = self.hit_angles % 2 * np.pi



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
        self.area_coverage = compute_area_coverage(self.hit_angles, np.array(self.coord_range),
                                     np.array(self.grid_range), self.indices, self.fov, self.grid_size,
                                     self.starts, self.counts, self.angle_resolution)
        return self.area_coverage
    
    def get_shown_map(self):
        canvas = np.zeros((self.voxel_grid_2d.shape[0], self.voxel_grid_2d.shape[1]))
        count = np.zeros((self.voxel_grid_2d.shape[0], self.voxel_grid_2d.shape[1]))
        for idx, (i, j, theta) in enumerate(self.indices):
            cov = self.area_coverage[idx]
            # 取均值
            canvas[i, j] += cov
            count[i, j] += 1
        canvas = np.where(count > 0, canvas / count, 0)
        return canvas

    def get_shown_map_base_on_angel(self, theta):
        canvas = np.zeros((self.voxel_grid_2d.shape[0], self.voxel_grid_2d.shape[1]))
        for idx, (i, j, theta_) in enumerate(self.indices):
            if theta_ == theta:
                cov = self.area_coverage[idx]
                canvas[i, j] = cov
        return canvas
        


@njit
def compute_area_coverage(voxel_hit_angles, coord_range, grid_range, indices, fov, grid_size,
                          starts, counts, angle_resolution, num_rays=20, max_distance=50):
    # 提前计算好射线的相对角度
    area_coverage = np.zeros(len(indices))
    (x_min, x_max, y_min, y_max) = coord_range
    (i_min, i_max, j_min, j_max) = grid_range
    show_freq = 1000

    for a_idx in range(len(indices)):

        i, j, theta = indices[a_idx]
        angle = theta * (2 * np.pi / angle_resolution)
        angle = angle % (2 * np.pi)
        start_angle = angle - fov / 2
        end_angle = angle + fov / 2

        start_angle = start_angle % (2 * np.pi)
        end_angle = end_angle % (2 * np.pi)

        cx = x_min + i * grid_size + grid_size / 2
        cy = y_min + j * grid_size + grid_size / 2

        angles = np.linspace(start_angle, end_angle, num=num_rays)

        # 预先计算cos和sin
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)

        coverage = np.zeros(num_rays)
        for ridx in range(num_rays):
            ang = angles[ridx]
            dx = cos_vals[ridx]
            dy = sin_vals[ridx]
            x2 = cx + max_distance * dx
            y2 = cy + max_distance * dy

            i2 = int(round((x2 - x_min) / grid_size))
            j2 = int(round((y2 - y_min) / grid_size))

            line_points = bresenham_line(i, j, i2, j2, grid_range)
            found = False
            for (i_, j_) in line_points:
                # 通过预处理的 starts, counts 快速获取对应voxel
                c = counts[i_, j_]
                if c > 0:
                    st = starts[i_, j_]
                    # 这里 idx_slice 对应的 voxel_hit_angles 全部检查
                    voxel_cov = compute_voxel_coverage(ang, voxel_hit_angles[st:st + c])
                    coverage[ridx] = voxel_cov
                    found = True
                    break
            if not found:
                coverage[ridx] = 0.0

        area_coverage[a_idx] = coverage.mean()
        if a_idx % show_freq == 0:
            print(a_idx)

    return area_coverage


@njit
def compute_voxel_coverage(ang, voxel_hit_angles, sigma=np.pi/6, weight=0.1):
    # 假设已无nan
    ang = ang % (2 * np.pi)
    diff = np.abs(voxel_hit_angles - ang)
    delta = np.minimum(diff, np.pi * 2 - diff)
    count = 0
    total = 0.0
    threshold = np.pi/9
    for d in delta:
        if d < threshold:
            val = np.exp(- (d*d) / (2 * sigma * sigma))
            total += val * weight
            count += 1
    if count == 0:
        return 0.0
    coverage = total  # 因为每个命中角度最大记1，相加后若>1截断
    if coverage > 1.0:
        coverage = 1.0
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

    if dx >= dy:
        err = dx / 2
        while x != x1:
            if not (i_min <= x < i_max and j_min <= y < j_max):
                break
            points.append((x, y))
            x += sx
            err -= dy
            if err < 0:
                y += sy
                err += dx
    else:
        err = dy / 2
        while y != y1:
            if not (i_min <= x < i_max and j_min <= y < j_max):
                break
            points.append((x, y))
            y += sy
            err -= dx
            if err < 0:
                x += sx
                err += dy
    return points
