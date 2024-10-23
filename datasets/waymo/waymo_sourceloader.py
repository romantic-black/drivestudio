from typing import Dict
import logging
import os
import json
import joblib
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor

from pytorch3d.transforms import matrix_to_quaternion
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData

logger = logging.getLogger()

# define each class's node type
OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.DeformableNodes
}
SMPLNODE_CLASSES = ["Pedestrian"]

# OpenCV to Dataset coordinate transformation
# opencv coordinate system: x right, y down, z front
# waymo coordinate system: x front, y left, z up
OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
)

# Waymo Camera List:
# 0: front_camera
# 1: front_left_camera
# 2: front_right_camera
# 3: left_camera
# 4: right_camera
AVAILABLE_CAM_LIST = [0, 1, 2, 3, 4]

class WaymoCameraData(CameraData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load_calibrations(self):

        # 加载相机内参
        # 1D 数组格式为 [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]。
        # ====!! 为了简化，我们没有使用畸变参数 !!====
        # 未来可以改进!!
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
        
        # 根据加载尺寸缩放内参
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )
        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array([k1, k2, p1, p2, k3])

        # 加载相机外参
        cam_to_ego = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        # 因为我们使用OpenCV坐标系生成相机光线，
        # 需要一个变换矩阵将光线从OpenCV坐标系转换到Waymo坐标系。
        # OpenCV坐标系: x 右, y 下, z 前
        # Waymo坐标系: x 前, y 左, z 上
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # 计算每帧图像的姿态和内参
        cam_to_worlds, ego_to_worlds = [], []
        intrinsics, distortions = [], []

        # 我们将相机姿态相对于第一时间步进行变换，使得
        # 第一个ego姿态的平移向量作为世界坐标系的原点。
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # 计算ego到世界的变换
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
            # 变换过程：
            #   (opencv_cam -> waymo_cam -> waymo_ego_vehicle) -> current_world
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        # 将内参、畸变参数和相机到世界的变换矩阵转换为PyTorch张量
        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

    @classmethod
    def get_camera2worlds(cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        # Load camera extrinsics
        cam_to_ego = np.loadtxt(os.path.join(data_path, "extrinsics", f"{cam_id}.txt"))
        cam_to_ego = cam_to_ego @ OPENCV2DATASET

        # Load ego poses and compute camera-to-world matrices
        cam_to_worlds = []
        ego_to_world_start = np.loadtxt(os.path.join(data_path, "ego_pose", f"{start_timestep:03d}.txt"))
        
        for t in range(start_timestep, end_timestep):
            ego_to_world_current = np.loadtxt(os.path.join(data_path, "ego_pose", f"{t:03d}.txt"))
            ego_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            cam2world = ego_to_world @ cam_to_ego
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

class WaymoPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.load_data()

    def load_cameras(self):
        # 生成时间戳列表
        self._timesteps = torch.arange(self.start_timestep, self.end_timestep)
        self.register_normalized_timestamps()   # 规范化至 [0, 1]
        # 对每个相机生成
        for idx, cam_id in enumerate(self.camera_list): # [0, 1, 2]
            logger.info(f"Loading camera {cam_id}")
            camera = WaymoCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,     # 0
                end_timestep=self.end_timestep,         # 199
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,  # True
                load_sky_mask=self.data_cfg.load_sky_mask,          # True
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],   # [2, 2, 2]
                undistort=self.data_cfg.undistort,      # True
                buffer_downscale=self.buffer_downscale, # 8
                device=self.device,
            )
            camera.load_time(self.normalized_time)
            # 生成基于图像的索引
            unique_img_idx = torch.arange(len(camera), device=self.device) * len(self.camera_list) + idx
            camera.set_unique_ids(
                unique_cam_idx = idx,
                unique_img_idx = unique_img_idx
            )
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera
    
    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """
        instances_info_path = os.path.join(self.data_path, "instances", "instances_info.json")
        frame_instances_path = os.path.join(self.data_path, "instances", "frame_instances.json")
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        # 考虑 frame 和 实例 两个维度
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))   # [199, 164, 4, 4]
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1
        
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for k, v in instances_info.items():
            # 映射到不同建模 model 的枚举
            instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
            for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
                instances_size[frame_idx, int(k)] = np.array(box_size)
        
        # get frame valid instances
        # shape (num_frames, num_instances)
        per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
        for frame_idx, valid_instances in frame_instances.items():
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1    # 扩展 frame_instances, 记录有效矩阵
        
        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(instances_pose[self.start_timestep:self.end_timestep]).float()
        instances_size = torch.from_numpy(instances_size[self.start_timestep:self.end_timestep]).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(per_frame_instance_mask[self.start_timestep:self.end_timestep]).bool()
        
        # 删除未出现过的实例
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]
        
        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(0).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id
        # (num_instances)
        self.instances_model_types = instances_model_types
        
        if self.data_cfg.load_smpl: # True
            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in AVAILABLE_CAM_LIST:   # [0,1,2,3,4]
                cam_to_worlds[cam_id] = WaymoCameraData.get_camera2worlds(
                    self.data_path, 
                    str(cam_id), 
                    self.start_timestep, 
                    self.end_timestep
                )

            # load SMPL parameters
            smpl_dict = joblib.load(os.path.join(self.data_path, "humanpose", "smpl.pkl"))
            frame_num = self.end_timestep - self.start_timestep
            
            smpl_human_all = {}
            for fi in tqdm(range(self.start_timestep, self.end_timestep), desc="Loading SMPL"):
                for instance_id, ins_smpl in smpl_dict.items():
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros((frame_num, 24, 4), dtype=torch.float32),
                            "smpl_trans": torch.zeros((frame_num, 3), dtype=torch.float32),
                            "smpl_betas": torch.zeros((frame_num, 10), dtype=torch.float32),
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool)
                        }
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
                    if ins_smpl["valid_mask"][fi]:
                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][fi - self.start_timestep] = betas
                        
                        body_pose = ins_smpl["smpl"]["body_pose"][fi]
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()
                        
                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )
                        
                        ii = instances_info[str(instance_id)]['frame_annotations']["frame_idx"].index(fi)
                        o2w = np.array(
                            instances_info[str(instance_id)]['frame_annotations']["obj_to_world"][ii]
                        )
                        o2w = torch.from_numpy(
                            np.linalg.inv(ego_to_world_start) @ o2w
                        )
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]
                        
                        smpl_human_all[instance_id]["smpl_quats"][fi - self.start_timestep] = smpl_quats
                        smpl_human_all[instance_id]["smpl_trans"][fi - self.start_timestep] = o2w[:3, 3]
                        smpl_human_all[instance_id]["frame_valid"][fi - self.start_timestep] = True

            self.smpl_human_all = smpl_human_all
            
class WaymoLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, "lidar", f"{t:03d}.bin")
            )
        self.lidar_filepaths = np.array(lidar_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        # Note that in the Waymo Open Dataset, the lidar coordinate system is the same
        # as the vehicle coordinate system
        lidar_to_worlds = []

        # 以 t=0 时刻 ego 位姿为原点
        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first ego pose as the origin of the world coordinate system.
        ego_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "ego_pose", f"{self.start_timestep:03d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:03d}.txt")
            )
            # compute ego_to_world transformation
            lidar_to_world = np.linalg.inv(ego_to_world_start) @ ego_to_world_current
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(    # [199, 4, 4]
            np.stack(lidar_to_worlds, axis=0)
        ).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        # flow/ground info are used for evaluation only
        flows, flow_classes, grounds = [], [], []
        # in waymo, we simplify timestamps as the time indices
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            # each lidar_info contains an Nx14 array
            # from left to right:
            # origins: 3d, points: 3d, flows: 3d, flow_class: 1d,
            # ground_labels: 1d, intensities: 1d, elongations: 1d, laser_ids: 1d
            lidar_info = np.memmap(
                self.lidar_filepaths[t],
                dtype=np.float32,
                mode="r",
            ).reshape(-1, 14)
            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            # select lidar points based on the laser id
            if self.data_cfg.only_use_top_lidar:    # False
                # laser_ids: 0: TOP, 1: FRONT, 2: SIDE_LEFT, 3: SIDE_RIGHT, 4: REAR
                lidar_info = lidar_info[lidar_info[:, 13] == 0]

            lidar_origins = torch.from_numpy(lidar_info[:, :3]).float()  # 提取原点
            lidar_points = torch.from_numpy(lidar_info[:, 3:6]).float()  # 提取点坐标
            lidar_ids = torch.from_numpy(lidar_info[:, 13]).float()  # 提取激光器ID
            lidar_flows = torch.from_numpy(lidar_info[:, 6:9]).float()  # 提取流动信息
            lidar_flow_classes = torch.from_numpy(lidar_info[:, 9]).long()  # 提取流动类别
            ground_labels = torch.from_numpy(lidar_info[:, 10]).long()  # 提取地面标签
            # we don't collect intensities and elongations for now

            # select lidar points based on a truncated ego-forward-directional range
            # this is to make sure most of the lidar points are within the range of the camera
            valid_mask = torch.ones_like(lidar_origins[:, 0]).bool()  # 创建有效掩码
            if self.data_cfg.truncated_max_range is not None:
                valid_mask = lidar_points[:, 0] < self.data_cfg.truncated_max_range  # 应用最大范围过滤
            if self.data_cfg.truncated_min_range is not None:
                valid_mask = valid_mask & (
                    lidar_points[:, 0] > self.data_cfg.truncated_min_range  # 应用最小范围过滤
                )
            lidar_origins = lidar_origins[valid_mask]  # 过滤原点
            lidar_points = lidar_points[valid_mask]  # 过滤点坐标
            lidar_ids = lidar_ids[valid_mask]  # 过滤激光器ID
            lidar_flows = lidar_flows[valid_mask]  # 过滤流动信息
            lidar_flow_classes = lidar_flow_classes[valid_mask]  # 过滤流动类别
            ground_labels = ground_labels[valid_mask]  # 过滤地面标签
            # transform lidar points from lidar coordinate system to world coordinate system
            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T  # 转换原点到世界坐标系
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T  # 转换点坐标到世界坐标系
            # scene flows are in the lidar coordinate system, so we need to rotate them
            lidar_flows = (self.lidar_to_worlds[t][:3, :3] @ lidar_flows.T).T  # 转换流动信息
            # compute lidar directions
            lidar_directions = lidar_points - lidar_origins  # 计算激光点的方向
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)  # 计算激光点的范围
            lidar_directions = lidar_directions / lidar_ranges  # 归一化方向
            # we use time indices as the timestamp for waymo dataset
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t  # 生成时间戳
            accumulated_num_rays += len(lidar_ranges)  # 累加激光点数量

            origins.append(lidar_origins)  # 添加原点
            directions.append(lidar_directions)  # 添加方向
            ranges.append(lidar_ranges)  # 添加范围
            laser_ids.append(lidar_ids)  # 添加激光器ID
            flows.append(lidar_flows)  # 添加流动信息
            flow_classes.append(lidar_flow_classes)  # 添加流动类别
            grounds.append(ground_labels)  # 添加地面标签
            # we use time indices as the timestamp for waymo dataset
            timesteps.append(lidar_timestamp)  # 添加时间戳

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )
        logger.info("Filter condition:")
        logger.info(f"  only_use_top_lidar: {self.data_cfg.only_use_top_lidar}")
        logger.info(f"  truncated_max_range: {self.data_cfg.truncated_max_range}")
        logger.info(f"  truncated_min_range: {self.data_cfg.truncated_min_range}")

        self.origins = torch.cat(origins, dim=0)  # 合并所有原点
        self.directions = torch.cat(directions, dim=0)  # 合并所有方向
        self.ranges = torch.cat(ranges, dim=0)  # 合并所有范围
        self.laser_ids = torch.cat(laser_ids, dim=0)  # 合并所有激光器ID
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()  # 初始化可见掩码
        self.colors = torch.ones_like(self.directions)  # 初始化颜色
        # becasue the flows here are velocities (m/s), and the fps of the lidar is 10,
        # we need to divide the velocities by 10 to get the displacements/flows
        # between two consecutive lidar scans
        self.flows = torch.cat(flows, dim=0) / 10.0  # 合并流动信息并归一化
        self.flow_classes = torch.cat(flow_classes, dim=0)  # 合并流动类别
        self.grounds = torch.cat(grounds, dim=0).bool()  # 合并地面标签

        # the underscore here is important.
        self._timesteps = torch.cat(timesteps, dim=0)  # 合并时间戳
        self.register_normalized_timestamps()  # 注册规范化时间戳

    def to(self, device: torch.device):
        super().to(device)
        self.flows = self.flows.to(device)
        self.flow_classes = self.flow_classes.to(device)
        self.grounds = self.grounds.to(self.device)

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        flows = self.flows[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
            "lidar_flows": flows,
        }

    def delete_invisible_pts(self) -> None:
        """
        Clear the unvisible points.
        """
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self.flows = self.flows[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")

