from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
import cv2
from tools.eval import do_evaluation
from utils.visualization import to8b
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset
from utils.osediff import OSEDiffInfer

from datasets.my_dataset import get_fake_gt_samples

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)

    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")

    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)

    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup", "test_images"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)

    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
                wandb.init(
                    project=args.project,
                    entity=args.entity,
                    sync_tensorboard=True,
                    settings=wandb.Settings(start_method="fork"),
                )
                is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)

    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")

    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./",
        ["configs", "datasets", "models", "utils", "tools"],
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg


def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = import_str(cfg.my_config.dataset_base_type)(data_cfg=cfg.data)

    # setup trainer
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

    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )

    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)

    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")

    # setup optimizer
    trainer.initialize_optimizer()

    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)

    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        # ----------------------------------------------------------------------------
        # ----------------------------     Validate     ------------------------------
        # 该 if 块不重要, 跳过
        if cfg.data.my_config.split_key_frame:
            if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0 and step >= 10000:
                logger.info("Visualizing...")
                with torch.no_grad():
                    render_results = render_images(  # 只是渲染, 没有训练
                        trainer=trainer,
                        dataset=dataset.test_image_set,
                        compute_metrics=False,
                        compute_error_map=cfg.render.vis_error,
                        vis_indices=None,
                    )
                test_indices = dataset.test_indices
                for i, img in enumerate(render_results["rgbs"]):
                    img = to8b(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    idx = test_indices[i]
                    frame_id, cam_id = idx // dataset.num_cams, idx % dataset.num_cams
                    name = f"{dataset.scene_idx:03d}_{frame_id:03d}_{cam_id}_{step:05d}"
                    cv2.imwrite(os.path.join(cfg.log_dir, "test_images", f"{name}.png"), img)
                del render_results
                torch.cuda.empty_cache()

        # ----------------------------------------------------------------------------
        # ----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()  # zero grad
        train_step_camera_downscale = trainer._get_downscale_factor()
        use_fake_gt = random.random() < 0.1
        if use_fake_gt and train_step_camera_downscale == 1.:
            image_infos, cam_infos = dataset.fake_gt_next()
        else:
            image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)

        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)

        # forward & backward
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
        )
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)

        # after training step
        trainer.postprocess_per_train_step(step=step)

        # ----------------------------------------------------------------------------
        # -------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/" + k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/" + k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(
            **{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        # ----------------------------------------------------------------------------
        # ----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
                (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) # and (args.resume_from is None)
        if do_save:
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )

        # ----------------------------------------------------------------------------
        # ------------------------    Cache Image Error    ---------------------------
        if (
                step > 0 and trainer.optim_general.cache_buffer_freq > 0
                and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")

        if step % cfg.my_config.fake_gt_add_freq == 0 and step > 0:
            if not do_save:
                trainer.save_checkpoint(
                    log_dir=cfg.log_dir,
                    save_only_model=True,
                    is_final=False,
                )
            render_dir = os.path.join(cfg.log_dir, f"render_{step:05d}")
            pred_dir = os.path.join(cfg.log_dir, f"pred_{step:05d}")
            width, height = 960, 640
            image_info_list, cam_info_list = [], []
            with torch.no_grad():
                os.makedirs(render_dir, exist_ok=True)
                for file in os.listdir(render_dir):
                    if file.endswith(".png"):
                        os.remove(os.path.join(render_dir, file))

                os.makedirs(pred_dir, exist_ok=True)
                for file in os.listdir(pred_dir):
                    if file.endswith(".png"):
                        os.remove(os.path.join(pred_dir, file))

                cam2worlds, intrinsics, norm_times, step_times, depth_maps = \
                    get_fake_gt_samples(
                        dataset,
                        num_points=cfg.my_config.num_sample,
                        min_coverage=0.6,
                        max_coverage=0.8,
                        cam_width=width,
                        cam_height=height,
                        radius=6,
                        grid_size=0.5,
                        angle_resolution=36,
                        farthest_sample=False
                    )

                for idx in range(len(cam2worlds)):
                    c2w = cam2worlds[idx]
                    intrinsic = intrinsics[idx]
                    depth_map = depth_maps[idx]
                    step_time = step_times[idx]
                    norm_time = norm_times[idx]

                    cam_info = {
                        "camera_to_world": c2w.to(device),
                        "intrinsics": intrinsic.to(device),
                        "height": torch.tensor(height, dtype=torch.long, device=device),
                        "width": torch.tensor(width, dtype=torch.long, device=device),
                    }

                    x, y = torch.meshgrid(
                        torch.arange(width),
                        torch.arange(height),
                        indexing="xy",
                    )
                    x, y = x.flatten(), y.flatten()
                    x, y = x.to(device), y.to(device)

                    pixel_coords = (
                        torch.stack([y / height, x / width], dim=-1)
                        .float()
                        .reshape(height, width, 2)
                    )
                    from datasets.base.pixel_source import get_rays

                    intrinsic = intrinsic * dataset.pixel_source.downscale_factor
                    intrinsic[2, 2] = 1.0
                    intrinsic = intrinsic.to(device)
                    c2w = c2w.to(device)
                    origins, viewdirs, direction_norm = get_rays(x, y, c2w, intrinsic)

                    viewdirs = viewdirs.reshape(height, width, 3)

                    image_id = torch.full(
                        (height, width),
                        0,
                        dtype=torch.long,
                    )

                    normalized_time = torch.full(
                        (height, width),
                        norm_time,
                        dtype=torch.float32,
                    )

                    image_info = {
                        "origins": origins.to(device),
                        "direction_norm": direction_norm.to(device),
                        "viewdirs": viewdirs.to(device),
                        "img_idx": image_id.to(device),
                        "pixel_coords": pixel_coords.to(device),
                        "normed_time": normalized_time.to(device),
                        "depth_map": depth_map.to(device),

                    }

                    output = trainer(image_info, cam_info, False)

                    sky_mask = output["opacity"].cpu().detach()
                    sky_mask = sky_mask.reshape(height, width)
                    sky_mask = (sky_mask > 0.5).float()

                    image_info["sky_masks"] = sky_mask.to(device)

                    img = to8b(output["rgb"])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    save_path = os.path.join(render_dir, f"{idx:03d}.png")
                    cv2.imwrite(save_path, img)

                    image_info_list.append(image_info)
                    cam_info_list.append(cam_info)

            model = OSEDiffInfer()
            model.infer(render_dir, pred_dir)
            model.clear_model()

            for idx in range(len(cam2worlds)):
                render_img_path = os.path.join(render_dir, f"{idx:03d}.png")
                pred_img_path = os.path.join(pred_dir, f"{idx:03d}.png")

                if not (os.path.exists(render_img_path) and os.path.exists(pred_img_path)):
                    print(pred_img_path)
                    continue

                pred_img = cv2.imread(pred_img_path)
                pred_img = cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR)
                pred_img = torch.from_numpy(pred_img).float() / 255.0

                image_info_list[idx]["pixels"] = pred_img.to(device)

            to_delete = [idx for idx, image_info in enumerate(image_info_list) if "pixels" not in image_info]
            for idx in reversed(to_delete):
                del image_info_list[idx]
                del cam_info_list[idx]

            torch.cuda.empty_cache()
            dataset.load_fake_gt(image_info_list, cam_info_list, True)

    logger.info("Training done!")

    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)

    return step


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)

    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")

    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str,
                        help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")

    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    final_step = main(args)
