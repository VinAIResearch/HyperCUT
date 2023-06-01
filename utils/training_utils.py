import argparse
import logging
import os
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from resources.consts import IMG_NORMALIZE_MEAN, IMG_NORMALIZE_STD


def get_optimizer(my_model, args):
    for k, v in my_model.named_parameters():
        if not v.requires_grad:
            print(f"Warning: {k} will not be optimized")

    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == "SGD":
        optimizer_function = optim.SGD
        kwargs = {"momentum": 0.9}
    elif args.optimizer == "ADAM":
        optimizer_function = optim.Adam
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    elif args.optimizer == "ADAMax":
        optimizer_function = optim.Adamax
        kwargs = {"betas": (0.9, 0.999), "eps": 1e-08}
    elif args.optimizer == "RMSprop":
        optimizer_function = optim.RMSprop
        kwargs = {"eps": 1e-08}

    kwargs["lr"] = args.lr
    kwargs["weight_decay"] = args.weight_decay

    logger = logging.getLogger("base")
    logger.info(f"Optimizer function: {args.optimizer}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")

    return optimizer_function(trainable, **kwargs)


def get_scheduler(my_optimizer, args):
    if args.decay_type == "step":
        scheduler = lrs.StepLR(my_optimizer, step_size=args.lr_decay, gamma=args.gamma)
    elif args.decay_type.find("step") >= 0:
        milestones = args.decay_type.split("_")
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(my_optimizer, milestones=milestones, gamma=args.gamma)

    logger = logging.getLogger("base")
    logger.info(f"Decay type: {args.decay_type}")

    return scheduler


def get_basic_parser():
    parser = argparse.ArgumentParser(description="Compression-Driven Frame Interpolation Training")

    # parameters
    # Directory Setting
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--metadata_root", type=str, default=None, required=True)

    # Learning Options
    parser.add_argument("--num_epochs", type=int, default=100, help="Max Epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--img_log_freq", type=int, help="saving image frequency")
    parser.add_argument("--train_limit", type=int, default=None, help="Maximum number of training data point")
    parser.add_argument("--val_limit", type=int, default=None, help="Maximum number of val data point")
    parser.add_argument("--test_limit", type=int, default=None, help="Maximum number of test data point")

    # Optimization specifications
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--lr_decay", type=int, default=3, help="learning rate decay per N epochs")
    parser.add_argument("--decay_type", type=str, default="step", help="learning rate decay type")
    parser.add_argument("--gamma", type=float, default=0.9, help="learning rate decay factor for step decay")
    parser.add_argument(
        "--optimizer",
        default="ADAMax",
        choices=("SGD", "ADAM", "RMSprop", "ADAMax"),
        help="optimizer to use (SGD | ADAM | RMSprop | ADAMax)",
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--load_epoch", type=int, default=-1, help="Load checkpoint from a specific epoch")
    parser.add_argument("--use_parallel", type=bool, default=True, help="Use parallel training")
    parser.add_argument("--image_size", type=int, help="learning rate decay factor for step decay")
    parser.add_argument("--pretrained_path", type=str, help="pretrained path")
    parser.add_argument("--hypercut_path", type=str, default="None", help="HyperCUT pretrained path")

    return parser


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s", datefmt="%y-%m-%d %H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}.log".format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")


def moduleNormalize(frame):
    return torch.cat(
        [(frame[:, 0:1, :, :] - 0.4631), (frame[:, 1:2, :, :] - 0.4352), (frame[:, 2:3, :, :] - 0.3990)], 1
    )


def CharbonnierFunc(data, epsilon=0.001):
    shape = data.shape
    return torch.mean(torch.sqrt(data**2 + epsilon**2), dim=shape[1:])


def unnormalize(img, max_pixel_value=255.0):
    # img: HxWxC

    mean, std = np.array(IMG_NORMALIZE_MEAN), np.array(IMG_NORMALIZE_STD)

    return img * (std * max_pixel_value) + mean * max_pixel_value


def unnormalize_tensor(tensor, max_pixel_value=255.0):
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    B, _, _, _ = tensor.shape

    mean, std = torch.Tensor(IMG_NORMALIZE_MEAN).to(tensor.device), torch.tensor(IMG_NORMALIZE_STD).to(tensor.device)
    mean = mean[None, :, None, None].repeat(B, 1, 1, 1)
    std = std[None, :, None, None].repeat(B, 1, 1, 1)

    return tensor * (std * max_pixel_value) + mean * max_pixel_value


def unnormalize_vid_tensor(vid_tensor, max_pixel_value=255.0):
    B, N, C, H, W = vid_tensor.shape
    vid_tensor = vid_tensor.view(-1, C, H, W)
    vid_tensor = unnormalize_tensor(vid_tensor).view(B, N, C, H, W)

    return vid_tensor


def tensor2img(tensor):
    img_np = tensor.cpu().detach().numpy()
    img_np = np.transpose(img_np, (1, 2, 0))

    img_np = np.clip(unnormalize(img_np), a_min=0, a_max=255).astype(np.uint8).copy()

    return img_np


def get_visualization(task, pred, dataPoint):
    if task == "classification":
        B, C, H, W = dataPoint["input"].shape
        r = np.random.randint(B)
        chosen_img = tensor2img(dataPoint["input"][r])
        return chosen_img
    elif task == "keypoint_detection":
        B, C, H, W = dataPoint["input"].shape
        r = np.random.randint(B)
        img = tensor2img(dataPoint["input"][r])
        gt_keypoints = dataPoint["gt"][r]
        pred_keypoints = pred[r].detach().cpu()

        gt_keypoints = [
            (int(gt_keypoints[2 * i].item()), int(gt_keypoints[2 * i + 1].item()))
            for i in range(0, gt_keypoints.shape[0] // 2)
        ]
        pred_keypoints = [
            (int(pred_keypoints[2 * i].item()), int(pred_keypoints[2 * i + 1].item()))
            for i in range(0, pred_keypoints.shape[0] // 2)
        ]

        for kp in gt_keypoints:
            cv2.circle(img, kp, 1, (255, 0, 0), -1)
        for kp in pred_keypoints:
            cv2.circle(img, kp, 1, (0, 255, 0), -1)

        return img
    else:
        raise NotImplementedError(f"Unrecognize {task}")
