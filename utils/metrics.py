import numpy as np
import torch
from utils.training_utils import unnormalize_vid_tensor


def calculate_PSNR(pred, gt):
    if pred.shape[0] > 1 or gt.shape[0] > 1:
        raise NotImplementedError("Not supported batch validation yet")

    pred = unnormalize_vid_tensor(pred)
    gt = unnormalize_vid_tensor(gt)

    mse = torch.mean((pred - gt) ** 2, dim=(2, 3, 4))
    res = (20 * torch.log10(255.0 / torch.sqrt(mse))).detach().cpu().numpy()
    return res, np.mean(res)


def calculate_pPSNR(pred, gt):
    if pred.shape[0] > 1 or gt.shape[0] > 1:
        raise NotImplementedError("Not supported batch validation yet")

    p1, mean_p1 = calculate_PSNR(pred, gt)
    p2, mean_p2 = calculate_PSNR(torch.flip(pred, [1]), gt)

    if mean_p1 > mean_p2:
        return p1, mean_p1
    return p2, mean_p2
