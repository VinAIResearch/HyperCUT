import torch
import numpy as np
from torch import nn
from models import get_backbone
from resources.consts import SEED
from utils.metrics import calculate_PSNR, calculate_pPSNR
import logging
from tqdm import tqdm
import os
import seaborn as sns
import os.path as osp
import cv2
from utils.training_utils import tensor2img


torch.manual_seed(SEED)


class Blur2Vid(nn.Module):
    '''
        f(xk, xmid) = f(g(x0, xmid))
    '''
    def __init__(self, **kwargs):
        super().__init__()

        self.logger = logging.getLogger('base')

        blur2vid_model = kwargs['backbone_name']
        self.model = get_backbone(blur2vid_model, kwargs['backbone_kwargs'])

    def forward(self, data):
        return self.model(data)

    def calc_loss(self, out, data, task='val'):
        return self.model.calc_loss(out, data, task)

    def validation(self, dataloader, display_step, limit, save_root):
        total_loss, total_samples, hits, consist, twice_consist = 0, 0, 0, 0, 0
        self.eval()

        metrics = {'PSNR': calculate_PSNR, 'pPSNR': calculate_pPSNR, 'PSNR_mean': 0, 'pPSNR_mean': 0}
        metric_scores = {x: 0 for x in metrics}

        with torch.no_grad():
            for data in tqdm(dataloader, total=len(dataloader)):
                data['gt'] = data['gt'].cuda()
                out = self(data)
                recon_frames = out['recon_frames']
                loss = self.calc_loss(out, data, 'val')
                try: 
                    total_loss += loss['total'].cpu().numpy()
                except:
                    total_loss += loss['total']
                    
                hits += loss['acc']['hit']
                consist += loss['acc']['con@2']
                twice_consist += loss['acc']['con@3']


                total_samples += 1
                assert data['gt'].shape[0] == 1

                for metric_name, metric_func in metrics.items():
                    if 'mean' in metric_name: continue
                    metric_arr, metric_mean = metric_func(recon_frames, data['gt'])
                    metric_scores[metric_name] += metric_arr
                    metric_scores[metric_name + "_mean"] += metric_mean

                if total_samples % display_step == 0:
                    _, N, _, _, _ = recon_frames.shape
                    img_blur = tensor2img(data['blur_img'][0])
                    dst = f'{save_root}/{total_samples:03d}/'
                    os.makedirs(osp.dirname(dst), exist_ok=True)
                    self.logger.info("SAVING VISUALIZATION")
                    self.logger.info(save_root)
                    for i in range(N):
                        img_pred = tensor2img(recon_frames[0, i, :, :, :])
                        img_gt = tensor2img(data['gt'][0, i, :, :, :])
                        
                        
                        dst = f'{save_root}/{total_samples:03d}/{i}.png'
                        cv2.imwrite(dst, np.hstack((img_blur, img_pred, img_gt)))
                        
                if limit is not None and total_samples >= limit:
                    break

        total_loss /= total_samples
        metric_scores = {x: np.around(y / total_samples, 2) for x, y in metric_scores.items()}

        metric_scores['hit'] = hits / total_samples
        metric_scores['con@2'] =  consist / total_samples
        metric_scores['con@3'] = twice_consist / total_samples

        self.train()

        metric_scores['loss'] = total_loss

        return metric_scores
