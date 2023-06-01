import torch
from torch.nn import functional as F
from models.backbones.purohit_et_al.main_components import BIE, RVD
from models.backbones.blur2vid_backbone_base import Blur2VidBackboneBase
from models.backbones.blur2vid_backbone_base import OrderInvariantBasedLoss
from models.backbones.blur2vid_backbone_base import HyperCUTBasedLoss
from models.backbones.blur2vid_backbone_base import NaiveBasedLoss


class Purohit(Blur2VidBackboneBase):
    def __init__(self, **backbone_kwargs):
        super().__init__(**backbone_kwargs)

        self.bie = BIE()
        self.rvd = RVD()

        state_dict = torch.load(backbone_kwargs['stage1_path'])
        state_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith('rvd.')}

        self.rvd.load_state_dict(state_dict)

    def forward(self, data):
        sharp_frames = data['gt']
        
        sharp_frames = sharp_frames.cuda()
        _, num_frames, _, _, _ = sharp_frames.shape

        h = self.bie(data['blur_img'].cuda(), sharp_frames[:, num_frames // 2, :, :, :])
        recon_frames, pred_flows = self.rvd(h, sharp_frames[:, num_frames // 2, :, :, :], num_frames)

        return {
            'recon_frames': recon_frames[0],
            'multiscale_recon_frames': recon_frames,
            'pred_flows': pred_flows
        }

    def get_order_inv_loss(self, **loss_kwargs):
        return MultiscaleOrderInvLoss(**loss_kwargs)

    def get_hypercut_loss(self, **loss_kwargs):
        return MultiscaleHyperCUTLoss(**loss_kwargs)

    def get_naive_loss(self, **loss_kwargs):
        return MultiscaleCharbonnierLoss(**loss_kwargs)

class MultiscaleOrderInvLoss(OrderInvariantBasedLoss):
    def __init__(self, **loss_kwargs):
        super().__init__(**loss_kwargs)
        self.mu = loss_kwargs['mu']

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        recon_frames = out['multiscale_recon_frames']
        pred_flow = out['pred_flows']

        B, num_frames, _, _, _ = recon_frames[0].shape

        num_scales = len(recon_frames)
        order_inv_loss = 0
        tv_reg = 0
        for scale in range(num_scales):
            pred = recon_frames[scale]

            for i in range(num_frames // 2):
                loss_acc = self.order_inv_loss(
                    pred[:, i, :, :, :],
                    pred[:, num_frames - i - 1, :, :, :],
                    gt[:, i, :, :, :],
                    gt[:, num_frames - i - 1, :, :, :],
                )
                order_inv_loss +=  loss_acc['loss'] / num_scales
                acc = loss_acc['acc']

            gt = torch.cat([F.interpolate(gt[:, i, :, :, :], scale_factor=0.5).unsqueeze(1) for i in range(num_frames)], dim=1)

        for flow in pred_flow:
            tv_reg += torch.abs(flow).mean() / num_scales

        total = order_inv_loss + self.mu * tv_reg

        loss = {
            'order_inv_loss': order_inv_loss.item(),
            'tv_reg': tv_reg.item(),
            'total': total,
            'acc': acc
        }

        return loss


class MultiscaleCharbonnierLoss(NaiveBasedLoss):
    def __init__(self, epsilon=0.001, **loss_kwargs):
        super().__init__(epsilon, **loss_kwargs)
        self.mu = loss_kwargs['mu']

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        recon_frames = out['multiscale_recon_frames']
        pred_flow = out['pred_flows']

        B, num_frames, _, _, _ = recon_frames[0].shape

        num_scales = len(recon_frames)
        char_loss = 0
        tv_reg = 0

        for scale in range(num_scales):
            pred = recon_frames[scale]

            for i in range(num_frames):
                ith_pred = pred[:, i, :, :, :]
                ith_gt = gt[:, i, :, :, :]
                char_loss += torch.mean(torch.sqrt((ith_pred - ith_gt) ** 2 + self.epsilon ** 2)) / num_scales

            gt = torch.cat([F.interpolate(gt[:, i, :, :, :], scale_factor=0.5).unsqueeze(1) for i in range(num_frames)], dim=1)

        for flow in pred_flow:
            tv_reg += torch.abs(flow).mean() / num_scales

        total = char_loss + self.mu * tv_reg

        loss = {
            'char_loss': char_loss.item(),
            'tv_reg': tv_reg.item(),
            'total': total
        }

        return loss


class MultiscaleHyperCUTLoss(HyperCUTBasedLoss):
    def __init__(self, epsilon=0.001, **loss_kwargs):
        super().__init__(epsilon, **loss_kwargs)

        self.mu = loss_kwargs['mu']

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        recon_frames = out['multiscale_recon_frames']
        pred_flow = out['pred_flows']

        B, num_frames, _, _, _ = recon_frames[0].shape

        num_scales = len(recon_frames)
        char_loss = 0
        hypercut_loss = 0
        order_inv_loss = 0
        tv_reg = 0
        acc = 0

        B, N, C, H, W = gt.shape

        for scale in range(num_scales):
            pred = recon_frames[scale]

            if scale == 0:
                loss_acc = self.calc_hypercut_loss(pred, task)
                hypercut_loss += loss_acc['loss'].mean()
                acc = loss_acc['acc']

            for i in range(num_frames // 2):
                loss_acc = self.order_inv_loss(
                    pred[:, i, :, :, :],
                    pred[:, num_frames - i - 1, :, :, :],
                    gt[:, i, :, :, :],
                    gt[:, num_frames - i - 1, :, :, :],
                )
                order_inv_loss +=  loss_acc['loss'] / num_scales
                # acc = loss_acc['acc']

            # for i in range(num_frames):
            #     ith_pred = pred[:, i, :, :, :]
            #     ith_gt = gt[:, i, :, :, :]
            #     char_loss += torch.mean(torch.sqrt((ith_pred - ith_gt) ** 2 + self.epsilon ** 2)) / num_scales

            gt = torch.cat([F.interpolate(gt[:, i, :, :, :], scale_factor=0.5).unsqueeze(1) for i in range(num_frames)], dim=1)

        for flow in pred_flow:
            tv_reg += torch.abs(flow).mean() / num_scales

        total = order_inv_loss + self.mu * tv_reg + self.alpha * hypercut_loss
        loss = {
            'char_loss': order_inv_loss.item(),
            'tv_reg': tv_reg.item(),
            'total': total,
            'acc': acc
        }

        return loss

