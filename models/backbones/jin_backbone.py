import torch
from models.backbones.jin_et_al.center_esti_model import CenterEsti
from models.backbones.jin_et_al.f35_n8_model import F35_N8
from models.backbones.jin_et_al.f26_n9_model import F26_N9
from models.backbones.jin_et_al.f17_n9_model import F17_N9
from models.backbones.blur2vid_backbone_base import Blur2VidBackboneBase
from models.backbones.blur2vid_backbone_base import OrderInvariantBasedLoss
from models.backbones.blur2vid_backbone_base import HyperCUTBasedLoss
from models.backbones.blur2vid_backbone_base import NaiveBasedLoss


class Jin(Blur2VidBackboneBase):
    def __init__(self, **backbone_kwargs):
        super().__init__(**backbone_kwargs)

        self.f4 = CenterEsti()
        self.f35 = F35_N8()
        self.f26 = F26_N9()
        self.f17 = F17_N9()

    def forward(self, data):
        recon_frames = None
        blur_img = data['blur_img'].cuda()

        pred_f4 = self.f4(blur_img)
        pred_f3, pred_f5 = self.f35(blur_img, pred_f4)
        pred_f2, pred_f6 = self.f26(blur_img, pred_f3, pred_f4, pred_f5)
        pred_f1, pred_f7 = self.f17(blur_img, pred_f2, pred_f3, pred_f5, pred_f6)

        recon_frames = [pred_f1, pred_f2, pred_f3, pred_f4, pred_f5, pred_f6, pred_f7]
        recon_frames = torch.stack(recon_frames, dim=1)

        return {
            'recon_frames': recon_frames,
        }

    def get_order_inv_loss(self, **loss_kwargs):
        return OrderInvariantBasedLoss(**loss_kwargs)

    def get_hypercut_loss(self, **loss_kwargs):
        return JinHyperCUTLoss(**loss_kwargs)

    def get_naive_loss(self, **loss_kwargs):
        return NaiveBasedLoss(**loss_kwargs)


class JinHyperCUTLoss(HyperCUTBasedLoss):
    def __init__(self, **loss_kwargs):
        super().__init__(**loss_kwargs)
        pass

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        pred = out['recon_frames']

        char_loss = 0
        loss_acc = self.calc_hypercut_loss(pred, task)
        hypercut_loss = loss_acc['loss'].mean()
        acc = loss_acc['acc']

        B, N, C, H, W = gt.shape

        hypercut_gt = self.calc_hypercut_loss(gt, task)['loss']
        for i in range(B):
            if hypercut_gt[i] > 0:
                gt[i] = torch.flip(gt[i], dims=[0])

        order_inv_loss = 0
        for i in range(N // 2):
            order_inv_loss += self.order_inv_loss(
                pred[:, i, :, :, :],
                pred[:, N - i - 1, :, :, :],
                gt[:, i, :, :, :],
                gt[:, N - i - 1, :, :, :],
            )['loss'] / N
        

        total = order_inv_loss + self.alpha * hypercut_loss

        return {
            'order_inv_loss': order_inv_loss,
            'hypercut_pred': hypercut_loss,
            'hypercut_gt': hypercut_gt.mean(),
            'total': total,
            'acc': acc
        }

