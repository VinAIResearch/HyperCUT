import torch
from torch import nn
import numpy as np
from models.hypercut_model import HyperCUT


class OrderInvariantLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss()

    def forward(self, x1, x2, y1, y2):
        acc = {
            'hit': 0,
            'con@2': 0,
            'con@3': 0
        }
        return {'loss': self.mse(torch.abs(x1 + x2), torch.abs(y1 + y2)) + self.mse(torch.abs(x1 - x2), torch.abs(y1 - y2)),
                'acc': acc}


class OrderInvariantBasedLoss(nn.Module):
    def __init__(self, **loss_kwargs):
        super().__init__()

        self.hypercut = HyperCUT(**loss_kwargs['HyperCUT'])
        self.order_inv_loss = OrderInvariantLoss()

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        pred = out['recon_frames']

        num_frames = gt.shape[1]
        order_inv_loss = 0

        for i in range(num_frames // 2):
            loss_acc = self.order_inv_loss(
                pred[:, i, :, :, :],
                pred[:, num_frames - i - 1, :, :, :],
                gt[:, i, :, :, :],
                gt[:, num_frames - i - 1, :, :, :],
            )
            order_inv_loss += loss_acc['loss'] / num_frames
            acc = loss_acc['acc']
        
        return {'total': order_inv_loss,
                'acc': acc}


class NaiveBasedLoss(nn.Module):
    def __init__(self, epsilon=0.001, **loss_kwargs):
        super().__init__()
        self.epsilon = epsilon
        self.hypercut = HyperCUT(**loss_kwargs['HyperCUT'])

    def forward(self, out, data, task):
        gt = data['gt'].cuda()
        pred = out['recon_frames']

        num_frames = gt.shape[1]
        char_loss = 0

        for i in range(num_frames):
            ith_pred = pred[:, i, :, :, :]
            ith_gt = gt[:, i, :, :, :]
            char_loss += torch.mean(torch.sqrt((ith_pred - ith_gt) ** 2 + self.epsilon ** 2)) / num_frames

        return {'total': char_loss}


class HyperCUTBasedLoss(nn.Module):
    def __init__(self, epsilon=0.001, **loss_kwargs):
        super().__init__()

        self.epsilon = epsilon
        self.alpha = loss_kwargs['HyperCUT']['alpha']

        self.hypercut = HyperCUT(**loss_kwargs['HyperCUT'])
        
        print('HyperCUT pretrained = ',loss_kwargs['HyperCUT']['pretrained_path'])
        self.hypercut.load_state_dict(
            torch.load(loss_kwargs['HyperCUT']['pretrained_path'])
        )

        self.softplus = nn.Softplus()

        for p in self.hypercut.parameters():
            p.requires_grad = False

        self.order_inv_loss = OrderInvariantLoss()

    def calc_hypercut_loss(self, x, task):
        # import pdb; pdb.set_trace()
        x = [a.squeeze(1) for a in torch.split(x, 1, 1)]
        first_pair = [x[0], x[-1]]
        second_pair = [x[1], x[-2]]
        third_pair = [x[2], x[-3]]

        ##### LOSS #####

        d00 = self.hypercut.f_func(self.hypercut.g_func(first_pair))
        d01 = self.hypercut.f_func(self.hypercut.g_func(list(reversed(first_pair))))

        d10 = self.hypercut.f_func(self.hypercut.g_func(second_pair))
        d11 = self.hypercut.f_func(self.hypercut.g_func(list(reversed(second_pair))))

        d20 = self.hypercut.f_func(self.hypercut.g_func(third_pair))
        d21 = self.hypercut.f_func(self.hypercut.g_func(list(reversed(third_pair))))
        
        
        loss = (self.softplus(self.hypercut.hyperplane(d00))
                + self.softplus(self.hypercut.hyperplane(d10))
                + self.softplus(self.hypercut.hyperplane(d20)))

        ##### ACCURACY #####
        hits, consist, twice_consist = 0, 0, 0
        
        if task == "val":
            u0 = self.hypercut.hyperplane(d00).item()
            v0 = self.hypercut.hyperplane(d01).item()
            
            if (u0 * v0) < 0:
                hits += 1

            u1 = self.hypercut.hyperplane(d10).item()
            u2 = self.hypercut.hyperplane(d20).item()

            if np.sign(u0) == np.sign(u1):
                consist += 1
            if np.sign(u0) == np.sign(u1) and np.sign(u1) == np.sign(u2):
                twice_consist += 1
        
        acc = {
            'hit': hits,
            'con@2': consist,
            'con@3': twice_consist
        }

        return {
            'loss': loss,
            'acc': acc
        }

        
    def forward(self, out, data):
        gt = data['gt'].cuda()
        pred = out['recon_frames']

        char_loss = 0
        hypercut_loss = self.calc_hypercut_loss(pred).mean()

        B, N, C, H, W = gt.shape

        hypercut_gt = self.calc_hypercut_loss(gt)
        for i in range(B):
            if hypercut_gt[i] > 0:
                gt[i] = torch.flip(gt[i], dims=[0])

        for i in range(N):
            ith_pred = pred[:, i, :, :, :]
            ith_gt = gt[:, i, :, :, :]
            char_loss += torch.mean(torch.sqrt((ith_pred - ith_gt) ** 2 + self.epsilon ** 2)) / N

        total = char_loss + self.alpha * hypercut_loss

        return {
            'char_loss': char_loss.item(),
            'hypercut_pred': hypercut_loss.item(),
            'hypercut_gt': hypercut_gt.mean().item(),
            'total': total
        }


class Blur2VidBackboneBase(nn.Module):
    def __init__(self, **backbone_kwargs):
        super().__init__()

        loss_kwargs = backbone_kwargs['loss_kwargs']
        loss_type = loss_kwargs['loss_type']
        print("Loss Type:", loss_type)

        if loss_type == 'order_inv':
            self.loss = self.get_order_inv_loss(**loss_kwargs)
        elif loss_type == 'hypercut':
            self.loss = self.get_hypercut_loss(**loss_kwargs)
        elif loss_type == 'naive':
            self.loss = self.get_naive_loss(**loss_kwargs)
        else:
            raise NotImplementedError(f'Unrecognized loss {loss_type}')

    def forward(self, data):
        pass

    def calc_loss(self, out, data, task='val'):
        return self.loss(out, data, task)
