import logging

import numpy as np
import torch
from models import get_backbone
from resources.consts import SEED
from torch import nn
from tqdm import tqdm


torch.manual_seed(SEED)


class HyperPlane(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

        self.hyperplane = nn.Parameter(torch.rand(out_dim + 1).cuda() * 2 - 1)
        self.hyperplane.requires_grad = False

    def project(self, x):
        B, C = x.shape
        coeff = self.hyperplane[:-1].unsqueeze(0).repeat(B, 1)
        res = x - self(x) / (torch.norm(self.hyperplane) ** 2) * coeff

        return res

    def forward(self, d):
        B, C = d.shape

        bias = self.hyperplane[C:]
        coeff = self.hyperplane[:C].unsqueeze(0).repeat(B, 1)

        res = torch.bmm(d.view(B, 1, C), coeff.view(B, C, 1) + bias).squeeze(1)

        return res


class HyperCUT(nn.Module):
    """
    f(xk, xmid) = f(g(x0, xmid))
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.logger = logging.getLogger("base")

        num_frames = kwargs["num_frames"]
        g_func_name = kwargs["g_func"]
        self.g_func = get_backbone(g_func_name, dict(num_frames=num_frames))
        # self.logger.info(self.g_func)

        in_dim = self.g_func.out_dim
        f_func_name = kwargs["f_func"]
        out_dim = kwargs["out_dim"]
        self.f_func = get_backbone(f_func_name, dict(in_dim=in_dim, out_dim=out_dim))
        # self.logger.info(self.f_func)

        self.hyperplane = HyperPlane(out_dim)

        self.softplus = nn.Softplus()

    def forward(self, data):
        sharp_frames = [x.cuda() for x in data["sharp_frames"]]
        first_pair = [sharp_frames[0], sharp_frames[-1]]
        second_pair = [sharp_frames[1], sharp_frames[-2]]
        third_pair = [sharp_frames[2], sharp_frames[-3]]

        d00 = self.f_func(self.g_func(first_pair))
        d01 = self.f_func(self.g_func(list(reversed(first_pair))))

        d10 = self.f_func(self.g_func(second_pair))
        d11 = self.f_func(self.g_func(list(reversed(second_pair))))

        d20 = self.f_func(self.g_func(third_pair))
        d21 = self.f_func(self.g_func(list(reversed(third_pair))))

        return d00, d01, d10, d11, d20, d21

    def calc_loss(self, out, data, task):
        d00, d01, d10, d11, d20, d21 = out

        diff_loss1 = self.softplus(self.hyperplane(d00) * self.hyperplane(d01)).mean()
        diff_loss2 = self.softplus(self.hyperplane(d10) * self.hyperplane(d11)).mean()
        diff_loss3 = self.softplus(self.hyperplane(d20) * self.hyperplane(d21)).mean()

        same_loss1 = self.softplus(-self.hyperplane(d00) * self.hyperplane(d10)).mean()
        same_loss2 = self.softplus(-self.hyperplane(d10) * self.hyperplane(d20)).mean()
        same_loss3 = self.softplus(-self.hyperplane(d00) * self.hyperplane(d20)).mean()

        diff_loss = 1 / 3.0 * (diff_loss1 + diff_loss2 + diff_loss3)
        same_loss = 1 / 3.0 * (same_loss1 + same_loss2 + same_loss3)
        total_loss = 0.5 * diff_loss + 0.5 * same_loss

        return {"total": total_loss, "diff": diff_loss, "same_loss": same_loss}

    def validation(self, dataloader, display_step, limit, save_root):
        total_loss, total_samples, hits, consist, twice_consist = 0, 0, 0, 0, 0
        self.eval()

        with torch.no_grad():
            for data in tqdm(dataloader, total=len(dataloader)):
                out = self(data)
                total_loss += self.calc_loss(out, None, None)["total"].item()
                total_samples += 1
                d00, d01, d10, d11, d20, d21 = out
                assert d00.shape[0] == 1

                u0 = self.hyperplane(d00).item()
                v0 = self.hyperplane(d01).item()
                if (u0 * v0) < 0:
                    hits += 1

                u1 = self.hyperplane(d10).item()
                u2 = self.hyperplane(d20).item()

                if np.sign(u0) == np.sign(u1):
                    consist += 1
                if np.sign(u0) == np.sign(u1) and np.sign(u1) == np.sign(u2):
                    twice_consist += 1

                if limit is not None and total_samples >= limit:
                    break

        total_loss /= total_samples

        self.train()

        return {
            "softplus": total_loss,
            "Hit": hits / total_samples,
            "Con@2": consist / total_samples,
            "Con@3": twice_consist / total_samples,
        }

    def test(self, dataloader):
        total_samples, consist, twice_consist = 0, 0, 0
        hits_06, hits_15, hits_24 = 0, 0, 0
        self.eval()

        def proc(x):
            if x >= 0:
                return 1
            return -1

        with torch.no_grad():
            for data in tqdm(dataloader):
                out = self(data)
                d00, d01, d10, d11, d20, d21 = out
                assert d00.shape[0] == 1

                u0 = self.hyperplane(d00).item()
                v0 = self.hyperplane(d01).item()
                if (u0 * v0) < 0:
                    hits_06 += 1
                total_samples += 1

                u1 = self.hyperplane(d10).item()
                v1 = self.hyperplane(d11).item()
                if (u1 * v1) < 0:
                    hits_15 += 1

                u2 = self.hyperplane(d20).item()
                v2 = self.hyperplane(d21).item()
                if (u2 * v2) < 0:
                    hits_24 += 1

                if np.sign(u0) == np.sign(u1):
                    consist += 1
                if np.sign(u0) == np.sign(u1) and np.sign(u1) == np.sign(u2):
                    twice_consist += 1

        print(f"Hits x0, x6 ratio: {hits_06 / total_samples}")
        print(f"Hits x1, x5 ratio: {hits_15 / total_samples}")
        print(f"Hits x2, x3 ratio: {hits_24 / total_samples}")
        print(f"(Con@2) x0, x1 consistency: {consist / total_samples}")
        print(f"(Con@3) x0, x1, x2 consistency: {twice_consist / total_samples}")

        self.train()
