import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, num_frames):
        super().__init__()
        self.out_dim = num_frames * 3

    def forward(self, frames):
        return torch.cat(frames, dim=1)
