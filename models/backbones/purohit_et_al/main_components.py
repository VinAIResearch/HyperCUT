import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        nf: int,
    ) -> None:
        super().__init__()

        self.conv1 = conv3x3(nf, nf)
        self.lrelu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(nf, nf)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(out)

        out += identity
        out = self.lrelu(out)

        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, h, w):
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=self.conv.weight.device),
        )


class Encoder(nn.Module):
    def __init__(self, inp_channels=3, immediate_channels=[16, 32, 64, 128], kernel_size=3):
        super(Encoder, self).__init__()

        assert kernel_size % 2 == 1

        module_list = []
        strides = [1] + [2] * len(immediate_channels)
        for x, y, stride in zip([inp_channels] + immediate_channels[:-1], immediate_channels, strides):
            module_list.append(
                nn.Sequential(
                    nn.Conv2d(x, y, kernel_size, stride=stride, padding=kernel_size // 2),
                    ResBlock(y),
                )
            )

        self.model = nn.ModuleList(module_list)

    def forward(self, x, return_immediate_features=False):
        fs = []
        for layer in self.model:
            x = layer(x)
            fs.append(x)

        if return_immediate_features:
            fs.reverse()
            return fs
        return x


class MyDeconv(nn.Module):
    def __init__(self, inp_nc, out_nc, kernel_size=5, scale_factor=2):
        super(MyDeconv, self).__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear")
        self.conv = nn.Conv2d(inp_nc, out_nc, kernel_size, padding=kernel_size // 2, bias=True)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.lrelu(x)

        return x


class FlowDecoder(nn.Module):
    def __init__(self, hidden_dim=128, immediate_channels=[16, 32, 64, 128], kernel_size=3):
        super(FlowDecoder, self).__init__()

        immediate_channels.reverse()
        self.num_scales = len(immediate_channels)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        self.deconvs = nn.ModuleList()
        for x, y in zip(immediate_channels[:-1], immediate_channels[1:]):
            self.deconvs.append(MyDeconv(x, y))

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(immediate_channels[0] * 2, 2, 3, padding=1, bias=True))
        for i in range(1, self.num_scales):
            self.convs.append(nn.Conv2d(immediate_channels[i] * 2 + 2, 2, 3, padding=1, bias=True))

    def forward(self, conv_lstm_out, encoder_features):
        inp = torch.cat((encoder_features[0], conv_lstm_out), dim=1)
        flows = [self.convs[0](inp)]

        for i in range(1, self.num_scales):
            flow_upsampled = self.upsample(flows[-1])
            conv_lstm_out = self.deconvs[i - 1](conv_lstm_out)
            inp = torch.cat((encoder_features[i], conv_lstm_out, flow_upsampled), dim=1)
            flows.append(self.convs[i](inp))

        return flows


class BIE(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(6, 64, 5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 5, stride=1, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, 5, stride=2, padding=2, bias=True),
            nn.LeakyReLU(0.1),
        )

    def forward(self, target_frame, middle_frame):
        return self.model(torch.cat((target_frame, middle_frame), dim=1))


class FrameWarper(nn.Module):
    def __init__(self):
        super(FrameWarper, self).__init__()

    def forward(sefl, x, flo):
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = grid + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).cuda()
        mask = F.grid_sample(mask, vgrid)

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask


class TransformerLayer(nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()

        self.warper = FrameWarper()

    def forward(self, frame, flows):
        warped_frames = []
        for flow in flows[::-1]:
            warped_frames.append(self.warper(frame, flow))
            frame = F.interpolate(frame, scale_factor=0.5)

        return warped_frames


class RVD(nn.Module):
    def __init__(self, immediate_channels=[16, 32, 164, 128]):
        super(RVD, self).__init__()

        self.num_scales = len(immediate_channels)
        self.flow_encoder = Encoder(inp_channels=2, immediate_channels=immediate_channels)
        self.conv_lstm = ConvLSTMCell(input_dim=128, hidden_dim=128, kernel_size=[3, 3], bias=True)
        self.flow_decoder = FlowDecoder(hidden_dim=128, immediate_channels=immediate_channels)
        self.transformer_layer = TransformerLayer()

    def forward(self, initial_hidden_state, middle_frame, num_frames):
        B, _, H, W = middle_frame.shape
        prev_flow = torch.zeros((B, 2, H, W)).cuda()

        flo = [None] * self.num_scales
        x = [None] * self.num_scales

        for i in range(num_frames):
            fs = self.flow_encoder(prev_flow, return_immediate_features=True)
            if i == 0:
                prev = initial_hidden_state, self.conv_lstm.init_hidden(B, fs[0].shape[2], fs[0].shape[3])[1]
            h, c = self.conv_lstm(fs[0], prev)
            flow_multiscale = self.flow_decoder(h, fs)
            x_multiscale = self.transformer_layer(middle_frame, flow_multiscale)
            for scale in range(self.num_scales):
                if x[scale] is None:
                    x[scale] = x_multiscale[scale].unsqueeze(1)
                else:
                    x[scale] = torch.cat((x[scale], x_multiscale[scale].unsqueeze(1)), dim=1)

                if flo[scale] is None:
                    flo[scale] = flow_multiscale[scale].unsqueeze(1)
                else:
                    flo[scale] = torch.cat((flo[scale], flow_multiscale[scale].unsqueeze(1)), dim=1)
            prev = (h, c)

        return x, flo
