from torch import nn


N_BLOCKS = 6
PADDING_TYPE = "reflect"


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding
                                   layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer,
                              and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                        is not implemented"
            )

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError(
                f"padding {padding_type} \
                                      is not implemented"
            )
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetIm2Vec(nn.Module):
    def __init__(self, in_dim, out_dim, nf=32):
        super().__init__()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_dim, nf, kernel_size=7, padding=0, bias=True),
            nn.LeakyReLU(True),
        ]

        n_downsampling = 5
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            inc = min(nf * mult, out_dim)
            if i == 0:
                inc = nf * mult
            ouc = min(nf * mult * 2, out_dim)
            model += [
                nn.Conv2d(inc, ouc, kernel_size=3, stride=2, padding=1, bias=True),
                nn.LeakyReLU(True),
            ]

        for i in range(N_BLOCKS):
            model += [
                ResnetBlock(
                    out_dim,
                    padding_type=PADDING_TYPE,
                    use_dropout=False,
                    use_bias=True,
                )
            ]

        self.model = nn.Sequential(*model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # breakpoint()
        feat = self.model(x)
        # for ind, layer in enumerate(self.model):
        # x = layer(x)
        # feat = x
        feat = self.avgpool(feat)
        feat = feat.squeeze(3).squeeze(2)

        return feat
