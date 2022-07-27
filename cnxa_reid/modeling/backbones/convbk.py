import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def chunk(x, N, use_gpu=True):
    # fearture = list()

    if use_gpu:
        out = torch.Tensor().cuda()
    else:
        out = torch.Tensor()
    A = torch.chunk(x, N, dim=-1)
    for a in A:
        B = torch.chunk(a, N, dim=-2)
        for i in range(len(B)):
            a = B[i]
            # out = torch.cat([out, a], dim=1)
            if use_gpu:
                out = torch.cat([out, a], dim=1).cuda()
            else:
                out = torch.cat([out, a], dim=1)

    return out


class PAD(nn.Module):
    def __init__(self, use_gpu=True, **kwargs):
        super(PAD, self).__init__()
        self.use_gpu = use_gpu

    def forward(self, x):
        # _, _, w, h = self.inputs.size()
        _, _, h, w = x.size()

        if w % 2 == 0 and not (h % 2 == 0):
            # out = torch.ones_like(inputs)
            # print("w为偶h为奇!!!")
            pad_h = nn.ZeroPad2d((0, 0, 0, 1))
            x = pad_h(x).cuda() if self.use_gpu else pad_h(x)
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_h(x)
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_h(x)
            else:
                x = x

        elif not (w % 2 == 0) and h % 2 == 0:
            # print("w为奇h为偶!!!")
            pad_w = nn.ZeroPad2d((0, 1, 0, 0))
            x = pad_w(x).cuda() if self.use_gpu else pad_w(x)
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            else:
                x = x

        elif not (w % 2 == 0 and h % 2 == 0):
            # print("w为奇h为奇!!!")
            pad_w_h = nn.ZeroPad2d((0, 1, 0, 1))
            x = pad_w_h(x).cuda() if self.use_gpu else pad_w_h(x)
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            else:
                x = x

        else:
            # print("w为偶h为偶!!!")
            # _, _, h, w = inputs.size()
            # print(inputs.shape)
            p = abs((h - w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda() if self.use_gpu else pad_(x)
            else:
                x = x

        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BIGKS(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=15, stride=1, I=0, N=0, use_gpu=True):
        super(BIGKS, self).__init__()
        self.channel = channel
        self.N = N
        self.use_gpu = use_gpu
        self.pad = PAD(use_gpu=use_gpu)
        k1 = kernel_size
        p1 = (k1 - 1) // 2

        self.bk1_dwconv = nn.Conv2d(channel, channel, kernel_size=k1, padding=p1, groups=channel)

        self.pwconv2 = nn.Sequential(
            nn.Linear(channel, channel * (ratio // 2)),
            nn.GELU(),
            nn.Linear(channel * (ratio // 2), channel)
        )

        self.channel_gear = nn.Conv2d(channel // N, channel, kernel_size=1, stride=1, padding=0)
        self.bk2_dw3conv = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, groups=channel)
        self.bk2_dwDconv = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, stride=1, groups=channel, padding=2)
        self.bk2_dw1conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, groups=channel)

        self.norm = LayerNorm(channel, eps=1e-6)
        self.pwconv1 = nn.Sequential(
            nn.Linear(channel, channel * ratio),
            nn.GELU(),
            nn.Linear(channel * ratio, channel // N),
        )

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, True),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, True),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        inputs = x
        b, c, _, _ = x.size()

        x = self.pad(x)

        x_bk1 = self.bk1_dwconv(x)
        x_bk1 = x_bk1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_bk1 = self.pwconv1(x_bk1)

        x_bk1 = x_bk1.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x_bk1_c = chunk(x_bk1, self.N // 2, use_gpu=self.use_gpu)

        x_bk1 = self.channel_gear(x_bk1)

        x_bk2_1 = self.bk2_dw3conv(x_bk1)
        x_bk2_1 = x_bk2_1.permute(0, 2, 3, 1)
        x_bk2_1 = self.pwconv2(x_bk2_1)
        x_bk2_1 = x_bk2_1.permute(0, 3, 1, 2)

        x_bk2_2 = self.bk2_dwDconv(x_bk1)
        x_bk2_2 = x_bk2_2.permute(0, 2, 3, 1)
        x_bk2_2 = self.pwconv2(x_bk2_2)
        x_bk2_2 = x_bk2_2.permute(0, 3, 1, 2)

        x_bk2_3 = self.bk2_dw1conv(x_bk1)
        x_bk2_3 = x_bk2_3.permute(0, 2, 3, 1)
        x_bk2_3 = self.pwconv2(x_bk2_3)
        x_bk2_3 = x_bk2_3.permute(0, 3, 1, 2)

        x_bk2 = x_bk2_1 + x_bk2_2 + x_bk2_3

        x_bk1_c_f = self.max_pool(x_bk1_c).view([b, c])
        x_bk2_f = self.max_pool(x_bk2).view([b, c])

        x1 = self.fc(x_bk1_c_f).view([b, c, 1, 1])
        x2 = self.fc(x_bk2_f).view([b, c, 1, 1])

        y = x1 + x2

        return inputs * self.sigmoid(y)


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class convbk(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., use_gpu=True, in_channel=768
                 ):
        super().__init__()

        self.bk = BIGKS(channel=dims[-1], I=13, N=4, use_gpu=use_gpu)
        print("start train our bk_attention!")
        # self.BK = BIGKS(channel=dims[-1])
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
            
        # self.last_layer = nn.Sequential(
           #  nn.Conv2d(dims[-1], in_channel, kernel_size=1),
           #  nn.ReLU(),
           #  nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        # )
        self.last_layer = nn.Conv2d(dims[-1], in_channel, kernel_size=3, padding=1, stride=1)
        # self.last_layer = nn.Conv2d(dims[-1], in_channel, kernel_size=1)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        _, _, h, w = x.size()
        x = x + self.bk(x)
        x = self.last_layer(x)

        # x = self.conv2048(x)
        # x = self.norm(x.mean([-2, -1]))
        # x = self.head(x)
        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)["model"]
        for i in param_dict:
            if 'head' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def freeze_backbone(self):
        backbone = [self.downsample_layers, self.stages]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def unfreeze_backbone(self):
        backbone = [self.downsample_layers, self.stages]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True

def ConvBK(num_classes: int, pretrained=True, use_gpu=True, weights='', in_channel=768, **kwargs):
    model = convbk(depths=[3, 3, 27, 3],
                   dims=[96, 192, 384, 768],
                   num_classes=num_classes,
                   use_gpu=use_gpu,
                   in_channel=in_channel)
    if pretrained:
        # model.load_state_dict(torch.load("/data/CNX_V3/model_data/convnext_base_1k_224_ema.pth")["model"])
        weights_dict = torch.load(weights)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
            elif "norm.weight" == k:
                del weights_dict[k]
            elif "norm.bias" == k:
                del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
    return model
#
#
# if __name__ == '__main__':
#     inputs = torch.randn([1, 3, 256, 128])
#
#     b, c, _, _ = inputs.size()
#     model = ConvBK(num_classes=80, use_gpu=False)
#     # model = BIGKS(channel=c, N=4, use_gpu=False)
#     out = model(inputs)
#     # print(out.shape)
