import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


def chunk(x, N):
    # fearture = list()

    out = torch.Tensor().cuda()
    A = torch.chunk(x, N, dim=-1)

    for a in A:
        B = torch.chunk(a, N, dim=-2)
        for i in range(len(B)):
            a = B[i]
            out = torch.cat([out, a], dim=1).cuda()

    return out


class PAD(nn.Module):
    def __init__(self, **kwargs):
        super(PAD, self).__init__()

    def forward(self, x):
        # _, _, w, h = self.inputs.size()
        _, _, h, w = x.size()

        if w % 2 == 0 and not (h % 2 == 0):
            # out = torch.ones_like(inputs)
            # print("w为偶h为奇!!!")
            pad_h = nn.ZeroPad2d((0, 0, 0, 1))
            x = pad_h(x).cuda()
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            else:
                x = x

        elif not (w % 2 == 0) and h % 2 == 0:
            # print("w为奇h为偶!!!")
            pad_w = nn.ZeroPad2d((0, 1, 0, 0))
            x = pad_w(x).cuda()
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            else:
                x = x

        elif not (w % 2 == 0 and h % 2 == 0):
            # print("w为奇h为奇!!!")
            pad_w_h = nn.ZeroPad2d((0, 1, 0, 1))
            x = pad_w_h(x).cuda()
            _, _, p_h, p_w = x.size()
            # print(inputs.shape)
            p = abs((p_h - p_w))

            if w > h:
                pad_ = nn.ZeroPad2d((0, 0, 1, 1))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
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
                    x = pad_(x).cuda()
            elif w < h:
                pad_ = nn.ZeroPad2d((1, 1, 0, 0))
                for i in range(0, p // 2):
                    x = pad_(x).cuda()
            else:
                x = x

        return x


class BIGKS(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=[21, 17], stride=1, I=0, N=0):
        super(BIGKS, self).__init__()
        self.channel = channel
        self.N = N
        self.pad = PAD()
        k1 = kernel_size[0]
        p1 = (k1 - 1) // 2

        k2 = kernel_size[1]
        p2 = (stride + k2 - I)

        self.bk1 = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=k1, padding=p1, stride=stride),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel // N, kernel_size=3, padding=1, stride=1)
        )

        self.bk2 = nn.Sequential(
            nn.Conv2d(channel // N, channel // N, kernel_size=k2, padding=p2, stride=stride),
            nn.ReLU(),
            nn.Conv2d(channel // N, channel, kernel_size=3, padding=1, stride=1),
        )
        # self.des1 = nn.Conv2d(self.channel, self.channel, kernel_size=12, padding=0, stride=1)
        # self.des2 = nn.Conv2d(self.channel, self.channel, kernel_size=4, padding=0, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel, True),
            nn.ReLU(),
            nn.Linear(self.channel, self.channel, True),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.desen = nn.Conv2d(channel // N, channel // N, kernel_size=I, padding=0, stride=1)

    def forward(self, x):
        inputs = x
        b, c, _, _ = x.size()

        x = self.pad(x)

        x_bk1 = self.bk1(x)

        x_bk2 = self.bk2(x_bk1)
        #_, _, bk2_h, bk2_w = x_bk2.size()

        x_bk1_c = chunk(x_bk1, self.N // 2)
        # _, _, bk1_h, bk1_w = x_bk1_c.size()

        avg_bk2 = self.max_pool(x_bk2).view([b, c])
        avg_bk1 = self.max_pool(x_bk1_c).view([b, c])
        # max_ = self.max_pool(x).view([b, c])

        bk1_fc = self.fc(avg_bk1).view([b, c, 1, 1])

        bk2_fc = self.fc(avg_bk2).view([b, c, 1, 1])

        y = bk1_fc + bk2_fc

        return inputs * y


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


class ConvBK(nn.Module):
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
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., pad=''
                 ):
        super().__init__()

        self.bk = BIGKS(channel=dims[-1], I=7, N=4, pad=pad)
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
        # print(x.shape)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x

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


def convbk(num_classes: int, pretrained=False, **kwargs):
    model = ConvBK(depths=[3, 3, 9, 3],
                   dims=[96, 192, 384, 768],
                   num_classes=num_classes)
    if pretrained:
        # model.load_state_dict(torch.load("/data/CNX_V3/model_data/convnext_base_1k_224_ema.pth")["model"])
        weights_dict = \
            torch.load("D://PyCharm/xiaozhijie/modify_project/CNX_V3/model_data/convnext_tiny_1k_224_ema.pth")["model"]
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


def convbk_base(num_classes: int, pretrained=False, pad='', **kwargs):
    model = ConvBK(depths=[3, 3, 27, 3],
                   dims=[128, 256, 512, 1024],
                   num_classes=num_classes,
                   pad=pad)
    if pretrained:
        # model.load_state_dict(torch.load("/data/CNX_V3/model_data/convnext_base_1k_224_ema.pth")["model"])
        weights_dict = torch.load("/data/CNX_ImageNet100/model_data/convnext_base_1k_224_ema.pth")["model"]
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


if __name__ == '__main__':
    inputs = torch.randn([2, 3, 224, 224])
    model = convbk(num_classes=100)
    out = model(inputs)
