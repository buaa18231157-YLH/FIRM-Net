import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

'''
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            padding: int = 1,
            downsample: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_layer(x)

        out += identity
        out = self.relu2(out)

        return out
'''

class BaseBEVResBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(
                self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                # nn.ZeroPad2d(1),
                BasicBlock(c_in_list[idx], num_filters[idx], layer_strides[idx], 1, True)
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    BasicBlock(num_filters[idx], num_filters[idx])
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters) if len(num_upsample_filters) > 0 else sum(num_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class ASPPNeck(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(ASPPNeck, self).__init__()
        self.model_cfg = model_cfg
        self.pre_conv = BasicBlock(input_channels)
        self.conv1x1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(input_channels, input_channels, 3, 3))
        self.post_conv = ConvBlock(input_channels * 6, input_channels, kernel_size=1, stride=1)
        self.num_bev_features = input_channels

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        # ups = []
        # ret_dict = {}
        x = spatial_features
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1, bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1, bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1, bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1, bias=None, padding=18, dilation=18)
        x = self.post_conv(torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))

        data_dict['spatial_features_2d'] = x

        return data_dict


class ASPPDeConvNeck(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(ASPPDeConvNeck, self).__init__()
        self.model_cfg = model_cfg
        self.upsample_rate = self.model_cfg.UPSAMPLE_RATE
        self.pre_conv = BasicBlock(input_channels)
        self.conv1x1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(input_channels, input_channels, 3, 3))
        self.post_conv = ConvBlock(input_channels * 6, input_channels, kernel_size=1, stride=1)
        self.num_bev_features = input_channels
        deconv_layer = ConvBlock(input_channels, input_channels, kernel_size=2, stride=2, padding=0, conv_layer=nn.ConvTranspose2d)
        num_layers = int(np.log2(self.upsample_rate))
        if self.upsample_rate > 1:
            self.deconv = nn.Sequential()
            for i in range(num_layers):
                self.deconv.add_module(f'deconv{i+1}', deconv_layer)
        else:
            self.deconv = nn.Identity()

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        # ups = []
        # ret_dict = {}
        x = spatial_features
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1, bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1, bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1, bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1, bias=None, padding=18, dilation=18)
        x = self.post_conv(torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        x = self.deconv(x)
        data_dict['spatial_features_2d'] = x

        return data_dict
        
class ASPPDeConvNeckV2(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super(ASPPDeConvNeckV2, self).__init__()
        self.model_cfg = model_cfg
        self.upsample_rate = self.model_cfg.UPSAMPLE_RATE
        self.pre_conv = BasicBlock(input_channels)
        self.conv1x1 = nn.Conv2d(
            input_channels, input_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(input_channels, input_channels, 3, 3))
        self.post_conv = ConvBlock(input_channels * 6, input_channels, kernel_size=1, stride=1)
        self.num_bev_features = input_channels
        deconv_layer = ConvBlock(input_channels * 6, input_channels, kernel_size=2, stride=2, padding=0, conv_layer=nn.ConvTranspose2d)
        num_layers = int(np.log2(self.upsample_rate))
        if self.upsample_rate > 1:
            self.deconv = nn.Sequential()
            # self.deconv.add_module('deconv_hid1', nn.ConvTranspose2d(input_channels, input_channels, 3, stride=1,padding=1))
            for i in range(num_layers):
                self.deconv.add_module(f'deconv{i+1}', deconv_layer)
        else:
            self.deconv = nn.Identity()

    def forward(self, data_dict):
        spatial_features = data_dict['spatial_features']
        # ups = []
        # ret_dict = {}
        x = spatial_features
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1, bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1, bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1, bias=None, padding=12, dilation=12)
        branch24 = F.conv2d(x, self.weight, stride=1, bias=None, padding=24, dilation=24)
        x = torch.cat((x, branch1x1, branch1, branch6, branch12, branch24), dim=1)
        x = self.deconv(x)
        up_branch1 = F.conv2d(x, self.weight, stride=1, bias=None, padding=1, dilation=1)
        up_branch3 = F.conv2d(x, self.weight, stride=1, bias=None, padding=3, dilation=3)
        up_branch6 = F.conv2d(x, self.weight, stride=1, bias=None, padding=6, dilation=6)
        up_branch12 = F.conv2d(x, self.weight, stride=1, bias=None, padding=12, dilation=12)
        up_branch24 = F.conv2d(x, self.weight, stride=1, bias=None, padding=24, dilation=24)
        x = self.post_conv(torch.cat((x, up_branch1, up_branch3, up_branch6, up_branch12, up_branch24), dim=1))
        data_dict['spatial_features_2d'] = x

        return data_dict

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = conv_layer(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.norm = norm_layer(out_channel)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channel, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(in_channel, in_channel, kernel_size=kernel_size)
        self.block2 = ConvBlock(in_channel, in_channel, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out