import os
from PIL import Image
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import torchvision
import numpy as np
from .kitti_object import *


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True).to(self.device)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ).to(self.device)

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D  # 70
        self.C = C  # 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征
        # self.eff = eff_preprocess(torchvision.models.efficientnet_b0(pretrained=False))
        self.eff = eff_preprocess(EfficientNet.from_name("efficientnet-b0"))

        self.up1 = Up(112 + 40, 256)  # 上采样模块，输入输出通道分别为112 + 80 和 256
        self.depthnet = nn.Conv2d(256, self.D + self.C, kernel_size=1, padding=0).to(self.device)  # 1x1卷积，变换维度

    def get_depth_dist(self, x):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)  # 使用efficientnet提取特征  x: 2 x 256 x 16 x 54
        # Depth
        x = self.depthnet(x)  # 1x1卷积变换维度  x: 2 x 134(C+D) x 16 x 54

        depth = self.get_depth_dist(x[:, :self.D])  # 第二个维度的前D个作为深度维，进行softmax  depth: 2 x 70 x 16 x 54
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)  # 将特征通道维和通道维利用广播机制相乘  new_x: 2 x 64 x 70 x 16 x 54

        return depth, new_x

    def get_eff_depth(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        multi_scale_img_feature = dict()
        prev_x = x
        # Blocks
        for idx, block in enumerate(list(self.eff._blocks.children())):
            x = block(x)
            if prev_x.size(2) > x.size(2):
                multi_scale_img_feature['reduction_{}'.format(len(multi_scale_img_feature))] = prev_x
            prev_x = x

        # Head
        multi_scale_img_feature['reduction_{}'.format(len(multi_scale_img_feature))] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(multi_scale_img_feature['reduction_4'], multi_scale_img_feature['reduction_3'])
        return x  # x: 2 x 256 x 16 x 54

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # depth: B x D x fH x fW(2 x 70 x 16 x 54)  x: B x C x D x fH x fW(2 x 64 x 70 x 16 x 54)

        return x


class Lifter(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC, bsz):
        super(Lifter, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_conf = grid_conf  # 网格配置参数
        self.data_aug_conf = data_aug_conf  # 数据增强配置参数
        self.batchsize = bsz  # 批大小
        self.downsample = self.data_aug_conf['downsample_factor']  # 下采样倍数
        self.resize_factor = self.data_aug_conf['original_image_size']/self.data_aug_conf['final_size']  # [2.875, 2.9296875]
        self.camC = outC  # 图像特征维度
        self.frustum = self.create_frustum()  # frustum: DxfHxfWx3(70*16*54*3)
        self.D = self.frustum.shape[1]  # D: 70
        self.camencode = CamEncode(self.D, self.camC, self.downsample)

        # toggle using QuickCumsum vs. autograd
        # self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfW, ogfH = self.data_aug_conf['final_size'][0]  # 原始图片大小  ogfH:128  ogfW:432
        fH, fW = torch.div(ogfH, self.downsample, rounding_mode='floor'), torch.div(ogfW, self.downsample, rounding_mode='floor') # 下采样8倍后图像大小  fH: 16  fW: 54
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(70 16 54)

        D, _, _ = ds.shape  # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到431上划分54个格子 xs: DxfHxfW(70 16 54)

        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(70 16 54)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1).unsqueeze(0).expand(self.batchsize, D, fH, fW, 3).clone()  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        frustum = self.frustum_to_original_img_coords(frustum.to(self.device))
        # return nn.Parameter(frustum, requires_grad=False)
        return frustum

    def frustum_to_original_img_coords(self, frustum):
        """
        Convert frustum coordinates to original image coordinates.
        """
        resize = self.resize_factor.view(self.batchsize, 1, 1, 1, 2)
        frustum[:, :, :, :, 0:2] *= resize
        frustum[:, :, :, :, 0:2] += resize / 2

        return frustum.round()

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 6(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]^T

        return points  # B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)

    def get_cam_feats(self, x):
        """Return B x D x H/downsample x W/downsample x C
        """
        # B, C, imH, imW = x.shape  # B: 2  C: 3  imH: 128  imW: 432

        # x = x.view(B * N, C, imH, imW)  # B和N两个维度合起来  x: 2 x 3 x 128 x 432
        x = self.camencode(x)  # 进行图像编码  x: B x C x D x fH x fW(2 x 64 x 70 x 16 x 54)
        x = x.permute(0, 2, 3, 4, 1)  # x: B x D x fH x fW x C(2 x 70 x 16 x 54 x 64)

        return x.reshape(self.batchsize, -1, 64)

    def get_voxels(self, x):
        x = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x 6 x 41 x 8 x 22 x 64)
        # geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x 6 x 41 x 8 x 22 x 3)

        # x = self.voxel_pooling(geom, x)  # x: 4 x 64 x 200 x 200

        return x

    def forward(self, x):  # , rots, trans, intrins, post_rots, post_trans
        # x:[4,6,3,128,352]
        # rots: [4,6,3,3]
        # trans: [4,6,3]
        # intrins: [4,6,3,3]
        # post_rots: [4,6,3,3]
        # post_trans: [4,6,3]
        x = self.get_voxels(x)  # 将图像转换到BEV下，x: B x C x 200 x 200 (4 x 64 x 200 x 200)
        # print(self.frustum.shape)
        return x, self.frustum.view(self.batchsize, -1, 3)


def get_RGBImage_size(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    size = np.array(img_file.size)
    img_file.close()
    return size

def RGBImage_preprocess(filename, cropsize=(432, 128)):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # print(img_file.size)
    img = img_file.resize(cropsize)
    rgb_png = np.array(img, dtype=int)
    img_file.close()

    rgb = rgb_png.astype(float)
    return rgb / 256.


def eff_preprocess(net):  # efficientnet_b0 preprocessing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net._blocks =  nn.Sequential(net._conv_stem, net._bn0, *list(net._blocks.children())[:-4])
    del net._conv_stem, net._bn0
    del net._bn1, net._conv_head, net._fc
    del net._avg_pooling, net._dropout, net._swish
    return net.to(device)


def convert_string_list_to_int_list(string_list):
    int_list = []
    for string in string_list:
        # 去除字符串中的字母
        string = ''.join(filter(str.isdigit, string))
        # 移除字符串开头的零，并将剩余部分转换为整数
        int_list.append(int(string))
    return int_list


def pseudo_image2new_velo(idx_list, coords, features, aug_param):
    dataset = kitti_object('/data2/yulehang/myDetection/OpenPCDet/data/kitti')
    # dataset = kitti_object('~/myDetection/OpenPCDet/data/kitti')
    coords_list = []
    feature_list = []
    for i, data_idx in enumerate(idx_list):
        calib = dataset.get_calibration(data_idx)
        # img_rgb = dataset.get_image(data_idx)
        pc_uvz = coords[i].cpu()
        cur_feat = features[i]
        pc_velo = calib.project_image_to_velo(pc_uvz)
        batch_mask = np.ones([pc_velo.shape[0], 1]) * i
        pc_velo = np.concatenate((pc_velo, batch_mask), axis=1)
        # print(pc_velo.shape)
        pc_velo = np.matmul(pc_velo, aug_param[i].T.cpu())
        valid_mask = (pc_velo[:, 0] > 0) & (pc_velo[:, 0] < 70.4) & (pc_velo[:, 1] > -40.0) & (pc_velo[:, 1] < 40.0)\
                     & (pc_velo[:, 2] > -3.0) & (pc_velo[:, 2] < 1.0)
        pc_velo = pc_velo[valid_mask]
        cur_feat = cur_feat[valid_mask]

        # np.save('/home/ylh/Desktop/temp_data/{}_mm.npy'.format('%06d'%(data_idx)), pc_velo)
        coords_list.append(pc_velo)
        feature_list.append(cur_feat)
    coords_new = coords_list[0]
    feature_new = feature_list[0]
    for i in range(len(coords_list) - 1):
        coords_new = np.concatenate((coords_new, coords_list[i + 1]), axis=0)
        feature_new = torch.cat((feature_new, feature_list[i + 1]), dim=0)
    return torch.tensor(coords_new), feature_new


def new_velo2voxelzyx(coords_velo, grid_conf, voxelization_param):
    coords = coords_velo.clone()
    coords[:, 0] = torch.div((coords[:, 0] - grid_conf['xbound'][0]) * voxelization_param[0], grid_conf['xbound'][1] - grid_conf['xbound'][0], rounding_mode='floor')
    coords[:, 1] = torch.div((coords[:, 1] - grid_conf['ybound'][0]) * voxelization_param[1], grid_conf['ybound'][1] - grid_conf['ybound'][0], rounding_mode='floor')
    coords[:, 2] = torch.div((coords[:, 2] - grid_conf['zbound'][0]) * voxelization_param[2], grid_conf['zbound'][1] - grid_conf['zbound'][0], rounding_mode='floor')
    coords = coords.long()
    coords_new0, coords_new1, coords_new2, coords_new3 = coords[:, 0], coords[:, 1], coords[:, 2],  coords[:, 3]
    return torch.stack([coords_new3, coords_new2, coords_new1, coords_new0], dim=1)

def main():
    '''
    test code.
    '''
    batchsize = 2
    cropsize = np.array([[432, 128]])
    cropsize_arr = cropsize.repeat(batchsize, axis=0)
    downsample_factor = 8
    rot_matrix = np.array([[[0.9872, -0.2515, 0, 0],
                    [0.2515, 0.9872, 0, 0],
                    [0,0,1.0188,0],
                    [0,0,0,1]],[[0.9872, -0.2515, 0, 0],
                    [0.2515, 0.9872, 0, 0],
                    [0,0,1.0188,0],
                    [0,0,0,1]]])
    voxelization_param = np.array([176, 200, 5])
    path = '/home/ylh/Desktop/mydetection/OpenPCDet_no_change/data/kitti/training/image_2'
    files = ['000000.png', '005827.png']
    img_path_lst = [os.path.join(path, each_file) for each_file in files]
    # print(os.path.exists(img_path))
    original_image_size = [get_RGBImage_size(img_path) for img_path in img_path_lst]
    original_image_size = np.array(original_image_size)
    img = [RGBImage_preprocess(img_path) for img_path in img_path_lst]  # 对图像进行预处理
    img = np.array(img)  # 将图像堆叠起来
    img = torch.tensor(img).to(torch.float32).permute(0, 3, 1, 2)
    # print(img.shape)
    # print(type(img))

    scale_factor = downsample_factor * original_image_size / cropsize_arr
    # print(img)

    # point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    xbound = np.array([0.0, 70.4])  # 限制x方向的范围并划分网格
    ybound = np.array([-40.0, 40.0])  # 限制y方向的范围并划分网格
    zbound = np.array([-3.0, 1.0])  # 限制z方向的范围并划分网格
    dbound = np.array([0.0, 70.0, 1.0])  # 深度范围1m-70m，应该足够

    grid_conf = {  # 网格配置
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'downsample_factor': downsample_factor,  # 8
        'final_size': torch.from_numpy(cropsize_arr),  # np.array([432, 128],[432, 128])
        'original_image_size': torch.tensor(original_image_size),  # np.array([1224, 370], [1242, 375])
        'rot_matrix': torch.tensor(rot_matrix)
    }
    model = Lifter(grid_conf, data_aug_conf, outC=64, bsz=2)
    encoded_features, coords_grid = model(img)
    # print(encoded_features.shape)
    # print(coords[coords[0]==0].shape)
    new_velo_coords, pseudo_feature = pseudo_image2new_velo(
        idx_list=convert_string_list_to_int_list(files), coords=coords_grid, features=encoded_features, aug_param=rot_matrix)
    print(new_velo_coords.shape)
    print(new_velo_coords)
    print(pseudo_feature.shape)
    print(pseudo_feature)
    voxel_coords = new_velo2voxelzyx(coords_velo=new_velo_coords, grid_conf=grid_conf, voxelization_param=voxelization_param)
    print(voxel_coords.shape)
    print(voxel_coords)



if __name__ == '__main__':
    main()
