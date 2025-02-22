from functools import partial
import torch
import numpy as np
import torch_scatter
import torch.nn as nn

from ...utils.spconv_utils import replace_feature, spconv
from .utils.IVDF_utils import Lifter, pseudo_image2new_velo, new_velo2voxelzyx

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def feature_index_fusion(feature1, index1, feature2, index2):
    assert feature1.shape[1] == feature2.shape[1]
    combined_coords = torch.cat((index1, index2), dim=0)
    # combined_features = torch.cat((feature1, feature2), dim=0)
    unique_coords, unique_indices, unique_counts = np.unique(combined_coords.data.cpu().numpy(), return_index=True,
                                                             return_counts=True, axis=0)
    same = torch.where((combined_coords == combined_coords[unique_indices[unique_counts == 2]][:, None]).all(-1))[1]
    same_idx = torch.stack((same[0::2], same[1::2]))
    res = torch.zeros(feature1.shape[0] + feature2.shape[0], feature1.shape[1])
    res[:feature1.shape[0], :] = feature1
    res[feature1.shape[0]:, :] = feature2
    res[same_idx[0], :] += res[same_idx[1], :]
    result, coords = res.data.cpu().numpy(), combined_coords.data.cpu().numpy()
    coords = np.delete(coords, same_idx[1].data.cpu().numpy(), axis=0)
    result = np.delete(result, same_idx[1].data.cpu().numpy(), axis=0)
    result, coords = torch.from_numpy(result), torch.from_numpy(coords)
    return result.to(feature1.device), coords.to(feature1.device)

class IVDFadd8x(nn.Module):
    """
    Implicit voxel_depth_fusion module.
    """

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.downsample_factor = 8
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        downsampled_image_shape = batch_dict['downsampled_image_shape'].type(torch.int32)
        voxelization_param = self.sparse_shape[::-1] // self.downsample_factor
        img_batch = batch_dict['image'].permute(0, 3, 1, 2)
        idx_list = [int(x) for x in batch_dict['frame_id']]
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
            'downsample_factor': self.downsample_factor,  # 8
            'final_size': batch_dict['downsampled_image_shape'].type(torch.int32),  # np.array([128, 432])
            'original_image_size': torch.tensor(batch_dict['image_shape'].data.cpu().numpy()[:, ::-1].copy(),
                                                dtype=torch.int32, device=batch_dict['image_shape'].device),
            'rot_matrix': batch_dict['lidar_aug_matrix']
        }
        model = Lifter(grid_conf, data_aug_conf, outC=64, bsz=batch_dict['batch_size'])
        encoded_features, coords_grid = model(img_batch)
        new_velo_coords, pseudo_feature = pseudo_image2new_velo(
            idx_list=idx_list, coords=coords_grid, features=encoded_features,
            aug_param=batch_dict['lidar_aug_matrix'])
        voxel_coords_liftedimg = new_velo2voxelzyx(coords_velo=new_velo_coords, grid_conf=grid_conf,
                                                   voxelization_param=voxelization_param)
        # print(voxel_coords_liftedimg)
        combined = torch.cat(((voxel_coords_liftedimg.to(self.device)), pseudo_feature), dim=1)
        unique_coords, unique_indices = torch.unique(combined[:, :4], return_inverse=True, dim=0)
        pooled_features = torch_scatter.scatter_max(combined[:, 4:], unique_indices.unsqueeze(1), dim=0)[0]
        pooled_coords = unique_coords.type(torch.int32)

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        fused_feat, fused_coords = feature_index_fusion(x_conv4.features, x_conv4.indices, pooled_features, pooled_coords)
        newx_conv4 = spconv.SparseConvTensor(
            features=fused_feat,
            indices=fused_coords.int(),
            spatial_shape=x_conv4.spatial_shape,
            batch_size=batch_size
        )
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(newx_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict