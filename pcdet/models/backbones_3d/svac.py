import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
from math import ceil
from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
from .dsvt import DSVTBlock, Stage_Reduction_Block, Stage_ReductionAtt_Block
from pcdet.models.model_utils.dsvt_utils import get_window_coors, get_inner_win_inds_cuda, get_pooling_index, \
    get_continous_inds
from pcdet.models.model_utils.dsvt_utils import PositionEmbeddingLearned
from .utils.svac_utils import get_weight, Att_fusion_concat, Att_fusion_addition


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


class SVAC8x(nn.Module):
    '''
    Sparse Voxel Attentive Convolution Backbone.
    '''

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dsvt_input_layer = self.DSVTInputInformation(self.model_cfg.INPUT_LAYER)
        self.input_information = self.model_cfg.INPUT_LAYER
        sparse_shape = grid_size[::-1].tolist()
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        # batch_size = self.model_cfg.BATCH_SIZE_PER_GPU
        downsample_stride = self.model_cfg.INPUT_LAYER.downsample_stride
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 32] -> [800, 704, 16]
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            # block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 16] -> [400, 352, 8]
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            # block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 8] -> [200, 176, 4]
            block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        # self.conv_branch = []
        # self.conv_branch.append(self.conv1)
        # self.conv_branch.append(self.conv2)
        # self.conv_branch.append(self.conv3)
        # self.conv_branch.append(self.conv4)
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        
        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 64,
            'x_conv3': 64,
            'x_conv4': 64
        }
        self.spatial_shape = {'x_conv0': np.array(sparse_shape)}
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            if not stage_id == stage_num - 1:
                sparse_shape = [sparse_shape[i] // downsample_stride[stage_id][i] for i in range(len(sparse_shape))]
                self.spatial_shape['x_conv{}'.format(stage_id + 1)] = np.array(sparse_shape)
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            # block_name_this_stage = block_name[stage_id]
            block_module = DSVTBlock
            block_list = []
            norm_list = []
            for i in range(num_blocks_this_stage):
                block_list.append(
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                 dropout, activation, batch_first=True)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num - 1:
                downsample_window = downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id + 1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction',
                                     Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction',
                                     Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.stage_num = stage_num
        self.set_info = set_info

    class DSVTInputInformation(nn.Module):

        def __init__(self, model_cfg):
            super().__init__()

            self.model_cfg = model_cfg
            self.sparse_shape = self.model_cfg.sparse_shape
            self.window_shape = self.model_cfg.window_shape
            self.downsample_stride = self.model_cfg.downsample_stride
            self.d_model = self.model_cfg.d_model
            self.set_info = self.model_cfg.set_info
            self.stage_num = len(self.d_model)

            self.hybrid_factor = self.model_cfg.hybrid_factor
            self.window_shape = [
                [self.window_shape[s_id], [self.window_shape[s_id][coord_id] * self.hybrid_factor[coord_id] \
                                           for coord_id in range(3)]] for s_id in range(self.stage_num)]
            self.shift_list = self.model_cfg.shifts_list
            self.normalize_pos = self.model_cfg.normalize_pos

            self.num_shifts = [2, ] * len(self.window_shape)

            self.sparse_shape_list = [self.sparse_shape]
            # compute sparse shapes for each stage
            for ds_stride in self.downsample_stride:
                last_sparse_shape = self.sparse_shape_list[-1]
                self.sparse_shape_list.append((ceil(last_sparse_shape[0] / ds_stride[0]),
                                               ceil(last_sparse_shape[1] / ds_stride[1]),
                                               ceil(last_sparse_shape[2] / ds_stride[2])))

            # position embedding layers
            self.posembed_layers = nn.ModuleList()
            for i in range(len(self.set_info)):
                input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2  # [3,3,3,2]
                stage_posembed_layers = nn.ModuleList()
                for j in range(self.set_info[i][1]):
                    block_posembed_layers = nn.ModuleList()
                    for s in range(self.num_shifts[i]):
                        block_posembed_layers.append(PositionEmbeddingLearned(input_dim, self.d_model[i]))
                    stage_posembed_layers.append(block_posembed_layers)
                self.posembed_layers.append(stage_posembed_layers)

        def forward(self, batch_dict):
            '''
            Args:
                batch_dict (dict):
                    The dict contains the following keys
                    - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]),
                        where N is the number of input voxels.
                    - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                        Each row is (batch_id, z, y, x).
                    - ...

            Returns:
                voxel_info (dict):
                    The dict contains the following keys
                    - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                        Each row is (batch_id, z, y, x).
                    - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                        2 indicates x-axis partition and y-axis partition.
                    - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                    - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the
                        number of remain voxels in stage_i;
                    - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region index used in pooling operation between stage_{i-1} and stage_{i}
                        with shape (N_{i-1}).
                    - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner region with shape (N_{i-1}). Combined with pooling_mapping_index_stage{i},
                        we can map each voxel in stage_{i-1} to pooling_preholder_feats_stage{i}, which are input of downsample operation.
                    - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder features initial with value 0.
                        Shape of (N_{i}, downsample_stride[i-1].prob(), d_moel[i-1]), where prob() returns the product of all elements.
                    - ...
            '''
            voxel_feats = batch_dict['voxel_features']
            voxel_coors = batch_dict['voxel_coords'].long()

            voxel_info = {}
            voxel_info['voxel_feats_stage0'] = voxel_feats.clone()
            voxel_info['voxel_coors_stage0'] = voxel_coors.clone()

            for stage_id in range(self.stage_num):  # [0,1,2,3]
                # window partition of corrsponding stage-map
                voxel_info = self.window_partition(voxel_info, stage_id)
                # generate set id of corrsponding stage-map
                voxel_info = self.get_set(voxel_info, stage_id)
                for block_id in range(self.set_info[stage_id][1]):
                    for shift_id in range(self.num_shifts[stage_id]):
                        voxel_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                            self.get_pos_embed(voxel_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id,
                                               block_id, shift_id)

                # compute pooling information
                if stage_id < self.stage_num - 1:
                    voxel_info = self.subm_pooling(voxel_info, stage_id)

            return voxel_info

        @torch.no_grad()
        def subm_pooling(self, voxel_info, stage_id):
            # x,y,z stride
            cur_stage_downsample = self.downsample_stride[stage_id]
            # batch_win_coords is from 1 of x, y
            batch_win_inds, _, index_in_win, batch_win_coors = get_pooling_index(
                voxel_info[f'voxel_coors_stage{stage_id}'],
                self.sparse_shape_list[stage_id],
                cur_stage_downsample)
            # compute pooling mapping index
            unique_batch_win_inds, contiguous_batch_win_inds = torch.unique(batch_win_inds, return_inverse=True)
            voxel_info[f'pooling_mapping_index_stage{stage_id + 1}'] = contiguous_batch_win_inds

            # generate empty placeholder features
            placeholder_prepool_feats = voxel_info[f'voxel_feats_stage0'].new_zeros((len(unique_batch_win_inds),
                                                                                     torch.prod(torch.IntTensor(
                                                                                         cur_stage_downsample)).item(),
                                                                                     self.d_model[stage_id]))
            voxel_info[f'pooling_index_in_pool_stage{stage_id + 1}'] = index_in_win
            voxel_info[f'pooling_preholder_feats_stage{stage_id + 1}'] = placeholder_prepool_feats

            # compute pooling coordinates
            unique, inverse = unique_batch_win_inds.clone(), contiguous_batch_win_inds.clone()
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
            pool_coors = batch_win_coors[perm]

            voxel_info[f'voxel_coors_stage{stage_id + 1}'] = pool_coors

            return voxel_info

        def get_set(self, voxel_info, stage_id):
            '''
            This is one of the core operation of DSVT.
            Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
            To make it clear and easy to follow, we do not use loop to process two shifts.
            Args:
                voxel_info (dict):
                    The dict contains the following keys
                    - batch_win_inds_s{i} (Tensor[float]): Windows indexs of each voxel with shape (N), computed by 'window_partition'.
                    - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner window of each voxel with shape (N, 3), computed by 'window_partition'.
                        Each row is (z, y, x).
                    - ...

            Returns:
                See from 'forward' function.
            '''
            batch_win_inds_shift0 = voxel_info[f'batch_win_inds_stage{stage_id}_shift0']
            coors_in_win_shift0 = voxel_info[f'coors_in_win_stage{stage_id}_shift0']
            set_voxel_inds_shift0 = self.get_set_single_shift(batch_win_inds_shift0, stage_id, shift_id=0,
                                                              coors_in_win=coors_in_win_shift0)
            voxel_info[f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0
            # compute key masks, voxel duplication must happen continuously
            prefix_set_voxel_inds_s0 = torch.roll(set_voxel_inds_shift0.clone(), shifts=1, dims=-1)
            prefix_set_voxel_inds_s0[:, :, 0] = -1
            set_voxel_mask_s0 = (set_voxel_inds_shift0 == prefix_set_voxel_inds_s0)
            voxel_info[f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0

            batch_win_inds_shift1 = voxel_info[f'batch_win_inds_stage{stage_id}_shift1']
            coors_in_win_shift1 = voxel_info[f'coors_in_win_stage{stage_id}_shift1']
            set_voxel_inds_shift1 = self.get_set_single_shift(batch_win_inds_shift1, stage_id, shift_id=1,
                                                              coors_in_win=coors_in_win_shift1)
            voxel_info[f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1
            # compute key masks, voxel duplication must happen continuously
            prefix_set_voxel_inds_s1 = torch.roll(set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
            prefix_set_voxel_inds_s1[:, :, 0] = -1
            set_voxel_mask_s1 = (set_voxel_inds_shift1 == prefix_set_voxel_inds_s1)
            voxel_info[f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1

            return voxel_info

        def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
            device = batch_win_inds.device
            # the number of voxels assigned to a set
            voxel_num_set = self.set_info[stage_id][0]
            # max number of voxels in a window
            max_voxel = self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][1] * \
                        self.window_shape[stage_id][shift_id][2]
            # get unique set indexs
            contiguous_win_inds = torch.unique(batch_win_inds, return_inverse=True)[1]
            voxelnum_per_win = torch.bincount(contiguous_win_inds)
            win_num = voxelnum_per_win.shape[0]
            setnum_per_win_float = voxelnum_per_win / voxel_num_set
            setnum_per_win = torch.ceil(setnum_per_win_float).long()
            set_win_inds, set_inds_in_win = get_continous_inds(setnum_per_win)

            # compution of Eq.3 in 'DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets' - https://arxiv.org/abs/2301.06051,
            # for each window, we can get voxel indexs belong to different sets.
            offset_idx = set_inds_in_win[:, None].repeat(1, voxel_num_set) * voxel_num_set
            base_idx = torch.arange(0, voxel_num_set, 1, device=device)
            base_select_idx = offset_idx + base_idx
            base_select_idx = base_select_idx * voxelnum_per_win[set_win_inds][:, None]
            base_select_idx = base_select_idx.double() / (setnum_per_win[set_win_inds] * voxel_num_set)[:,
                                                         None].double()
            base_select_idx = torch.floor(base_select_idx)
            # obtain unique indexs in whole space
            select_idx = base_select_idx
            select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel

            # this function will return unordered inner window indexs of each voxel
            inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
            global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
            _, order1 = torch.sort(global_voxel_inds)

            # get y-axis partition results
            global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
                                      coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * \
                                      self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 0]
            _, order2 = torch.sort(global_voxel_inds_sorty)
            inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
            inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[
                order1])  # get y-axis ordered inner window indexs of each voxel
            voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * contiguous_win_inds
            voxel_inds_padding_sorty = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
            voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(0, voxel_inds_in_batch_sorty.shape[0],
                                                                               dtype=torch.long, device=device)
            set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]

            # get x-axis partition results
            global_voxel_inds_sortx = contiguous_win_inds * max_voxel + \
                                      coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * \
                                      self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 0]
            _, order2 = torch.sort(global_voxel_inds_sortx)
            inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
            inner_voxel_inds_sortx.scatter_(dim=0, index=order2, src=inner_voxel_inds[
                order1])  # get x-axis ordered inner window indexs of each voxel
            voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * contiguous_win_inds
            voxel_inds_padding_sortx = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
            voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(0, voxel_inds_in_batch_sortx.shape[0],
                                                                               dtype=torch.long, device=device)
            set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]

            all_set_voxel_inds = torch.stack((set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)
            return all_set_voxel_inds

        @torch.no_grad()
        def window_partition(self, voxel_info, stage_id):
            for i in range(2):
                batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'voxel_coors_stage{stage_id}'],
                                                                self.sparse_shape_list[stage_id],
                                                                self.window_shape[stage_id][i], i == 1,
                                                                self.shift_list[stage_id][i])

                voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
                voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win

            return voxel_info

        def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
            '''
            Args:
            coors_in_win: shape=[N, 3], order: z, y, x
            '''
            # [N,]
            window_shape = self.window_shape[stage_id][shift_id]

            embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
            if len(window_shape) == 2:
                ndim = 2
                win_x, win_y = window_shape
                win_z = 0
            elif window_shape[-1] == 1:
                ndim = 2
                win_x, win_y = window_shape[:2]
                win_z = 0
            else:
                win_x, win_y, win_z = window_shape
                ndim = 3

            assert coors_in_win.size(1) == 3
            z, y, x = coors_in_win[:, 0] - win_z / 2, coors_in_win[:, 1] - win_y / 2, coors_in_win[:, 2] - win_x / 2

            if self.normalize_pos:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
                z = z / win_z * 2 * 3.1415  # [-pi, pi]

            if ndim == 2:
                location = torch.stack((x, y), dim=-1)
            else:
                location = torch.stack((x, y, z), dim=-1)
            pos_embed = embed_layer(location)

            return pos_embed

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """

        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        
        newvoxel_features, newvoxel_coords = batch_dict['voxel_features'], batch_dict[
            'voxel_coords']
        voxel_info_DSVT_branch = {}
        batch_size = batch_dict['batch_size']
        SVAC_input_list = []
        input_sp_tensor = spconv.SparseConvTensor(
            features=newvoxel_features,
            indices=newvoxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        SVAC_input_list.append(x)
        batch_dict['multi_scale_3d_features'] = {}
        # SVAC_stage
        for stage_id in range(self.stage_num):
            current_stage_raw_sp_tensor = eval('self.conv{}(SVAC_input_list[stage_id])'.format(stage_id + 1))
            # current_stage_raw_sp_tensor = self.conv_branch[stage_id](SVAC_input_list[stage_id])
            voxel_info_DSVT_branch['voxel_feats_stage{}'.format(stage_id)] = current_stage_raw_sp_tensor.features
            voxel_info_DSVT_branch[
                'voxel_coors_stage{}'.format(stage_id)] = current_stage_raw_sp_tensor.indices.long()
            # DSVT_stage
            voxel_info_DSVT_branch = self.dsvt_input_layer.window_partition(voxel_info_DSVT_branch,
                                                                            stage_id=stage_id)
            voxel_info_DSVT_branch = self.dsvt_input_layer.get_set(voxel_info_DSVT_branch, stage_id=stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    voxel_info_DSVT_branch[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                        self.dsvt_input_layer.get_pos_embed(
                            voxel_info_DSVT_branch[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id,
                            block_id, shift_id)
            # compute pooling information
            if stage_id < self.stage_num - 1:
                voxel_info_DSVT_branch = self.dsvt_input_layer.subm_pooling(voxel_info_DSVT_branch, stage_id)

            set_voxel_inds_list = [voxel_info_DSVT_branch[f'set_voxel_inds_stage{stage_id}_shift{i}'] for i in
                                   range(self.num_shifts[stage_id])]
            set_voxel_masks_list = [voxel_info_DSVT_branch[f'set_voxel_mask_stage{stage_id}_shift{i}'] for i in
                                    range(self.num_shifts[stage_id])]
            pos_embed_list = [[voxel_info_DSVT_branch[f'pos_embed_stage{stage_id}_block{b}_shift{i}']
                               for i in range(self.num_shifts[stage_id])] for b in
                              range(self.set_info[stage_id][1])]
            if stage_id < self.stage_num - 1:
                pooling_mapping_index = voxel_info_DSVT_branch[f'pooling_mapping_index_stage{stage_id + 1}']
                pooling_index_in_pool = voxel_info_DSVT_branch[f'pooling_index_in_pool_stage{stage_id + 1}']
                pooling_preholder_feats = voxel_info_DSVT_branch[f'pooling_preholder_feats_stage{stage_id + 1}']

            output = voxel_info_DSVT_branch['voxel_feats_stage{}'.format(stage_id)]
            block_id = 0

            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                output = block(output, set_voxel_inds_list, set_voxel_masks_list, pos_embed_list[i],
                               block_id=block_id)
                # fusion_module = Att_fusion_addition(output.shape[1], residual.shape[1])
                # output = fusion_module(output, residual)
                output = residual_norm_layers[i](output + residual)
                output = residual_norm_layers[i](output)  # 加权
                block_id += 1
                # output = residual_norm_layers[i](output + current_stage_sp_tensor)
                current_stage_refined_sp_tensor = spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id)],
                    batch_size=batch_size
                )
                batch_dict['multi_scale_3d_features'][
                    'x_conv{}'.format(stage_id + 1)] = current_stage_refined_sp_tensor
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats.type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index, pooling_index_in_pool] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)

                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(
                        prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError
            if stage_id < self.stage_num - 1:
                SVAC_input_list.append(spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id + 1)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id + 1)],
                    batch_size=batch_size
                ))
            else:
                final_feature = spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id)],
                    batch_size=batch_size
                )
        batch_dict.update({
            # 'encoded_spconv_tensor': out.dense(),
            'encoded_spconv_tensor': final_feature,
            'encoded_spconv_tensor_stride': 8,
        })

        batch_dict.update({
            'encoded_spconv_tensor_stride': 8
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


class SVAC4x(nn.Module):
    '''
    Sparse Voxel Attentive Convolution Backbone.
    '''

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.dsvt_input_layer = self.DSVTInputInformation(self.model_cfg.INPUT_LAYER)
        self.input_information = self.model_cfg.INPUT_LAYER
        sparse_shape = grid_size[::-1].tolist()
        block_name = self.model_cfg.block_name
        set_info = self.model_cfg.set_info
        d_model = self.model_cfg.d_model
        nhead = self.model_cfg.nhead
        downsample_stride = self.model_cfg.INPUT_LAYER.downsample_stride
        dim_feedforward = self.model_cfg.dim_feedforward
        dropout = self.model_cfg.dropout
        activation = self.model_cfg.activation
        self.reduction_type = self.model_cfg.get('reduction_type', 'attention')
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]  # + [1, 0, 0]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 64, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(64),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv1', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [800, 704, 16]-> [400, 352, 8]
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            # block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [400, 352, 8] -> [200, 176, 4]
            # block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        # self.conv_branch = []
        # self.conv_branch.append(self.conv1)
        # self.conv_branch.append(self.conv2)
        # self.conv_branch.append(self.conv3)
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)

        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 64,
            'x_conv2': 64,
            'x_conv3': 64
        }
        self.spatial_shape = {'x_conv0': np.array(sparse_shape)}
        # Sparse Regional Attention Blocks
        stage_num = len(block_name)
        for stage_id in range(stage_num):
            if not stage_id == stage_num - 1:
                sparse_shape = [sparse_shape[i] // downsample_stride[stage_id][i] for i in range(len(sparse_shape))]
                self.spatial_shape['x_conv{}'.format(stage_id + 1)] = np.array(sparse_shape)
            num_blocks_this_stage = set_info[stage_id][-1]
            dmodel_this_stage = d_model[stage_id]
            dfeed_this_stage = dim_feedforward[stage_id]
            num_head_this_stage = nhead[stage_id]
            # block_name_this_stage = block_name[stage_id]
            block_module = DSVTBlock
            block_list = []
            norm_list = []
            for i in range(num_blocks_this_stage):
                block_list.append(
                    block_module(dmodel_this_stage, num_head_this_stage, dfeed_this_stage,
                                 dropout, activation, batch_first=True)
                )
                norm_list.append(nn.LayerNorm(dmodel_this_stage))
            self.__setattr__(f'stage_{stage_id}', nn.ModuleList(block_list))
            self.__setattr__(f'residual_norm_stage_{stage_id}', nn.ModuleList(norm_list))

            # apply pooling except the last stage
            if stage_id < stage_num - 1:
                downsample_window = downsample_stride[stage_id]
                dmodel_next_stage = d_model[stage_id + 1]
                pool_volume = torch.IntTensor(downsample_window).prod().item()
                if self.reduction_type == 'linear':
                    cat_feat_dim = dmodel_this_stage * torch.IntTensor(downsample_window).prod().item()
                    self.__setattr__(f'stage_{stage_id}_reduction',
                                     Stage_Reduction_Block(cat_feat_dim, dmodel_next_stage))
                elif self.reduction_type == 'maxpool':
                    self.__setattr__(f'stage_{stage_id}_reduction', torch.nn.MaxPool1d(pool_volume))
                elif self.reduction_type == 'attention':
                    self.__setattr__(f'stage_{stage_id}_reduction',
                                     Stage_ReductionAtt_Block(dmodel_this_stage, pool_volume))
                else:
                    raise NotImplementedError

        self.num_shifts = [2] * stage_num
        self.stage_num = stage_num
        self.set_info = set_info

    class DSVTInputInformation(nn.Module):

        def __init__(self, model_cfg):
            super().__init__()

            self.model_cfg = model_cfg
            self.sparse_shape = self.model_cfg.sparse_shape
            self.window_shape = self.model_cfg.window_shape
            self.downsample_stride = self.model_cfg.downsample_stride
            self.d_model = self.model_cfg.d_model
            self.set_info = self.model_cfg.set_info
            self.stage_num = len(self.d_model)

            self.hybrid_factor = self.model_cfg.hybrid_factor
            self.window_shape = [
                [self.window_shape[s_id], [self.window_shape[s_id][coord_id] * self.hybrid_factor[coord_id] \
                                           for coord_id in range(3)]] for s_id in range(self.stage_num)]
            self.shift_list = self.model_cfg.shifts_list
            self.normalize_pos = self.model_cfg.normalize_pos

            self.num_shifts = [2, ] * len(self.window_shape)

            self.sparse_shape_list = [self.sparse_shape]
            # compute sparse shapes for each stage
            for ds_stride in self.downsample_stride:
                last_sparse_shape = self.sparse_shape_list[-1]
                self.sparse_shape_list.append((ceil(last_sparse_shape[0] / ds_stride[0]),
                                               ceil(last_sparse_shape[1] / ds_stride[1]),
                                               ceil(last_sparse_shape[2] / ds_stride[2])))

            # position embedding layers
            self.posembed_layers = nn.ModuleList()
            for i in range(len(self.set_info)):
                input_dim = 3 if self.sparse_shape_list[i][-1] > 1 else 2  # [3,3,3,2]
                stage_posembed_layers = nn.ModuleList()
                for j in range(self.set_info[i][1]):
                    block_posembed_layers = nn.ModuleList()
                    for s in range(self.num_shifts[i]):
                        block_posembed_layers.append(PositionEmbeddingLearned(input_dim, self.d_model[i]))
                    stage_posembed_layers.append(block_posembed_layers)
                self.posembed_layers.append(stage_posembed_layers)

        def forward(self, batch_dict):
            '''
            Args:
                batch_dict (dict):
                    The dict contains the following keys
                    - voxel_features (Tensor[float]): Voxel features after VFE with shape (N, d_model[0]),
                        where N is the number of input voxels.
                    - voxel_coords (Tensor[int]): Shape of (N, 4), corresponding voxel coordinates of each voxels.
                        Each row is (batch_id, z, y, x).
                    - ...

            Returns:
                voxel_info (dict):
                    The dict contains the following keys
                    - voxel_coors_stage{i} (Tensor[int]): Shape of (N_i, 4). N is the number of voxels in stage_i.
                        Each row is (batch_id, z, y, x).
                    - set_voxel_inds_stage{i}_shift{j} (Tensor[int]): Set partition index with shape (2, set_num, set_info[i][0]).
                        2 indicates x-axis partition and y-axis partition.
                    - set_voxel_mask_stage{i}_shift{i} (Tensor[bool]): Key mask used in set attention with shape (2, set_num, set_info[i][0]).
                    - pos_embed_stage{i}_block{i}_shift{i} (Tensor[float]): Position embedding vectors with shape (N_i, d_model[i]). N_i is the
                        number of remain voxels in stage_i;
                    - pooling_mapping_index_stage{i} (Tensor[int]): Pooling region index used in pooling operation between stage_{i-1} and stage_{i}
                        with shape (N_{i-1}).
                    - pooling_index_in_pool_stage{i} (Tensor[int]): Index inner region with shape (N_{i-1}). Combined with pooling_mapping_index_stage{i},
                        we can map each voxel in stage_{i-1} to pooling_preholder_feats_stage{i}, which are input of downsample operation.
                    - pooling_preholder_feats_stage{i} (Tensor[int]): Preholder features initial with value 0.
                        Shape of (N_{i}, downsample_stride[i-1].prob(), d_moel[i-1]), where prob() returns the product of all elements.
                    - ...
            '''
            voxel_feats = batch_dict['voxel_features']
            voxel_coors = batch_dict['voxel_coords'].long()

            voxel_info = {}
            voxel_info['voxel_feats_stage0'] = voxel_feats.clone()
            voxel_info['voxel_coors_stage0'] = voxel_coors.clone()

            for stage_id in range(self.stage_num):  # [0,1,2,3]
                # window partition of corrsponding stage-map
                voxel_info = self.window_partition(voxel_info, stage_id)
                # generate set id of corrsponding stage-map
                voxel_info = self.get_set(voxel_info, stage_id)
                for block_id in range(self.set_info[stage_id][1]):
                    for shift_id in range(self.num_shifts[stage_id]):
                        voxel_info[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                            self.get_pos_embed(voxel_info[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id,
                                               block_id, shift_id)

                # compute pooling information
                if stage_id < self.stage_num - 1:
                    voxel_info = self.subm_pooling(voxel_info, stage_id)

            return voxel_info

        @torch.no_grad()
        def subm_pooling(self, voxel_info, stage_id):
            # x,y,z stride
            cur_stage_downsample = self.downsample_stride[stage_id]
            # batch_win_coords is from 1 of x, y
            batch_win_inds, _, index_in_win, batch_win_coors = get_pooling_index(
                voxel_info[f'voxel_coors_stage{stage_id}'],
                self.sparse_shape_list[stage_id],
                cur_stage_downsample)
            # compute pooling mapping index
            unique_batch_win_inds, contiguous_batch_win_inds = torch.unique(batch_win_inds, return_inverse=True)
            voxel_info[f'pooling_mapping_index_stage{stage_id + 1}'] = contiguous_batch_win_inds

            # generate empty placeholder features
            placeholder_prepool_feats = voxel_info[f'voxel_feats_stage0'].new_zeros((len(unique_batch_win_inds),
                                                                                     torch.prod(torch.IntTensor(
                                                                                         cur_stage_downsample)).item(),
                                                                                     self.d_model[stage_id]))
            voxel_info[f'pooling_index_in_pool_stage{stage_id + 1}'] = index_in_win
            voxel_info[f'pooling_preholder_feats_stage{stage_id + 1}'] = placeholder_prepool_feats

            # compute pooling coordinates
            unique, inverse = unique_batch_win_inds.clone(), contiguous_batch_win_inds.clone()
            perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
            inverse, perm = inverse.flip([0]), perm.flip([0])
            perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
            pool_coors = batch_win_coors[perm]

            voxel_info[f'voxel_coors_stage{stage_id + 1}'] = pool_coors

            return voxel_info

        def get_set(self, voxel_info, stage_id):
            '''
            This is one of the core operation of DSVT.
            Given voxels' window ids and relative-coords inner window, we partition them into window-bounded and size-equivalent local sets.
            To make it clear and easy to follow, we do not use loop to process two shifts.
            Args:
                voxel_info (dict):
                    The dict contains the following keys
                    - batch_win_inds_s{i} (Tensor[float]): Windows indexs of each voxel with shape (N), computed by 'window_partition'.
                    - coors_in_win_shift{i} (Tensor[int]): Relative-coords inner window of each voxel with shape (N, 3), computed by 'window_partition'.
                        Each row is (z, y, x).
                    - ...

            Returns:
                See from 'forward' function.
            '''
            batch_win_inds_shift0 = voxel_info[f'batch_win_inds_stage{stage_id}_shift0']
            coors_in_win_shift0 = voxel_info[f'coors_in_win_stage{stage_id}_shift0']
            set_voxel_inds_shift0 = self.get_set_single_shift(batch_win_inds_shift0, stage_id, shift_id=0,
                                                              coors_in_win=coors_in_win_shift0)
            voxel_info[f'set_voxel_inds_stage{stage_id}_shift0'] = set_voxel_inds_shift0
            # compute key masks, voxel duplication must happen continuously
            prefix_set_voxel_inds_s0 = torch.roll(set_voxel_inds_shift0.clone(), shifts=1, dims=-1)
            prefix_set_voxel_inds_s0[:, :, 0] = -1
            set_voxel_mask_s0 = (set_voxel_inds_shift0 == prefix_set_voxel_inds_s0)
            voxel_info[f'set_voxel_mask_stage{stage_id}_shift0'] = set_voxel_mask_s0

            batch_win_inds_shift1 = voxel_info[f'batch_win_inds_stage{stage_id}_shift1']
            coors_in_win_shift1 = voxel_info[f'coors_in_win_stage{stage_id}_shift1']
            set_voxel_inds_shift1 = self.get_set_single_shift(batch_win_inds_shift1, stage_id, shift_id=1,
                                                              coors_in_win=coors_in_win_shift1)
            voxel_info[f'set_voxel_inds_stage{stage_id}_shift1'] = set_voxel_inds_shift1
            # compute key masks, voxel duplication must happen continuously
            prefix_set_voxel_inds_s1 = torch.roll(set_voxel_inds_shift1.clone(), shifts=1, dims=-1)
            prefix_set_voxel_inds_s1[:, :, 0] = -1
            set_voxel_mask_s1 = (set_voxel_inds_shift1 == prefix_set_voxel_inds_s1)
            voxel_info[f'set_voxel_mask_stage{stage_id}_shift1'] = set_voxel_mask_s1

            return voxel_info

        def get_set_single_shift(self, batch_win_inds, stage_id, shift_id=None, coors_in_win=None):
            device = batch_win_inds.device
            # the number of voxels assigned to a set
            voxel_num_set = self.set_info[stage_id][0]
            # max number of voxels in a window
            max_voxel = self.window_shape[stage_id][shift_id][0] * self.window_shape[stage_id][shift_id][1] * \
                        self.window_shape[stage_id][shift_id][2]
            # get unique set indexs
            contiguous_win_inds = torch.unique(batch_win_inds, return_inverse=True)[1]
            voxelnum_per_win = torch.bincount(contiguous_win_inds)
            win_num = voxelnum_per_win.shape[0]
            setnum_per_win_float = voxelnum_per_win / voxel_num_set
            setnum_per_win = torch.ceil(setnum_per_win_float).long()
            set_win_inds, set_inds_in_win = get_continous_inds(setnum_per_win)

            # compution of Eq.3 in 'DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets' - https://arxiv.org/abs/2301.06051,
            # for each window, we can get voxel indexs belong to different sets.
            offset_idx = set_inds_in_win[:, None].repeat(1, voxel_num_set) * voxel_num_set
            base_idx = torch.arange(0, voxel_num_set, 1, device=device)
            base_select_idx = offset_idx + base_idx
            base_select_idx = base_select_idx * voxelnum_per_win[set_win_inds][:, None]
            base_select_idx = base_select_idx.double() / (setnum_per_win[set_win_inds] * voxel_num_set)[:,
                                                         None].double()
            base_select_idx = torch.floor(base_select_idx)
            # obtain unique indexs in whole space
            select_idx = base_select_idx
            select_idx = select_idx + set_win_inds.view(-1, 1) * max_voxel

            # this function will return unordered inner window indexs of each voxel
            inner_voxel_inds = get_inner_win_inds_cuda(contiguous_win_inds)
            global_voxel_inds = contiguous_win_inds * max_voxel + inner_voxel_inds
            _, order1 = torch.sort(global_voxel_inds)

            # get y-axis partition results
            global_voxel_inds_sorty = contiguous_win_inds * max_voxel + \
                                      coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][0] * \
                                      self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 0]
            _, order2 = torch.sort(global_voxel_inds_sorty)
            inner_voxel_inds_sorty = -torch.ones_like(inner_voxel_inds)
            inner_voxel_inds_sorty.scatter_(dim=0, index=order2, src=inner_voxel_inds[
                order1])  # get y-axis ordered inner window indexs of each voxel
            voxel_inds_in_batch_sorty = inner_voxel_inds_sorty + max_voxel * contiguous_win_inds
            voxel_inds_padding_sorty = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
            voxel_inds_padding_sorty[voxel_inds_in_batch_sorty] = torch.arange(0, voxel_inds_in_batch_sorty.shape[0],
                                                                               dtype=torch.long, device=device)
            set_voxel_inds_sorty = voxel_inds_padding_sorty[select_idx.long()]

            # get x-axis partition results
            global_voxel_inds_sortx = contiguous_win_inds * max_voxel + \
                                      coors_in_win[:, 2] * self.window_shape[stage_id][shift_id][1] * \
                                      self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 1] * self.window_shape[stage_id][shift_id][2] + \
                                      coors_in_win[:, 0]
            _, order2 = torch.sort(global_voxel_inds_sortx)
            inner_voxel_inds_sortx = -torch.ones_like(inner_voxel_inds)
            inner_voxel_inds_sortx.scatter_(dim=0, index=order2, src=inner_voxel_inds[
                order1])  # get x-axis ordered inner window indexs of each voxel
            voxel_inds_in_batch_sortx = inner_voxel_inds_sortx + max_voxel * contiguous_win_inds
            voxel_inds_padding_sortx = -1 * torch.ones((win_num * max_voxel), dtype=torch.long, device=device)
            voxel_inds_padding_sortx[voxel_inds_in_batch_sortx] = torch.arange(0, voxel_inds_in_batch_sortx.shape[0],
                                                                               dtype=torch.long, device=device)
            set_voxel_inds_sortx = voxel_inds_padding_sortx[select_idx.long()]

            all_set_voxel_inds = torch.stack((set_voxel_inds_sorty, set_voxel_inds_sortx), dim=0)
            return all_set_voxel_inds

        @torch.no_grad()
        def window_partition(self, voxel_info, stage_id):
            for i in range(2):
                batch_win_inds, coors_in_win = get_window_coors(voxel_info[f'voxel_coors_stage{stage_id}'],
                                                                self.sparse_shape_list[stage_id],
                                                                self.window_shape[stage_id][i], i == 1,
                                                                self.shift_list[stage_id][i])

                voxel_info[f'batch_win_inds_stage{stage_id}_shift{i}'] = batch_win_inds
                voxel_info[f'coors_in_win_stage{stage_id}_shift{i}'] = coors_in_win

            return voxel_info

        def get_pos_embed(self, coors_in_win, stage_id, block_id, shift_id):
            '''
            Args:
            coors_in_win: shape=[N, 3], order: z, y, x
            '''
            # [N,]
            window_shape = self.window_shape[stage_id][shift_id]

            embed_layer = self.posembed_layers[stage_id][block_id][shift_id]
            if len(window_shape) == 2:
                ndim = 2
                win_x, win_y = window_shape
                win_z = 0
            elif window_shape[-1] == 1:
                ndim = 2
                win_x, win_y = window_shape[:2]
                win_z = 0
            else:
                win_x, win_y, win_z = window_shape
                ndim = 3

            assert coors_in_win.size(1) == 3
            z, y, x = coors_in_win[:, 0] - win_z / 2, coors_in_win[:, 1] - win_y / 2, coors_in_win[:, 2] - win_x / 2

            if self.normalize_pos:
                x = x / win_x * 2 * 3.1415  # [-pi, pi]
                y = y / win_y * 2 * 3.1415  # [-pi, pi]
                z = z / win_z * 2 * 3.1415  # [-pi, pi]

            if ndim == 2:
                location = torch.stack((x, y), dim=-1)
            else:
                location = torch.stack((x, y, z), dim=-1)
            pos_embed = embed_layer(location)

            return pos_embed

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """

        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):

        newvoxel_features, newvoxel_coords = batch_dict['voxel_features'], batch_dict[
            'voxel_coords']
        voxel_info_DSVT_branch = {}
        batch_size = batch_dict['batch_size']
        SVAC_input_list = []
        input_sp_tensor = spconv.SparseConvTensor(
            features=newvoxel_features,
            indices=newvoxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        SVAC_input_list.append(x)
        batch_dict['multi_scale_3d_features'] = {}
        # SVAC_stage
        for stage_id in range(self.stage_num):
            current_stage_raw_sp_tensor = eval('self.conv{}(SVAC_input_list[stage_id])'.format(stage_id + 1))
            # current_stage_raw_sp_tensor = self.conv_branch[stage_id](SVAC_input_list[stage_id])
            voxel_info_DSVT_branch['voxel_feats_stage{}'.format(stage_id)] = current_stage_raw_sp_tensor.features
            voxel_info_DSVT_branch[
                'voxel_coors_stage{}'.format(stage_id)] = current_stage_raw_sp_tensor.indices.long()
            # DSVT_stage
            voxel_info_DSVT_branch = self.dsvt_input_layer.window_partition(voxel_info_DSVT_branch,
                                                                            stage_id=stage_id)
            voxel_info_DSVT_branch = self.dsvt_input_layer.get_set(voxel_info_DSVT_branch, stage_id=stage_id)
            for block_id in range(self.set_info[stage_id][1]):
                for shift_id in range(self.num_shifts[stage_id]):
                    voxel_info_DSVT_branch[f'pos_embed_stage{stage_id}_block{block_id}_shift{shift_id}'] = \
                        self.dsvt_input_layer.get_pos_embed(
                            voxel_info_DSVT_branch[f'coors_in_win_stage{stage_id}_shift{shift_id}'], stage_id,
                            block_id, shift_id)
            # compute pooling information
            if stage_id < self.stage_num - 1:
                voxel_info_DSVT_branch = self.dsvt_input_layer.subm_pooling(voxel_info_DSVT_branch, stage_id)

            set_voxel_inds_list = [voxel_info_DSVT_branch[f'set_voxel_inds_stage{stage_id}_shift{i}'] for i in
                                   range(self.num_shifts[stage_id])]
            set_voxel_masks_list = [voxel_info_DSVT_branch[f'set_voxel_mask_stage{stage_id}_shift{i}'] for i in
                                    range(self.num_shifts[stage_id])]
            pos_embed_list = [[voxel_info_DSVT_branch[f'pos_embed_stage{stage_id}_block{b}_shift{i}']
                               for i in range(self.num_shifts[stage_id])] for b in
                              range(self.set_info[stage_id][1])]
            if stage_id < self.stage_num - 1:
                pooling_mapping_index = voxel_info_DSVT_branch[f'pooling_mapping_index_stage{stage_id + 1}']
                pooling_index_in_pool = voxel_info_DSVT_branch[f'pooling_index_in_pool_stage{stage_id + 1}']
                pooling_preholder_feats = voxel_info_DSVT_branch[f'pooling_preholder_feats_stage{stage_id + 1}']

            output = voxel_info_DSVT_branch['voxel_feats_stage{}'.format(stage_id)]
            block_id = 0

            block_layers = self.__getattr__(f'stage_{stage_id}')
            residual_norm_layers = self.__getattr__(f'residual_norm_stage_{stage_id}')
            for i in range(len(block_layers)):
                block = block_layers[i]
                residual = output.clone()
                output = block(output, set_voxel_inds_list, set_voxel_masks_list, pos_embed_list[i],
                               block_id=block_id)
                output = residual_norm_layers[i](output + residual)  # 加权
                block_id += 1
                # output = residual_norm_layers[i](output + current_stage_sp_tensor)
                current_stage_refined_sp_tensor = spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id)],
                    batch_size=batch_size
                )
                batch_dict['multi_scale_3d_features'][
                    'x_conv{}'.format(stage_id + 1)] = current_stage_refined_sp_tensor
            if stage_id < self.stage_num - 1:
                # pooling
                prepool_features = pooling_preholder_feats.type_as(output)
                pooled_voxel_num = prepool_features.shape[0]
                pool_volume = prepool_features.shape[1]
                prepool_features[pooling_mapping_index, pooling_index_in_pool] = output
                prepool_features = prepool_features.view(prepool_features.shape[0], -1)

                if self.reduction_type == 'linear':
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features)
                elif self.reduction_type == 'maxpool':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features).squeeze(-1)
                elif self.reduction_type == 'attention':
                    prepool_features = prepool_features.view(pooled_voxel_num, pool_volume, -1).permute(0, 2, 1)
                    key_padding_mask = torch.zeros((pooled_voxel_num, pool_volume)).to(
                        prepool_features.device).int()
                    output = self.__getattr__(f'stage_{stage_id}_reduction')(prepool_features, key_padding_mask)
                else:
                    raise NotImplementedError
            if stage_id < self.stage_num - 1:
                SVAC_input_list.append(spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id + 1)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id + 1)],
                    batch_size=batch_size
                ))
            else:
                final_feature = spconv.SparseConvTensor(
                    features=output,
                    indices=voxel_info_DSVT_branch['voxel_coors_stage{}'.format(stage_id)].int(),
                    spatial_shape=self.spatial_shape['x_conv{}'.format(stage_id)],
                    batch_size=batch_size
                )
        batch_dict.update({
            # 'encoded_spconv_tensor': out.dense(),
            'encoded_spconv_tensor': final_feature,
            'encoded_spconv_tensor_stride': 4,
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
            }
        })

        return batch_dict