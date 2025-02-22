import torch
import torch.nn as nn
import torch.nn.functional as F


class get_weight(nn.Module):
    def __init__(self, channels):
        super().__init__()
        device = torch.device('cuda:0')
        self.dim3_in, self.dim2_in = channels
        middle = self.dim2_in // 4
        self.fc1 = nn.Linear(self.dim3_in, middle).to(device)
        self.fc2 = nn.Linear(self.dim2_in, middle).to(device)
        self.fc3 = nn.Linear(2 * middle, 2).to(device)

        '''
        self.conv1 = nn.Sequential(nn.Conv1d(self.pseudo_in, self.valid_in, 1),
                                   nn.BatchNorm1d(self.valid_in),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(self.valid_in, self.valid_in, 1),
                                   nn.BatchNorm1d(self.valid_in),
                                   nn.ReLU())
        '''

    def forward(self, dim3_feas, dim2_feas):
        dim3_feas_f_ = self.fc1(dim3_feas)
        dim2_feas_f_ = self.fc2(dim2_feas)
        dim3_dim2_feas_f = torch.cat([dim3_feas_f_, dim2_feas_f_], dim=-1)
        weight = torch.sigmoid(self.fc3(dim3_dim2_feas_f))
        dim3_weight = weight[:, 0].view(-1, 1)
        dim2_weight = weight[:, 1].view(-1, 1)

        dim3_features_att = dim3_feas * dim3_weight
        dim2_features_att = dim2_feas * dim2_weight

        return dim3_features_att, dim2_features_att


class Att_fusion_addition(nn.Module):
    def __init__(self, dim3_feat_in, dim2_feat_in):
        super().__init__()
        self.weight_layer = get_weight(channels=[dim3_feat_in, dim2_feat_in])

    def forward(self, dim3_features, dim2_features):
        dim3_features_att, dim2_features_att = self.weight_layer(dim3_features, dim2_features)
        fusion_features = dim3_features_att + dim2_features_att
        return fusion_features

class Att_fusion_concat(nn.Module):
    def __init__(self, dim3_feat_in, dim2_feat_in):
        super().__init__()
        device = torch.device('cuda:0')
        self.weight_layer = get_weight(channels=[dim3_feat_in, dim2_feat_in])
        self.fc = nn.Linear(dim3_feat_in + dim2_feat_in, dim3_feat_in).to(device)

    def forward(self, dim3_features, dim2_features):
        dim3_features_att, dim2_features_att = self.weight_layer(dim3_features, dim2_features)
        fusion_features = torch.cat([dim3_features_att, dim2_features_att], dim=1)
        # fusion_features = dim3_features_att + dim2_features_att
        fusion_features = self.fc(fusion_features)
        return fusion_features


def main():
    pseudo_features = torch.randn(666, 32)
    valid_features = torch.randn(666, 32)
    fusion_features = Att_fusion_addition(32, 32)(pseudo_features, valid_features)
    print(fusion_features.shape)
    print(pseudo_features)
    print(valid_features)
    print(fusion_features)
    return 0


if __name__ == '__main__':
    main()
