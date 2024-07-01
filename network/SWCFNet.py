import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils
import numpy as np

class Network(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.config.num_layers):
            if j < 3:
                d_in = d_out + 2 * self.config.d_out[-j-2]
                d_out = 2 * self.config.d_out[-j-2]
            else:
                d_in = 4 * self.config.d_out[-4]
                d_out = 2 * self.config.d_out[-4]
            self.decoder_blocks.append(pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True))

        self.fc1 = pt_utils.Conv2d(d_out, 64, kernel_size=(1, 1), bn=True)

        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)

        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)

        self.global_f = Global_Attention(4, 32, 2)
        
        inpu = 32
        out = 128
        self.mlp = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range((self.config.num_layers-1)):
            self.mlp.append(MLP_block(inpu, out))
            inpu = out
            out = 2*out
        out_ = 32
        for i in range((self.config.num_layers)):
            self.conv.append(conv_block(out_*2, out_))
            if i == 0:
                out_ = 4*out_
            else:
                out_ = 2*out_
        
    def forward(self, end_points):
        features = end_points['features']
        features = self.fc0(features)
        features = features.unsqueeze(dim=3)
        
        input_xyz = end_points['xyz'][0]
        input_intensity = end_points['intensity'][0]
        input = torch.concat((input_xyz, input_intensity), dim=-1)
        input = input.transpose(2, 1)
        global_feature = self.global_f(input) # [5, 32, 45056]
        
        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])
            
            if i == 0:
                f_lg = global_feature.permute((0, 2, 1)) * f_encoder_i.squeeze(-1).permute((0, 2, 1))
                f_lg = torch.sum(f_lg, dim=-1)
                f_g_norm = torch.norm(f_encoder_i.squeeze(-1).permute((0, 2, 1)), p=2, dim=2)
                f_proj = f_lg / f_g_norm
                f_proj = f_proj.unsqueeze(-1) * f_encoder_i.squeeze(-1).permute((0, 2, 1))
                f_orth = global_feature.permute((0, 2, 1)) - f_proj
                f_orth = f_orth.permute((0, 2, 1)).unsqueeze(-1)
                f_encoder_i = torch.cat((f_orth, f_encoder_i), dim=1)
                g_f = self.random_sample(global_feature.unsqueeze(-1), end_points['sub_idx'][i]).squeeze(-1)
            else:
                g_f = self.mlp[i-1](g_f)
                f_lg = g_f.permute((0, 2, 1)) * f_encoder_i.squeeze(-1).permute((0, 2, 1))
                f_lg = torch.sum(f_lg, dim=-1)
                f_g_norm = torch.norm(f_encoder_i.squeeze(-1).permute((0, 2, 1)), p=2, dim=2)
                f_proj = f_lg / f_g_norm
                f_proj = f_proj.unsqueeze(-1) * f_encoder_i.squeeze(-1).permute((0, 2, 1))
                f_orth = g_f.permute((0, 2, 1)) - f_proj
                f_orth = f_orth.permute((0, 2, 1)).unsqueeze(-1)
                f_encoder_i = torch.cat((f_orth, f_encoder_i), dim=1)
                g_f = self.random_sample(g_f.unsqueeze(-1), end_points['sub_idx'][i]).squeeze(-1)
            f_encoder_i = self.conv[i](f_encoder_i)
            
            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])

        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.config.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        end_points['logits'] = f_out
        return end_points

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features

class MLP_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = torch.nn.Conv1d(d_in, d_out, 1)
        self.bn = nn.BatchNorm1d(d_out)
    def forward(self, feature):
        return F.relu(self.bn(self.conv(feature)))

class conv_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = torch.nn.Conv2d(d_in, d_out, (1, 1))
    def forward(self, feature):
        return self.conv(feature)

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out//2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out*2, d_out*2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out*2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_pc = self.lfa(xyz, f_pc, neigh_idx)
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)

class Building_block(nn.Module):
    def __init__(self, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out//2, kernel_size=(1, 1), bn=True)

        self.weight = WeightNet(3, 16)
        
        self.fc = nn.Conv2d(d_out//2, d_out//2, (1, 1), bias=False)
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = d_out//2
        mlp = []
        for i in range(3):
            mlp.append(d_out)
            if i == 2:
                mlp.append(2*d_out)
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

    def forward(self, xyz, feature, neigh_idx):
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx).permute((0, 3, 1, 2))
        
        weights = self.weight(f_xyz)
        
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        feature_tile = feature.permute((0, 2, 3, 1)).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_f = feature_tile - f_neighbours
        feature_r = relative_f.permute((0, 3, 1, 2))

        score = F.softmax(self.fc(feature_r), dim=3)
        feature_r = score * feature_r
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feature_r =  F.relu(bn(conv(feature_r)))
        B, _, N, _ = feature_r.size()
        new_features = torch.matmul(input=feature_r.permute(0, 2, 1, 3), other = weights.permute(0, 2, 3, 1)).view(B, N, -1)
        new_features = self.linear(new_features)
        new_features = self.bn_linear(new_features.permute(0, 2, 1)).unsqueeze(-1)

        return new_features

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)
        relative_xyz = xyz_tile - neighbor_xyz

        return relative_xyz

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):

        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)
        return features

class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)

        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):

        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg
    
class Global_Attention(nn.Module):
    def __init__(self, d_in, d_out, h):
        super().__init__()
        self.h = h
        
        self.fc = nn.Conv1d(d_in, d_out, 1, bias=False)
        self.proj_keyvalue_conv = nn.Conv1d(d_out, 2 * d_out, 1, bias=False)
        self.proj_query_conv = nn.Conv1d(d_out, d_out, 1, bias=False)
        
        self.pool = nn.AdaptiveAvgPool1d(176)
        self.sr = nn.Conv1d(d_out, d_out, 1)
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.GELU()
        
        self.ff_conv = nn.Sequential(
            nn.Conv1d(d_out, 2 * d_out, 1),
            nn.BatchNorm1d(2 * d_out, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(2 * d_out, d_out, 1),
            nn.BatchNorm1d(d_out, eps=1e-6, momentum=0.99)
            )

    def forward(self, xyz):
        
        xyz_proj = self.fc(xyz)
        B, C, N = xyz_proj.shape
        
        proj_q = self.proj_query_conv(xyz_proj).view(B, self.h, -1, N).permute((0, 1, 3, 2))
        
        xyz_proj_ = self.sr(self.pool(xyz_proj)).permute((0, 2, 1))
        xyz_proj_ = self.norm(xyz_proj_)
        xyz_proj_ = self.act(xyz_proj_).permute((0, 2, 1))
        
        proj_kv = self.proj_keyvalue_conv(xyz_proj_).view(B, C//self.h, 2, self.h, -1).permute((2, 0, 3, 1, 4))
        proj_k, proj_v = proj_kv[0], proj_kv[1]
        
        proj_kq_matmul = torch.matmul(proj_q, proj_k)
        
        proj_coef = F.softmax(proj_kq_matmul / np.sqrt(C//self.h), dim=3)
        
        proj_matmul = torch.matmul(proj_v, proj_coef.permute((0, 1, 3, 2)))
        
        global_f = proj_matmul.view(B, -1, N)
        
        global_f = global_f + self.ff_conv(global_f + xyz_proj)
        
        return global_f