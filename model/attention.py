import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MyAttention(nn.Module):
    def __init__(self, feat_dim, kW, kH):
        super(MyAttention, self).__init__()

        # 卷积层
        self.lr_q_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        self.hr_k_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)
        self.hr_v_conv = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, groups=feat_dim)

        # 参数化卷积层
        self.adaptive_conv = nn.Conv2d(feat_dim*2, 2, kernel_size=3, padding=1)
        self.unfold = nn.Unfold(kernel_size=(kH, kW), padding=(kH//2, kW//2))

        self.softmax = nn.Softmax(dim=1)
        self.kW = kW
        self.kH = kH

    def forward(self, hr_feat, lr_feat):
        # 保持原有预处理
        N, C, H, W = hr_feat.shape
        lr_feat = F.interpolate(lr_feat, (H, W), mode='bilinear', align_corners=True)

        # 特征提取
        hr_value = self.hr_v_conv(hr_feat)
        hr_key = self.hr_k_conv(hr_feat)
        lr_query = self.lr_q_conv(lr_feat)

        # 动态权重计算
        fused_feat = torch.cat([lr_query, hr_key], dim=1)
        dynamic_weights = torch.sigmoid(self.adaptive_conv(fused_feat))

        # 多尺度相似度计算
        hr_key_unfold = self.unfold(hr_key).view(N, C, self.kH*self.kW, H, W)
        similarity = torch.einsum('nchw,nckhw->nkhw', lr_query, hr_key_unfold)
        weight = self.softmax(similarity.view(N, self.kH*self.kW, H, W))

        # 自适应融合
        hr_value_unfold = self.unfold(hr_value).view(N, C, self.kH*self.kW, H, W)
        attention_result = dynamic_weights[:,0:1] * torch.einsum('nckhw,nkhw->nchw', hr_value_unfold, weight)
        attention_result += dynamic_weights[:,1:2] * hr_value

        return lr_feat + attention_result
