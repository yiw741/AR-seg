import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo
from torch.nn import BatchNorm2d
from AR_seg.attention import MyAttention

resnet_weight = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    # 每一大层第一层会进行残差，在他之下的都不会，直到下一个大块，也就是旁边的乘数
    for i in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)
class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=3, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=4, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=6, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=3, stride=2)
        self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet_weight['resnet34'])
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            # 赋值给self_state_dict也就是本模型
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []

        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)

            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())

        return wd_params, nowd_params


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):
    def __init__(self, nchannle, scale=2):
        super(UpSample, self).__init__()
        out_channel = nchannle * scale * scale
        self.conv = nn.Conv2d(in_channels=nchannle, out_channels=out_channel, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):
    def __init__(self, in_channel, mid_channel, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_channel, mid_channel, kernel_size=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_channel, n_classes, kernel_size=1, bias=False)
        self.up_factor = up_factor
        self.up = nn.Upsample(scale_factor=up_factor, mode="bilinear", align_corners=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


# 通道注意力
class AttentionInChannal(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super(AttentionInChannal, self).__init__()
        self.conv = ConvBNReLU(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)  # 生成注意力权重
        self.bn = nn.BatchNorm2d(out_channel)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn(atten)
        atten = self.sigmoid(atten)
        x = x * atten
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet34()

        self.arm16 = AttentionInChannal(256, 128)
        self.arm32 = AttentionInChannal(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)
        self.up32 = nn.Upsample(scale_factor=2.)
        self.up16 = nn.Upsample(scale_factor=2.)

        self.init_weight()

    def forward(self, x):

        feat8, feat16, feat32 = self.resnet(x)


        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)

        avg = self.conv_avg(avg)


        feat32_arm = self.arm32(feat32)

        feat32_sum = feat32_arm + avg

        feat32_up = self.up32(feat32_sum)

        feat32_up = F.interpolate(feat32_up, [feat16.shape[-2], feat16.shape[-1]], mode='bilinear', align_corners=True)

        feat32_up = self.conv_head32(feat32_up)


        feat16_arm = self.arm16(feat16)

        feat16_sum = feat16_arm + feat32_up

        feat16_up = self.up16(feat16_sum)

        feat16_up = self.conv_head16(feat16_up)


        return feat16_up, feat32_up

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 128, ks=3, stride=2, padding=1)
        self.conv4 = ConvBNReLU(128, 128, ks=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, 1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        fcat = self.convblk(fcat)
        atten = torch.mean(fcat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.bn1(atten)
        atten = self.sigmoid(atten)
        fout = torch.mul(fcat, atten) + fcat
        return fout

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetV1WithFuse(nn.Module):
    def __init__(self, out_chan, mode="train", attention_type="local", k=7, *args, **kwargs):
        super(BiSeNetV1WithFuse, self).__init__()
        self.mode = mode
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, out_chan, up_factor=8)

        self.feature_conv = self.conv_out.conv
        self.final_conv = self.conv_out.conv_out  # 修正变量名拼写错误
        self.up_conv = self.conv_out.up
        self.middle_dim = 256

        if self.mode == "train":
            self.conv_out16 = BiSeNetOutput(128, 128, out_chan, up_factor=4)
            self.conv_out32 = BiSeNetOutput(128, 128, out_chan, up_factor=2)

        if attention_type == "local":
            self.fuse_attention = MyAttention(self.middle_dim, kH=k, kW=k)

        # 确保 init_weight 方法存在
        self.init_weight()


    def forward(self, x, ref_p=None):  # 添加默认值处理
        if self.mode == 'train':
            feat_out16, feat_out32, middle_feat = self.forward_phase1(x)
            out, out_p = self.forward_phase2(middle_feat, ref_p)
            return out, out_p, feat_out16, feat_out32
        else:
            middle_feat = self.forward_phase1(x)
            out, out_p = self.forward_phase2(middle_feat, ref_p)
            return out

    def forward_phase1(self, x):
        feat_cp8, feat_cp16 = self.context_path(x)
        feat_sp = self.spatial_path(x)
        feat_sp = F.interpolate(feat_sp, feat_cp8.shape[-2:], mode='bilinear', align_corners=True)
        feat_fuse = self.ffm(feat_sp, feat_cp8)
        middle_feat = self.feature_conv(feat_fuse)  # 修正为feature_conv

        if self.mode == 'train':
            feat_out16 = self.conv_out16(feat_cp8)
            feat_out32 = self.conv_out32(feat_cp16)
            return feat_out16, feat_out32, middle_feat
        return middle_feat

    def forward_phase2(self, middle_feat, ref_p):
        if ref_p is None:  # 处理空输入
            ref_p = torch.zeros_like(middle_feat)

        p = self.fuse_attention(ref_p, middle_feat)
        out = self.final_conv(p)
        out = self.up_conv(out)
        return out, p

    def init_weight(self):
            for ly in self.children():
                if isinstance(ly, nn.Conv2d):
                    nn.init.kaiming_normal_(ly.weight, a=1)
                    if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

class OhemCELoss(nn.Module):

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


if __name__ == "__main__":
    net = BiSeNetV1WithFuse(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(1, 3, 360, 480).cuda()
    out, out16, out32, out_p = net(in_ten)
    print(out.shape)
    print(out16.shape)
    print(out32.shape)

