import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .ops import CBR, resize_to
from mmengine.model import constant_init, kaiming_init



class ISPM(nn.Module):
    def __init__(self, in_dim, num_groups=4, num_frames=1):
        super(ISPM, self).__init__()
        self.num_frames = num_frames

        self.conv = CBR(in_dim, 3 * in_dim, 3, 1, 1)  
        self.conv_input = CBR(3 * in_dim, 3 * in_dim, 1)  
        self.initial_merge = CBR(3 * in_dim, 3 * in_dim, 1)  
        
        self.num_groups = num_groups
        self.trans = nn.Sequential(
            CBR(3 * in_dim // num_groups, in_dim // num_groups, 1),
            CBR(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def process_attention(self, input):

        attn = self.conv_input(input)  # BT,3C,H,W
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=3, ng=self.num_groups)
        attn = self.trans(attn)  # BTG,3,H,W
        attn = attn.unsqueeze(dim=2)  # BTG,3,1,H,W

        x = self.initial_merge(input)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=3, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)

        return x
    
    def forward(self, x):
        
        input = self.conv(x)
        
        t =self.num_frames
 
        input = rearrange(input, "(b t) c h w -> t b c h w", t=t)
        m_out = torch.stack(list(map(self.process_attention, input)))
        m_out = rearrange(m_out, "t b c h w ->(b t) c h w ", t=t)
        return m_out
    
    
class ISPM_MS(nn.Module):
    def __init__(self, in_dim=64, num_groups=4, num_frames=1):
        super(ISPM_MS, self).__init__()
        self.num_frames = num_frames
        self.nb = 3
        self.conv_l_pre = CBR(in_dim, in_dim, 3, 1, 1)
        self.conv_s_pre = CBR(in_dim, in_dim, 3, 1, 1)

        self.conv_l = CBR(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_m = CBR(in_dim, in_dim, 3, 1, 1)  # intra-branch
        self.conv_s = CBR(in_dim, in_dim, 3, 1, 1)  # intra-branch

        self.conv = CBR(in_dim, 3 * in_dim, 3, 1, 1)  
        self.conv_input = CBR(3 * in_dim, 3 * in_dim, 1)  
        self.initial_merge = CBR(3 * in_dim, 3 * in_dim, 1)  
        
        self.num_groups = num_groups
        self.trans = nn.Sequential(
            CBR(3 * in_dim // num_groups, in_dim // num_groups, 1),
            CBR(in_dim // num_groups, in_dim // num_groups, 3, 1, 1),
            nn.Conv2d(in_dim // num_groups, 3, 1),
            nn.Softmax(dim=1),
        )

    def process_attention(self, input):

        attn = self.conv_input(input)  # BT,3C,H,W
        attn = rearrange(attn, "bt (nb ng d) h w -> (bt ng) (nb d) h w", nb=self.nb, ng=self.num_groups)
        attn = self.trans(attn)  # BTG,3,H,W
        attn = attn.unsqueeze(dim=2)  # BTG,3,1,H,W

        x = self.initial_merge(input)
        x = rearrange(x, "bt (nb ng d) h w -> (bt ng) nb d h w", nb=self.nb, ng=self.num_groups)
        x = (attn * x).sum(dim=1)
        x = rearrange(x, "(bt ng) d h w -> bt (ng d) h w", ng=self.num_groups)

        return x
    
    def forward(self, l, m, s):
        
        tgt_size = m.shape[2:]

        l = self.conv_l_pre(l)
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        s = self.conv_s_pre(s)
        s = resize_to(s, tgt_hw=m.shape[2:])

        l = self.conv_l(l)
        m = self.conv_m(m)
        s = self.conv_s(s)
        input = torch.cat([l, m, s], dim=1)  # BT,3C,H,W

        t =self.num_frames
 
        input = rearrange(input, "(b t) c h w -> t b c h w", t=t)
        m_out = torch.stack(list(map(self.process_attention, input)))
        m_out = rearrange(m_out, "t b c h w ->(b t) c h w ", t=t)
        
        # m_out = x
        return m_out
    
    
class ITCM(nn.Module):
    def __init__(self, in_c, num_frames=1):
        super().__init__()
        self.num_frames = num_frames
        
        self.final_relu = nn.ReLU(True)
        out_channels= in_c
        self.F1 =  CBR(in_c, out_channels, kernel_size=3, padding=1)
        self.F2 =  CBR(in_c, out_channels, kernel_size=3, padding=1)
        self.getalpha = CA()

        # Define layers
        self.temperal_proj_norm = nn.LayerNorm(2*in_c, elementwise_affine=False)

        self.initial_merge = CBR( 2*in_c, 2*in_c, 1)
        self.F3 = CBR(in_c * 2, in_c, kernel_size=1)
        self.FL = CBR(in_c, in_c, kernel_size=1)
        self.out = CBR(in_c * 2, in_c, kernel_size=3, padding=1)

        for t in self.parameters():
            nn.init.zeros_(t)
        
    def frame(self, x1, x2):

        # Process inputs through initial convolutions
        x1 = self.F1(x1)
        x2 = self.F2(x2)

        # Concatenate features
        cat2 = torch.cat([x1, x2], dim=1)  # 2,128,48,48
        shifted_x_tmp = torch.roll(cat2, shifts=1, dims=-1)
        diff_tmp = shifted_x_tmp - cat2  # B,C,H,W,T
        diff_tmp = self.temperal_proj_norm(diff_tmp.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        y = self.out(diff_tmp)
        alpha = self.getalpha(y)

        fused_feature = torch.cat([x1 * alpha,  x2 * (1 - alpha)], dim=1)  
        fused_feature = self.F3(fused_feature) 

        return fused_feature
    
    def forward(self, x):

        t=self.num_frames
        feat = rearrange(x, "(b t) c h w -> t b c h w", t=t)

        feat = [self.frame(feat[i], feat[i-1]) if i >= 2 else self.FL(feat[i]) for i in range(feat.size(0))]
        out = rearrange(feat, "t b c h w ->(b t) c h w ", t=t)

        return self.final_relu(out + x)
    
class CSAM(nn.Module):
    def __init__(self, in_channels):
        super(CSAM, self).__init__()


        self.cat02 =CBR(in_channels*2, in_channels, 3, stride=1, padding=1)
        self.cat012 =CBR(in_channels*2, in_channels, 3, stride=1, padding=1)

        self.temperal_proj_norm = nn.LayerNorm(in_channels * 2, elementwise_affine=False)
        self.pooling_type = 'att'
        self.channel_add_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels*16, kernel_size=1),
                nn.LayerNorm([in_channels*16, 1, 1]),
                nn.ReLU(inplace=True),  
                nn.Conv2d(in_channels*16, in_channels, kernel_size=1),
                nn.Sigmoid()
                )
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)

    
    def forward(self, feature0, feature2):

        feature0 = F.interpolate(feature0, size=feature2.size()[2:], mode='bilinear', align_corners=True)
        ###1
        feat02 = torch.cat([feature0, feature2], dim=1)  # 2,128,48,48
        feat02 = self.cat02(feat02)
        #channel-atten
        alpha = self.channel_add_conv(feat02)

        feat0120 = torch.cat([feature2 * alpha,  feature0 * (1 - alpha)], dim=1)  # 2,128,48,48
        fused_feature = self.cat012(feat0120)
        
        return fused_feature
    
def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)
 


class CA(nn.Module):
    def __init__(self, lf=True):
        super(CA, self).__init__()
        self.ap = nn.AdaptiveAvgPool2d(1) if lf else nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.ap(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    

class SAP(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=3):

        super().__init__()
        self.conv1x1_1 = CBR(in_dim, 2 * out_dim, 1)
        self.conv1x1_2 = CBR(out_dim, out_dim, 1)
        self.conv3x3_1 = CBR(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_2 = CBR(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.conv3x3_3 = CBR(out_dim, out_dim, 3, dilation=dilation, padding=dilation)
        self.fuse = nn.Sequential(CBR(5 * out_dim, out_dim, 1), CBR(out_dim, out_dim, 3, 1, 1))

    def forward(self, x):
        y = self.conv1x1_1(x)
        y1, y5 = y.chunk(2, dim=1)

        # dilation branch
        y2 = self.conv3x3_1(y1)
        y3 = self.conv3x3_2(y2)
        y4 = self.conv3x3_3(y3)

        # global branch
        y0 = torch.mean(y5, dim=(2, 3), keepdim=True)
        y0 = self.conv1x1_2(y0)
        y0 = resize_to(y0, tgt_hw=x.shape[-2:])
        return self.fuse(torch.cat([y0, y1, y2, y3, y4], dim=1))
