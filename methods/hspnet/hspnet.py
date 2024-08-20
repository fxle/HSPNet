import abc
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone.pvt_v2_eff import pvt_v2_eff_b2, pvt_v2_eff_b5
from .layers import ISPM, SAP, ITCM, CSAM, ISPM_MS
from .ops import CBR, PixelNormalizer

LOGGER = logging.getLogger("main")

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    
    return (wbce + wiou).mean()

class _HSPNet_Base(nn.Module):
    @staticmethod
    def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = 0, 1

        ual_coef = 1.0
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            if method == "linear":
                ratio = (max_coef - min_coef) / (max_point - min_point)
                ual_coef = ratio * (iter_percentage - min_point)
            elif method == "cos":
                perc = (iter_percentage - min_point) / (max_point - min_point)
                normalized_coef = (1 - np.cos(perc * np.pi)) / 2
                ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
        return ual_coef

    @abc.abstractmethod
    def body(self):
        pass


    def forward(self, data, iter_percentage=1, **kwargs):

        logits = self.body(data=data)
        
        if self.training:
            mask = data["mask"]
            prob = logits.sigmoid()

            losses = []
            loss_str = []

            # sod_loss = F.binary_cross_entropy_with_logits(input=logits, target=mask, reduction="mean")
            sod_loss = structure_loss(logits, mask)
            # sod_loss = hybrid_e_loss(logits, mask)
            losses.append(sod_loss)
            loss_str.append(f"bce: {sod_loss.item():.5f}")

            ual_coef = self.get_coef(iter_percentage=iter_percentage, method="cos", milestones=(0, 1))
            ual_loss = ual_coef * (1 - (2 * prob - 1).abs().pow(2)).mean()
            losses.append(ual_loss)
            loss_str.append(f"powual_{ual_coef:.5f}: {ual_loss.item():.5f}")

            return dict(vis=dict(sal=prob), loss=sum(losses), loss_str=" ".join(loss_str))
        else:
            return logits


    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "clip." in name:
                    param.requires_grad = False
                    param_groups["fixed"].append(param)
                else:
                    param_groups["retrained"].append(param)
        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class PvtV2B2_HSPNet(_HSPNet_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        n_groups=4,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tran_5 = SAP(self.embed_dims[3], out_dim=mid_dim)
        self.ispm_5 = ISPM(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_5 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_4 = CBR(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.ispm_4 = ISPM(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_4 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_3 = CBR(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.ispm_3 = ISPM(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_3 = ITCM(mid_dim,  num_frames=num_frames)

        self.tran_2 = CBR(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.ispm_2 = ISPM(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_2 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), CBR(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.pred= nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            CBR(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        self.csam1 = CSAM(64)
        self.csam2 = CSAM(64)
        self.csam0 = CSAM(64)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5


    def body(self, data):
         
        feats = self.normalize_encoder(data["image"]) 

        spa = self.ispm_5(self.tran_5(feats[3]))   
        tem5 = self.itcm_5(spa)

        spa = self.ispm_4(self.tran_4(feats[2]))
        tem4 = self.itcm_4(spa)

        spa = self.ispm_3(self.tran_3(feats[1]))
        tem3 = self.itcm_3(spa)

        spa = self.ispm_2(self.tran_2(feats[0]))
        tem2 = self.itcm_2(spa)
        
        tem54 = self.csam2(tem5, tem4)
        tem543 = self.csam1(tem54, tem3)
        st = self.csam0(tem543, tem2)

        st = self.tran_1(st)
        
        return self.pred(st)


class PvtV2B5_HSPNet(PvtV2B2_HSPNet):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_HSPNet(PvtV2B5_HSPNet):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups


class PvtV2B2_HSPNet_MS(_HSPNet_Base):
    def __init__(
        self,
        pretrained=True,
        num_frames=1,
        input_norm=True,
        mid_dim=64,
        n_groups=4,
        use_checkpoint=False,
    ):
        super().__init__()
        self.set_backbone(pretrained=pretrained, use_checkpoint=use_checkpoint)

        self.embed_dims = self.encoder.embed_dims
        self.tran_5 = SAP(self.embed_dims[3], out_dim=mid_dim)
        self.ispm_5 = ISPM_MS(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_5 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_4 = CBR(self.embed_dims[2], mid_dim, 3, 1, 1)
        self.ispm_4 = ISPM_MS(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_4 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_3 = CBR(self.embed_dims[1], mid_dim, 3, 1, 1)
        self.ispm_3 = ISPM_MS(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_3 = ITCM(mid_dim,  num_frames=num_frames)

        self.tran_2 = CBR(self.embed_dims[0], mid_dim, 3, 1, 1)
        self.ispm_2 = ISPM_MS(mid_dim, n_groups, num_frames=num_frames)
        self.itcm_2 = ITCM(mid_dim, num_frames=num_frames)

        self.tran_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), CBR(64, mid_dim, 3, 1, 1)
        )

        self.normalizer = PixelNormalizer() if input_norm else nn.Identity()
        self.pred= nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            CBR(64, 32, 3, 1, 1),
            nn.Conv2d(32, 1, 1),
        )

        self.csam1 = CSAM(64)
        self.csam2 = CSAM(64)
        self.csam0 = CSAM(64)

    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b2(pretrained=pretrained, use_checkpoint=use_checkpoint)

    def normalize_encoder(self, x):
        x = self.normalizer(x)
        features = self.encoder(x)
        c2 = features["reduction_2"]
        c3 = features["reduction_3"]
        c4 = features["reduction_4"]
        c5 = features["reduction_5"]
        return c2, c3, c4, c5


    def body(self, data):
         
        l_trans_feats = self.normalize_encoder(data["image_l"])
        m_trans_feats = self.normalize_encoder(data["image_m"])
        s_trans_feats = self.normalize_encoder(data["image_s"])

        spa = self.ispm_5(self.tran_5(l_trans_feats[3]), self.tran_5(m_trans_feats[3]), self.tran_5(s_trans_feats[3]))
        tem5 = self.itcm_5(spa)

        spa = self.ispm_4(self.tran_4(l_trans_feats[2]), self.tran_4(m_trans_feats[2]), self.tran_4(s_trans_feats[2]))
        tem4 = self.itcm_4(spa)
        
        spa = self.ispm_3(self.tran_3(l_trans_feats[1]), self.tran_3(m_trans_feats[1]), self.tran_3(s_trans_feats[1]))
        tem3 = self.itcm_3(spa)

        spa = self.ispm_2(self.tran_2(l_trans_feats[0]), self.tran_2(m_trans_feats[0]), self.tran_2(s_trans_feats[0]))
        tem2 = self.itcm_2(spa)
        
        tem54 = self.csam2(tem5, tem4)
        tem543 = self.csam1(tem54, tem3)
        st = self.csam0(tem543, tem2)

        st = self.tran_1(st)
        
        return self.pred(st)


class PvtV2B5_HSPNet_MS(PvtV2B2_HSPNet_MS):
    def set_backbone(self, pretrained: bool, use_checkpoint: bool):
        self.encoder = pvt_v2_eff_b5(pretrained=pretrained, use_checkpoint=use_checkpoint)


class videoPvtV2B5_HSPNet_MS(PvtV2B5_HSPNet_MS):
    def get_grouped_params(self):
        param_groups = {"pretrained": [], "fixed": [], "retrained": []}
        for name, param in self.named_parameters():
            if name.startswith("encoder.patch_embed1."):
                param.requires_grad = False
                param_groups["fixed"].append(param)
            elif name.startswith("encoder."):
                param_groups["pretrained"].append(param)
            else:
                if "temperal_proj" in name:
                    param_groups["retrained"].append(param)
                else:
                    param_groups["pretrained"].append(param)

        LOGGER.info(
            f"Parameter Groups:{{"
            f"Pretrained: {len(param_groups['pretrained'])}, "
            f"Fixed: {len(param_groups['fixed'])}, "
            f"ReTrained: {len(param_groups['retrained'])}}}"
        )
        return param_groups
