import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unidac.models.dpt_decoder import DPTHead
from unidac.models.scale_est import ScaleMapEstimation

class UniDAC(nn.Module):
    def __init__(
        self,
        pixel_encoder:nn.Module,
        pixel_decoder:nn.Module,
        rope_lat_weight: bool,
        scale_head: nn.Module,
        eps: float=1e-6,
        **kwargs
    ):
        super().__init__()
        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = pixel_decoder
        self.rope_lat_weight = rope_lat_weight
        self.scale_head = scale_head
        self.eps = eps
        
        print("RoPE-Lat-Weight: ", self.rope_lat_weight)

    def forward(
        self,
        image: torch.Tensor,
        lat_range: torch.Tensor,
        long_range: torch.Tensor,
        gt: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        lat_grid: Optional[torch.Tensor] = None,
        long_grid: Optional[torch.Tensor] = None
    ):
        B,_,H,W = image.shape
        losses = {"opt": {}, "stat": {}}
        original_shape = gt.shape[-2:] if gt is not None else image.shape[-2:]

        if self.rope_lat_weight:
            feat_cls = self.pixel_encoder(image, lat_grid=lat_grid, return_cls_token=True) # List of [(features, cls)]
        else:
            feat_cls = self.pixel_encoder(image, return_cls_token=True)
        
        out = self.pixel_decoder(feat_cls)

        pred_rel_depth = sum(
            [
                torch.nn.functional.interpolate(
                    x.clone(),
                    size=original_shape,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                for x in out
            ]
        ) / len(out)

        # pred_rel_clamp = torch.clamp(pred_rel_depth.detach(), min=self.eps)
        with torch.no_grad():
            mask_ = attn_mask
            if mask_ is not None:
                pred_rel_nan= torch.where(mask_.bool(), pred_rel_depth.detach(), torch.nan)
                pred_rel_depth_median = torch.nanmedian(pred_rel_nan.view(B, 1, -1), dim=-1).values.view(pred_rel_depth.shape[:2])
            else:
                # print("No mask")
                pred_rel_depth_median = torch.median(pred_rel_depth.view(B, 1, -1), dim=-1).values.view(pred_rel_depth.shape[:2])
        
        pred_rel_depth_norm = pred_rel_depth.detach().clone()/pred_rel_depth_median[...,None,None]

        feat_g = feat_cls[-1][0]
        scale_map = self.scale_head(feat_g, rel_depth=pred_rel_depth_norm.detach(), mask=mask_, lat_grid=lat_grid)
        pred_metric_depth = scale_map * (pred_rel_depth_norm.detach())

        return (
            pred_metric_depth if pred_metric_depth.shape[1] == 1 else pred_metric_depth[:, :3],
            losses,
            {"outs": out, "queries": None},
        )
    
    def load_pretrained(self, model_file):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        dict_model = torch.load(model_file, map_location=device)
        # handle the case checkpoint including training states
        if 'model' in dict_model: 
            dict_model = dict_model['model']
        new_state_dict = deepcopy(
            {k.replace("module.", ""): v for k, v in dict_model.items()}
        )
        self.load_state_dict(new_state_dict)

    def get_params(self, config):
        backbone_lr = config["model"]["pixel_encoder"].get(
            "lr_dedicated", config["training"]["lr"] / 10
        )
        params = [
            {"params": self.pixel_decoder.parameters()},
            {"params": self.scale_head.parameters()},
            {"params": self.pixel_encoder.parameters()},
        ]
        max_lrs = [config["training"]["lr"]]  + [backbone_lr]
        return params, max_lrs
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @classmethod
    def build(cls, config: Dict[str, Dict[str, Any]]):
        pixel_encoder_img_size = config["model"]["pixel_encoder"]["img_size"]
        pixel_encoder_pretrained = config["model"]["pixel_encoder"].get(
            "pretrained", None
        )
        config_backone = {"img_size": np.array(pixel_encoder_img_size)}
        if pixel_encoder_pretrained is not None:
            config_backone["pretrained"] = pixel_encoder_pretrained
        import importlib

        mod = importlib.import_module("unidac.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        if 'vit' in config["model"]["pixel_encoder"]["name"]:
            pixel_encoder_config = {
                **config["training"],
                **config["data"],
                **config["model"]["pixel_encoder"],
            }
            pixel_encoder = pixel_encoder_factory(pixel_encoder_config)
            rope_lat_weight = pixel_encoder_config.get('rope_lat_weight', False)
        else:
            pixel_encoder = pixel_encoder_factory(**config_backone)
            rope_lat_weight = False
        
        pixel_decoder = DPTHead(**config["model"]["pixel_decoder"])

        try:
            img_H, img_W = config['data']['crop_size']
        except:
            img_H, img_W = config['data']['fwd_sz']

        scale_head = ScaleMapEstimation(
            embed_dim=pixel_encoder.embed_dim,
            patch_size=pixel_encoder.patch_size,
            num_heads=pixel_encoder.num_heads,
            rope_embed=pixel_encoder.rope_embed if hasattr(pixel_encoder, 'rope_embed') else None,
            mask_k_bias=True,
            norm_layer="layernormbf16",
            img_H=img_H,
            img_W=img_W,
            dino_state_dict=pixel_encoder.blocks[-1].state_dict()
        )

        return deepcopy(
            cls(
                pixel_encoder=pixel_encoder,
                pixel_decoder=pixel_decoder,
                rope_lat_weight=rope_lat_weight,
                scale_head=scale_head
            )
        )