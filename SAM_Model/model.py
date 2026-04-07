"""
RGBD Model
RGB-Depth salient object detection using SAM2 with Dual-LoRA and Gated Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from SAM_Model.build_sam import build_sam2
from .adapter import apply_dual_lora_to_hiera_trunk


class GatingFusion(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, 2, 1, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, rgb_emb, depth_emb):
        concat = torch.cat([rgb_emb, depth_emb], dim=1)
        weights = self.gate(concat)
        w_rgb, w_depth = weights[:, 0:1, ...], weights[:, 1:2, ...]
        return w_rgb * rgb_emb + w_depth * depth_emb


class AddFusion(nn.Module):
    def forward(self, rgb_emb, depth_emb):
        return rgb_emb + depth_emb


class Model(nn.Module):
    def __init__(self, cfg, embed_dim: int = 256):
        super().__init__()
        self.cfg = cfg
        self.use_high_res = getattr(cfg, "use_high_res_features", True)

        # Build SAM2
        self.sam = build_sam2(
            config_file=getattr(cfg, "sam2_cfg", "sam2.1/sam2.1_hiera_l.yaml"),
            ckpt_path=getattr(cfg, "sam2_ckpt", None),
            device="cuda",
            mode="eval",
            apply_postprocessing=False,
            use_high_res_features=getattr(cfg, "use_high_res_features", True),
            use_edge_fusion_step1=getattr(cfg, "use_edge_fusion_step1", False),
            use_edge_fusion_step2=getattr(cfg, "use_edge_fusion_step2", False),
        )

        # Image Encoder Projection
        with torch.no_grad():
            dev = next(self.sam.parameters()).device
            dummy = torch.zeros(1, 3, 64, 64, device=dev)
            enc_out = self.sam.image_encoder(dummy)
            c_raw = enc_out["vision_features"].shape[1]

        self.neck_rgb   = nn.Identity() if c_raw == embed_dim else nn.Conv2d(c_raw, embed_dim, 1, bias=False)
        self.neck_depth = nn.Identity() if c_raw == embed_dim else nn.Conv2d(c_raw, embed_dim, 1, bias=False)

        # FPN Projection Layers
        if self.use_high_res:
            self.hi_proj1_rgb   = nn.Conv2d(embed_dim, 64, 1, bias=False)
            self.hi_proj1_depth = nn.Conv2d(embed_dim, 64, 1, bias=False)
            self.hi_proj0_rgb   = nn.Conv2d(embed_dim, 32, 1, bias=False)
            self.hi_proj0_depth = nn.Conv2d(embed_dim, 32, 1, bias=False)

        # Freeze backbone and prompt encoder
        for p in self.sam.image_encoder.parameters():    p.requires_grad_(False)
        for p in self.sam.sam_prompt_encoder.parameters(): p.requires_grad_(False)
        if getattr(cfg, "freeze_mask_decoder", False):
            for p in self.sam.sam_mask_decoder.parameters(): p.requires_grad_(False)

        # Apply Dual-LoRA
        apply_dual_lora_to_hiera_trunk(
            self.sam.image_encoder.trunk,
            r=getattr(cfg, "lora_rank", 16),
            alpha=getattr(cfg, "lora_alpha", 32),
            dropout=getattr(cfg, "lora_dropout", 0.05),
            target_stages=getattr(cfg, "lora_target_stages", [0, 1, 2, 3]),
        )

        # Fusion Layers
        self.fusion_type = getattr(cfg, "fusion_type", "gated")
        if self.fusion_type == "gated":
            self.fuse = GatingFusion(dim=embed_dim)
            if self.use_high_res:
                self.fuse_hi1 = GatingFusion(dim=64)
                self.fuse_hi0 = GatingFusion(dim=32)
        else:
            self.fuse = AddFusion()
            if self.use_high_res:
                self.fuse_hi1 = AddFusion()
                self.fuse_hi0 = AddFusion()

        # Unlock trainable parameters
        for m in [self.neck_rgb, self.neck_depth]:
            for p in m.parameters(): p.requires_grad_(True)

        if self.use_high_res:
            for m in [self.hi_proj0_rgb, self.hi_proj0_depth,
                      self.hi_proj1_rgb, self.hi_proj1_depth,
                      self.fuse_hi0, self.fuse_hi1]:
                for p in m.parameters(): p.requires_grad_(True)

        decoder = self.sam.sam_mask_decoder
        if hasattr(decoder, 'injector_s1') and decoder.injector_s1 is not None:
            for p in decoder.injector_s1.parameters(): p.requires_grad_(True)
            print("[Model] Unlocked Injector S1 parameters.")
        if hasattr(decoder, 'injector_s2') and decoder.injector_s2 is not None:
            for p in decoder.injector_s2.parameters(): p.requires_grad_(True)
            print("[Model] Unlocked Injector S2 parameters.")

        self._fpn_rgb = None
        self._fpn_depth = None
        self._print_param_count()

    def _print_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] total={total:,} | trainable={train:,} ({100.0 * train / max(total, 1):.4f}%)")

    @staticmethod
    def _ensure_3ch(x):
        return x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)

    def _set_lora_modality(self, modality: str):
        for blk in self.sam.image_encoder.trunk.blocks:
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv') and hasattr(blk.attn.qkv, 'set_modality'):
                blk.attn.qkv.set_modality(modality)

    def _encode(self, images, modality):
        images = self._ensure_3ch(images)
        self._set_lora_modality(modality)
        out = self.sam.image_encoder(images)
        fpn = out.get("backbone_fpn", None)
        if modality == "rgb":
            self._fpn_rgb = fpn
        else:
            self._fpn_depth = fpn
        return out["vision_features"]

    def _encode_rgb(self, images):   return self.neck_rgb(self._encode(images, "rgb"))
    def _encode_depth(self, depths): return self.neck_depth(self._encode(depths, "depth"))

    def _decode(self, img_emb, prompts=None):
        points_tuple = None
        if isinstance(prompts, dict):
            if 'points' in prompts and 'labels' in prompts:
                points_tuple = (prompts['points'], prompts['labels'])
        elif isinstance(prompts, tuple):
            points_tuple = prompts

        sparse, dense = self.sam.sam_prompt_encoder(points=points_tuple, boxes=None, masks=None)
        img_pe = self.sam.sam_prompt_encoder.get_dense_pe()
        if img_pe.shape[-2:] != img_emb.shape[-2:]:
            img_pe = F.interpolate(img_pe, img_emb.shape[-2:], mode="bilinear", align_corners=False)

        f_hi = None
        if self.use_high_res and self._fpn_rgb is not None and self._fpn_depth is not None:
            f1 = self.fuse_hi1(self.hi_proj1_rgb(self._fpn_rgb[1]),   self.hi_proj1_depth(self._fpn_depth[1]))
            f0 = self.fuse_hi0(self.hi_proj0_rgb(self._fpn_rgb[0]),   self.hi_proj0_depth(self._fpn_depth[0]))
            f_hi = (f0, f1)

        repeat_image = (sparse is not None and sparse.shape[0] != img_emb.shape[0])
        masks_logits, *_ = self.sam.sam_mask_decoder(
            image_embeddings=img_emb,
            image_pe=img_pe,
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=repeat_image,
            multimask_output=False,
            high_res_features=f_hi,
        )
        return masks_logits

    def forward(self, images, depths, point_prompts=None):
        rgb_emb = self._encode_rgb(images)
        dep_emb = self._encode_depth(depths)
        img_emb = self.fuse(rgb_emb, dep_emb)

        masks_no_prompt = self._decode(img_emb, prompts=None)
        masks_prompt    = self._decode(img_emb, prompts=point_prompts)

        up = lambda x: F.interpolate(x, (images.shape[-2], images.shape[-1]), mode="bilinear", align_corners=False)
        return up(masks_no_prompt), up(masks_prompt)
