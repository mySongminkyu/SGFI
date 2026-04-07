# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from .sam2_utils import LayerNorm2d, MLP
from .fusion_modules import SemanticInjector


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        dynamic_multimask_via_stability: bool = False,
        dynamic_multimask_stability_delta: float = 0.05,
        dynamic_multimask_stability_thresh: float = 0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        use_edge_fusion_step1: bool = False,
        use_edge_fusion_step2: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token    = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens  = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.use_high_res_features  = use_high_res_features
        self.use_edge_fusion_step1  = use_edge_fusion_step1
        self.use_edge_fusion_step2  = use_edge_fusion_step2

        self.injector_s1 = None
        self.injector_s2 = None

        if self.use_high_res_features and self.use_edge_fusion_step1:
            self.injector_s1 = SemanticInjector(transformer_dim // 4)
        if self.use_high_res_features and self.use_edge_fusion_step2:
            self.injector_s2 = SemanticInjector(transformer_dim // 8)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
             for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_multimask_outputs + 1,
            iou_head_depth, sigmoid_output=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        self.dynamic_multimask_via_stability    = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta  = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

        self._step_maps = {}
        self._mask_maps = {}

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image, high_res_features=None):
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, repeat_image, high_res_features,
        )

        if multimask_output:
            masks    = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks    = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]
        else:
            sam_tokens_out = mask_tokens_out[:, 0:1]

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, repeat_image, high_res_features=None):
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)

        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src     = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out  = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)

        self._step_maps = {}
        self._mask_maps = {}

        dc1, ln1, act1, dc2, act2 = self.output_upscaling

        has_hr = self.use_high_res_features and (high_res_features is not None)
        if has_hr:
            feat_s0, feat_s1 = high_res_features
        else:
            feat_s0, feat_s1 = None, None

        # Step 1: upsample 64 -> 128 (OSI)
        step1 = dc1(src)
        self._step_maps["step1_raw"] = step1.detach()

        if has_hr:
            if self.injector_s1 is not None:
                step1, mask1 = self.injector_s1(low_res=step1, high_res=feat_s1)
                self._mask_maps["step1_mask"] = mask1
            else:
                step1 = step1 + feat_s1

        up_after1 = act1(ln1(step1))

        # Step 2: upsample 128 -> 256 (BSI)
        step2 = dc2(up_after1)
        self._step_maps["step2_raw"] = step2.detach()

        if has_hr:
            if self.injector_s2 is not None:
                step2, mask2 = self.injector_s2(low_res=step2, high_res=feat_s0)
                self._mask_maps["step2_mask"] = mask2
            else:
                step2 = step2 + feat_s0

        upscaled_embedding = act2(step2)

        hyper_in_list = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def get_step_maps(self):
        return self._step_maps

    def get_mask_maps(self):
        return self._mask_maps

    def _get_stability_scores(self, mask_logits):
        mask_logits     = mask_logits.flatten(-2)
        stability_delta = self.dynamic_multimask_stability_delta
        area_i = torch.sum(mask_logits >  stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        multimask_logits     = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds     = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds           = torch.arange(multimask_iou_scores.size(0), device=all_iou_scores.device)
        best_multimask_logits     = multimask_logits[batch_inds, best_scores_inds].unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds].unsqueeze(1)

        singlemask_logits     = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores      = self._get_stability_scores(singlemask_logits)
        is_stable             = stability_scores >= self.dynamic_multimask_stability_thresh

        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits, best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores, best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out