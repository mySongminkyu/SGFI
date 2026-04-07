# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import torch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig, ListConfig


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny":        ("sam2/sam2_hiera_t.yaml",    "sam2_hiera_tiny.pt"),
    "facebook/sam2-hiera-small":       ("sam2/sam2_hiera_s.yaml",    "sam2_hiera_small.pt"),
    "facebook/sam2-hiera-base-plus":   ("sam2/sam2_hiera_b+.yaml",   "sam2_hiera_base_plus.pt"),
    "facebook/sam2-hiera-large":       ("sam2/sam2_hiera_l.yaml",    "sam2_hiera_large.pt"),
    "facebook/sam2.1-hiera-tiny":      ("sam2.1/sam2.1_hiera_t.yaml",  "sam2.1_hiera_tiny.pt"),
    "facebook/sam2.1-hiera-small":     ("sam2.1/sam2.1_hiera_s.yaml",  "sam2.1_hiera_small.pt"),
    "facebook/sam2.1-hiera-base-plus": ("sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "facebook/sam2.1-hiera-large":     ("sam2.1/sam2.1_hiera_l.yaml",  "sam2.1_hiera_large.pt"),
}


def _retarget_cfg(cfg, src_prefix="sam2.", dst_prefix="SAM_Model."):
    """Retarget Hydra config _target_ fields from sam2.* to SAM_Model.*"""
    def _fix(s: str) -> str:
        if s.startswith(src_prefix):
            s = dst_prefix + s[len(src_prefix):]
        s = s.replace("SAM_Model.modeling.backbones.", "SAM_Model.modeling.")
        s = s.replace("SAM_Model.modeling.sam.transformer.", "SAM_Model.modeling.transformer.")
        return s

    def _rec(node):
        if isinstance(node, DictConfig):
            if "_target_" in node and isinstance(node["_target_"], str):
                node["_target_"] = _fix(node["_target_"])
            for k in node.keys():
                _rec(node[k])
        elif isinstance(node, ListConfig):
            for i in range(len(node)):
                _rec(node[i])

    _rec(cfg)
    return cfg


def build_sam2(
    config_file: str,
    ckpt_path: str | None = None,
    device: str = "cuda",
    mode: str = "eval",
    hydra_overrides_extra: list[str] | None = None,
    apply_postprocessing: bool = True,
    **kwargs,
):
    """
    Build a SAM2 model from a config file and optionally load a checkpoint.

    Args:
        config_file: absolute path or path relative to ./configs (e.g., "sam2.1/sam2.1_hiera_l.yaml")
        ckpt_path: path to *.pt checkpoint (optional)
        device: "cuda" or "cpu"
        mode: "eval" or "train"
        hydra_overrides_extra: additional Hydra override strings
        apply_postprocessing: enable stability-based multimask fallback
        **kwargs: use_high_res_features, use_edge_fusion_step1, use_edge_fusion_step2
    """
    overrides = list(hydra_overrides_extra or [])
    if apply_postprocessing:
        overrides += [
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        ]

    for key, value in kwargs.items():
        value_str = str(value).lower() if isinstance(value, bool) else str(value)
        if key in ["use_edge_fusion_step1", "use_edge_fusion_step2"]:
            overrides.append(f"++model.sam_mask_decoder_extra_args.{key}={value_str}")
            print(f"[build_sam2] Override: model.sam_mask_decoder_extra_args.{key}={value_str}")
        elif key == "use_high_res_features":
            overrides.append(f"++model.use_high_res_features_in_sam={value_str}")
            print(f"[build_sam2] Override: model.use_high_res_features_in_sam={value_str}")

    if os.path.isabs(config_file):
        cfg_dir  = os.path.dirname(config_file)
        cfg_name = os.path.basename(config_file)
        if cfg_name.endswith(".yaml"):
            cfg_name = cfg_name[:-5]
        with initialize_config_dir(config_dir=cfg_dir, job_name="sam2"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
    else:
        CONFIG_ROOT = os.path.join(os.path.dirname(__file__), "configs")
        cfg_name = config_file.split("configs/", 1)[-1]
        if cfg_name.endswith(".yaml"):
            cfg_name = cfg_name[:-5]
        with initialize_config_dir(config_dir=CONFIG_ROOT, job_name="sam2"):
            cfg = compose(config_name=cfg_name, overrides=overrides)

    OmegaConf.resolve(cfg)
    cfg = _retarget_cfg(cfg)
    model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, ckpt_path)

    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _hf_download(model_id: str):
    from huggingface_hub import hf_hub_download
    cfg_rel, ckpt_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=ckpt_name)
    return cfg_rel, ckpt_path


def build_sam2_hf(model_id: str, **kwargs):
    cfg_rel, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=cfg_rel, ckpt_path=ckpt_path, **kwargs)


def _load_checkpoint(model, ckpt_path: str | None):
    if not ckpt_path:
        return
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    missing_keys, unexpected_keys = model.load_state_dict(sd, strict=False)

    if missing_keys:
        logging.warning("[SAM2] Missing keys (newly added modules will be randomly initialized):")
        for k in missing_keys:
            logging.warning(f"    - {k}")

    if unexpected_keys:
        logging.warning("[SAM2] Unexpected keys in checkpoint (ignored):")
        for k in unexpected_keys:
            logging.warning(f"    - {k}")

    logging.info("[SAM2] Loaded checkpoint with non-strict mode.")
