import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Single-modality LoRA wrapper for linear layers."""

    def __init__(self, base_linear: nn.Linear, r=16, alpha=32, drop_rate=0.05):
        super().__init__()
        assert r > 0
        self.base = base_linear
        in_features  = base_linear.in_features
        out_features = base_linear.out_features

        self.down    = nn.Linear(in_features, r, bias=False)
        self.up      = nn.Linear(r, out_features, bias=False)
        self.scale   = alpha / r
        self.dropout = nn.Dropout(drop_rate)

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.up(self.down(self.dropout(x))) * self.scale


class DualLoRALinear(nn.Module):
    """
    Dual-LoRA layer for RGB/Depth two-stream adaptation.
    Shares a frozen base linear layer with two independent LoRA branches.
    """

    def __init__(self, base_linear: nn.Linear, r=16, alpha=32, drop_rate=0.05):
        super().__init__()
        assert r > 0
        self.base = base_linear
        in_features  = base_linear.in_features
        out_features = base_linear.out_features

        self.down_rgb   = nn.Linear(in_features, r, bias=False)
        self.up_rgb     = nn.Linear(r, out_features, bias=False)
        self.down_depth = nn.Linear(in_features, r, bias=False)
        self.up_depth   = nn.Linear(r, out_features, bias=False)

        self.scale   = alpha / r
        self.dropout = nn.Dropout(drop_rate)

        for lora_down in [self.down_rgb, self.down_depth]:
            nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
        for lora_up in [self.up_rgb, self.up_depth]:
            nn.init.zeros_(lora_up.weight)

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.modality = "rgb"

    def set_modality(self, modality: str):
        assert modality in ["rgb", "depth"]
        self.modality = modality

    def forward(self, x):
        base_out = self.base(x)
        if self.modality == "rgb":
            lora_out = self.up_rgb(self.down_rgb(self.dropout(x))) * self.scale
        else:
            lora_out = self.up_depth(self.down_depth(self.dropout(x))) * self.scale
        return base_out + lora_out


def apply_dual_lora_to_hiera_trunk(
    trunk: nn.Module,
    r=16,
    alpha=32,
    dropout=0.05,
    target_stages=[0, 1, 2, 3],
):
    """
    Apply Dual-LoRA to the qkv projections of the Hiera trunk.

    Args:
        trunk: Hiera backbone module
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: dropout rate
        target_stages: list of stage indices to apply LoRA
    """
    if not hasattr(trunk, 'blocks'):
        print("[Dual-LoRA] Error: No blocks attribute found")
        return {"total_blocks": 0, "start_idx": 0, "hooked_blocks": 0}

    blocks      = trunk.blocks
    total_blocks = len(blocks)
    stage_ends  = getattr(trunk, 'stage_ends', None)

    if stage_ends is None:
        print("[Dual-LoRA] Warning: No stage_ends, applying to last 50%")
        start_idx      = total_blocks // 2
        target_indices = list(range(start_idx, total_blocks))
    else:
        stage_ranges = []
        prev_end = -1
        for stage_idx, end in enumerate(stage_ends):
            stage_ranges.append((stage_idx, prev_end + 1, end + 1))
            prev_end = end

        target_indices = []
        for stage_idx, start, end in stage_ranges:
            if stage_idx in target_stages:
                target_indices.extend(range(start, end))

        start_idx = min(target_indices) if target_indices else 0

    hooked = 0
    for i in target_indices:
        if i >= total_blocks:
            continue
        blk  = blocks[i]
        attn = getattr(blk, 'attn', None)
        if attn is None:
            continue
        qkv = getattr(attn, 'qkv', None)
        if isinstance(qkv, nn.Linear):
            attn.qkv = DualLoRALinear(qkv, r=r, alpha=alpha, drop_rate=dropout)
            hooked += 1

    print(f"[Dual-LoRA] total={total_blocks}, stages={target_stages}, "
          f"start={start_idx}, hooked={hooked}")

    return {"total_blocks": total_blocks, "start_idx": start_idx, "hooked_blocks": hooked}
