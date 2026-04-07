
import torch
from types import SimpleNamespace
from thop import profile, clever_format

from SAM_Model.model import Model

cfg = SimpleNamespace(
    sam2_cfg              = '/data/minkyu/Final/RGBD/SAM_Model/modeling/sam2.1_hiera_l.yaml',
    sam2_ckpt             = '/data/minkyu/Final/sam2.1_hiera_large.pt',
    use_high_res_features = True,
    use_edge_fusion_step1 = True,
    use_edge_fusion_step2 = True,
    fusion_type           = 'gated',
    freeze_mask_decoder   = False,
    lora_rank             = 16,
    lora_alpha            = 32,
    lora_dropout          = 0.05,
    lora_target_stages    = [0, 1, 2, 3],
)

model = Model(cfg).cuda().eval()

# Params
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Params    : {total/1e6:.2f}M")
print(f"Trainable Params: {trainable/1e6:.2f}M")

# FLOPs
dummy_rgb   = torch.randn(1, 3, 1024, 1024).cuda()
dummy_depth = torch.randn(1, 3, 1024, 1024).cuda()

with torch.no_grad():
    flops, params = profile(model, inputs=(dummy_rgb, dummy_depth, None), verbose=False)

flops_str, params_str = clever_format([flops, params], "%.2f")
print(f"FLOPs : {flops_str}")
print(f"Params: {params_str}")