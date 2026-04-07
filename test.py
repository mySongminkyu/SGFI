'''

RGBD-only Testing Script

Example:
    python test.py \
        --sam2_cfg sam2.1/sam2.1_hiera_l \
        --data_dir ./data \
        --save_dir ./output/RGBD \
        --sam2_ckpt ./checkpoints/sam2.1_hiera_large.pt \
        --weights ./checkpoints/RGBD_model.pth \
        --use_high_res_features \
        --use_edge_fusion_step1 \
        --use_edge_fusion_step2

'''

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from data import test_dataset
from SAM_Model.model import Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size',  type=int, default=1024)
    parser.add_argument('--data_dir',  type=str, required=True, help='RGBD dataset root')
    parser.add_argument('--save_dir',  type=str, required=True, help='Output directory')
    parser.add_argument('--sam2_cfg',  type=str, required=True, help='Path to sam2.1_hiera_*.yaml')
    parser.add_argument('--sam2_ckpt', type=str, required=True, help='Path to sam2.1_hiera_*.pt')
    parser.add_argument('--weights',   type=str, required=True, help='Trained weights (.pth)')

    parser.add_argument('--use_high_res_features',  action='store_true')
    parser.add_argument('--use_edge_fusion_step1',  action='store_true')
    parser.add_argument('--use_edge_fusion_step2',  action='store_true')
    parser.add_argument('--fusion_type', type=str, default="gated", choices=["gated", "add"])

    parser.add_argument('--lora_rank',         type=int,   default=16)
    parser.add_argument('--lora_alpha',        type=int,   default=32)
    parser.add_argument('--lora_dropout',      type=float, default=0.05)
    parser.add_argument('--lora_target_stages', nargs='+', type=int, default=[0, 1, 2, 3])

    parser.add_argument('--sets', nargs='*', default=None, help='Benchmark sets to evaluate (default: all)')
    parser.add_argument('--fp16', action='store_true', help='FP16 inference')

    args = parser.parse_args()

    from types import SimpleNamespace
    cfg = SimpleNamespace(
        img_size               = args.img_size,
        sam2_cfg               = args.sam2_cfg,
        sam2_ckpt              = args.sam2_ckpt,
        use_high_res_features  = args.use_high_res_features,
        use_edge_fusion_step1  = args.use_edge_fusion_step1,
        use_edge_fusion_step2  = args.use_edge_fusion_step2,
        fusion_type            = args.fusion_type,
        freeze_image_encoder   = True,
        freeze_prompt_encoder  = True,
        freeze_mask_decoder    = False,
        lora_rank              = args.lora_rank,
        lora_alpha             = args.lora_alpha,
        lora_dropout           = args.lora_dropout,
        lora_target_stages     = args.lora_target_stages,
    )

    print('[Info] Building model...')
    model = Model(cfg)

    print(f'[Info] Loading weights: {args.weights}')
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")

    ckpt    = torch.load(args.weights, map_location='cpu')
    state   = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing:
        fusion_missing = [k for k in missing if 'injector' in k]
        if fusion_missing:
            print("[CRITICAL WARNING] Injector modules missing — did you train with --use_edge_fusion_step1/2?")
            print(f"  -> {fusion_missing[:5]}")
        else:
            print(f'[Warn] Missing keys: {len(missing)}')

    if unexpected:
        fusion_unexpected = [k for k in unexpected if 'injector' in k]
        if fusion_unexpected:
            raise RuntimeError(
                "Weights contain Injector modules but model didn't create them. "
                "Did you forget --use_edge_fusion_step1 or --use_edge_fusion_step2?"
            )
        else:
            print(f'[Warn] Unexpected keys: {len(unexpected)}')

    model = model.cuda().eval()

    test_sets = args.sets or ['DES', 'LFSD', 'NJU2K_Test', 'NLPR_Test', 'SIP', 'STERE', 'SSD']
    os.makedirs(args.save_dir, exist_ok=True)

    print('[Info] Start inference...')
    with torch.no_grad():
        for dataset in test_sets:
            save_path = os.path.join(args.save_dir, dataset)
            os.makedirs(save_path, exist_ok=True)

            img_dir   = os.path.join(args.data_dir, 'test_data', 'img',   dataset) + '/'
            gt_dir    = os.path.join(args.data_dir, 'test_data', 'gt',    dataset) + '/'
            depth_dir = os.path.join(args.data_dir, 'test_data', 'depth', dataset) + '/'

            if not os.path.exists(img_dir):
                print(f'[Skip] {dataset} - not found')
                continue

            loader = test_dataset(img_dir, gt_dir, depth_dir, args.img_size)
            print(f'[Processing] {dataset} ({loader.size} images)')

            for _ in range(loader.size):
                image, gt, depth, name = loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                image = image.cuda(non_blocking=True)
                depth = depth.cuda(non_blocking=True)

                if args.fp16:
                    with torch.cuda.amp.autocast():
                        pred_np, _ = model(image, depth, None)
                else:
                    pred_np, _ = model(image, depth, None)

                pred_np = F.interpolate(pred_np, size=gt.shape, mode='bilinear', align_corners=False)
                pred_np = pred_np.sigmoid().cpu().numpy().squeeze()
                pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
                pred_np = (pred_np * 255).astype(np.uint8)

                cv2.imwrite(os.path.join(save_path, name), pred_np)

    print(f'\n[Done] Results saved to: {args.save_dir}')


if __name__ == '__main__':
    main()
