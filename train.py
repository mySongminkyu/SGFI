"""
RGBD-only Training Script

Example:
    python train.py \
        --sam2_ckpt ./checkpoints/sam2.1_hiera_large.pt \
        --sam2_cfg sam2.1/sam2.1_hiera_l \
        --train_RGBD_dir ./data \
        --val_RGBD_dir ./data \
        --save_path ./output \
        --use_high_res_features \
        --use_edge_fusion_step1 \
        --use_edge_fusion_step2 \
        --seed 42

"""

import os
import argparse
import random
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

from SAM_Model.model import Model
from data import get_loader, test_dataset, seed_worker
from lscloss import LocalSaliencyCoherence


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Seed set to {seed}")


parser = argparse.ArgumentParser(description='RGBD Training')
parser.add_argument('--epoch',       type=int,   default=30)
parser.add_argument('--lr',          type=float, default=5e-5)
parser.add_argument('--batchsize',   type=int,   default=1)
parser.add_argument('--img_size',    type=int,   default=1024)
parser.add_argument('--sam2_ckpt',   type=str,   required=True)
parser.add_argument('--sam2_cfg',    type=str,   required=True)
parser.add_argument('--save_path',      type=str, required=True)
parser.add_argument('--train_RGBD_dir', type=str, required=True)
parser.add_argument('--val_RGBD_dir',   type=str, required=True)
parser.add_argument('--val_sets',       type=str, default='DES')
parser.add_argument('--lora_rank',          type=int,   default=16)
parser.add_argument('--lora_alpha',         type=int,   default=32)
parser.add_argument('--lora_dropout',       type=float, default=0.05)
parser.add_argument('--lora_target_stages', type=int, nargs='+', default=[0, 1, 2, 3])
parser.add_argument('--use_high_res_features', action='store_true')
parser.add_argument('--use_edge_fusion_step1', action='store_true')
parser.add_argument('--use_edge_fusion_step2', action='store_true')
parser.add_argument('--fusion_type', type=str, default='gated', choices=['gated', 'add'])
parser.add_argument('--seed', type=int, default=42)
opt = parser.parse_args()


def structure_loss(pred, mask):
    weit  = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def prototype_propagation_loss(pred_map, feature_map, scribble_mask):
    """Prototype-based Feature Propagation Loss (Eq. 8-11 in the paper)"""
    mask_small = F.interpolate(scribble_mask, size=feature_map.shape[-2:], mode='nearest')
    B, C, H, W = feature_map.shape
    loss = 0.0
    valid_batches = 0

    for b in range(B):
        feat = feature_map[b].view(C, -1).permute(1, 0)
        mask = mask_small[b].view(-1)
        fg_indices = (mask > 0.5).nonzero(as_tuple=True)[0]

        if len(fg_indices) > 0:
            fg_feats = feat[fg_indices]
            prototype = fg_feats.mean(dim=0)
            feat_norm = F.normalize(feat, p=2, dim=1)
            proto_norm = F.normalize(prototype.unsqueeze(0), p=2, dim=1)
            sim_map = torch.mm(feat_norm, proto_norm.t()).view(H, W)
            target_map = torch.relu(sim_map)
            loss += F.mse_loss(pred_map[b, 0], target_map.detach())
            valid_batches += 1

    if valid_batches > 0:
        return loss / valid_batches
    else:
        return torch.tensor(0.0, device=feature_map.device, requires_grad=True)


@torch.no_grad()
def eval_and_save(test_loaders, model, epoch, save_dir,
                  best_mae_dict, best_epoch_dict, best_mean_mae, best_mean_epoch):
    model.eval()
    mae_dict = {}

    for dataset_name, loader in test_loaders.items():
        mae_local = 0.0
        for _ in range(loader.size):
            image, gt, depth, name = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True)
            res, _ = model(image, depth, None)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_local += np.abs(res - gt).mean()
        mae_local /= loader.size
        mae_dict[dataset_name] = mae_local

        if (dataset_name not in best_mae_dict) or (mae_local < best_mae_dict[dataset_name]):
            best_mae_dict[dataset_name]   = mae_local
            best_epoch_dict[dataset_name] = epoch
            torch.save(model.state_dict(),
                       str(Path(save_dir) / f'{dataset_name}_best_{epoch}_{mae_local:.6f}.pth'))

    mean_mae = sum(mae_dict.values()) / max(1, len(test_loaders))
    if (best_mean_mae is None) or (mean_mae < best_mean_mae):
        best_mean_mae   = mean_mae
        best_mean_epoch = epoch
        torch.save(model.state_dict(),
                   str(Path(save_dir) / f'mean_best_{epoch}_{mean_mae:.6f}.pth'))

    cur_str  = " | ".join([f"{k}:{v:.4f}" for k, v in mae_dict.items()])
    best_str = " | ".join([f"{k}:{best_mae_dict[k]:.4f}(ep{best_epoch_dict[k]})" for k in best_mae_dict])
    print(f"[VAL]  Epoch {epoch:03d} | mean_mae: {mean_mae:.4f} | {cur_str}")
    print(f"[BEST] mean_best: {best_mean_mae:.4f} (ep{best_mean_epoch}) | {best_str}")
    return best_mae_dict, best_epoch_dict, best_mean_mae, best_mean_epoch


if __name__ == '__main__':
    set_seed(opt.seed)
    print("Start")

    model = Model(opt).cuda()

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(x in n.lower() for x in ["bias", "norm", "lora"]) or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.0}, {"params": no_decay, "weight_decay": 0.0}],
        lr=opt.lr,
    )

    warmup_epochs = max(1, opt.epoch // 10)
    min_lr_ratio  = 0.01

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        t = float(epoch - warmup_epochs) / float(max(1, opt.epoch - warmup_epochs))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    train_loader = get_loader(
        os.path.join(opt.train_RGBD_dir, 'train_data', 'img')   + '/',
        os.path.join(opt.train_RGBD_dir, 'train_data', 'depth') + '/',
        os.path.join(opt.train_RGBD_dir, 'train_data', 'gt')    + '/',
        os.path.join(opt.train_RGBD_dir, 'train_data', 'mask')  + '/',
        os.path.join(opt.train_RGBD_dir, 'train_data', 'gray')  + '/',
        batchsize=opt.batchsize, trainsize=opt.img_size,
        num_workers=0, generator=g, worker_init_fn=seed_worker,
    )

    val_datasets = [s.strip() for s in opt.val_sets.split(',') if s.strip()]
    test_loaders = {
        name: test_dataset(
            os.path.join(opt.val_RGBD_dir, 'test_data', 'img',   name) + '/',
            os.path.join(opt.val_RGBD_dir, 'test_data', 'gt',    name) + '/',
            os.path.join(opt.val_RGBD_dir, 'test_data', 'depth', name) + '/',
            opt.img_size,
        ) for name in val_datasets
    }

    total_step = len(train_loader)
    CE         = torch.nn.BCELoss()
    loss_lsc   = LocalSaliencyCoherence().cuda()
    save_path  = Path(opt.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    best_mae_dict, best_epoch_dict = {}, {}
    best_mean_mae, best_mean_epoch = None, None

    for epoch in range(1, opt.epoch + 1):
        model.train()

        for i, pack in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images, depths, gts, masks, grays, input_point, input_label, name = pack
            images      = images.cuda();      depths      = depths.cuda()
            gts         = gts.cuda();         masks       = masks.cuda()
            input_point = input_point.cuda(); input_label = input_label.cuda()

            SAM_mask1, SAM_mask2 = model(
                    images, depths, {'points': input_point, 'labels': input_label}
                )

            SAM_mask_prob2 = torch.sigmoid(SAM_mask2)
            str_loss  = structure_loss(SAM_mask1, SAM_mask_prob2.detach())
            sal_loss2 = (
                images.size(2) * images.size(3) * images.size(0) / (torch.sum(masks) + 1e-8)
            ) * CE(SAM_mask_prob2 * masks, gts * masks)

            images_       = F.interpolate(images,         scale_factor=0.25, mode="bilinear", align_corners=True)
            result_final_ = F.interpolate(SAM_mask_prob2, scale_factor=0.25, mode="bilinear", align_corners=True)
            lsc_loss2 = loss_lsc(
                result_final_, [{"weight": 1, "xy": 6, "rgb": 0.1}],
                5, {'rgb': images_}, images_.shape[2], images_.shape[3],
            )['loss']

            # Prototype supervision loss (Eq. 11)
            proto_loss = 0.0
            decoder = model.sam.sam_mask_decoder
            if "step1_mask" in decoder.get_mask_maps() and \
               "step1_raw" in decoder.get_step_maps():
                proto_loss = prototype_propagation_loss(
                    decoder.get_mask_maps()["step1_mask"],
                    decoder.get_step_maps()["step1_raw"],
                    gts,
                )

            loss = str_loss + sal_loss2 + lsc_loss2 + 1.0 * proto_loss
            loss.backward()
            optimizer.step()

            if i % 100 == 0 or i == total_step:
                s1_log  = decoder.injector_s1.scale_factor.item() if decoder.injector_s1 else 0.0
                s2_log  = decoder.injector_s2.scale_factor.item() if decoder.injector_s2 else 0.0
                pval    = proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss
                print(
                    f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], '
                    f'Step [{i:04d}/{total_step:04d}], '
                    f'str: {str_loss.item():.4f}, sal: {sal_loss2.item():.4f}, '
                    f'proto: {pval:.4f}, total: {loss.item():.4f} | '
                    f'lr: {optimizer.param_groups[0]["lr"]:.2e} | '
                    f'S1: {s1_log:.2f}, S2: {s2_log:.2f}'
                )

        scheduler.step()
        best_mae_dict, best_epoch_dict, best_mean_mae, best_mean_epoch = eval_and_save(
            test_loaders, model, epoch, save_path,
            best_mae_dict, best_epoch_dict, best_mean_mae, best_mean_epoch,
        )

    torch.save(model.state_dict(), str(save_path / f'final_epoch_{opt.epoch}.pth'))
