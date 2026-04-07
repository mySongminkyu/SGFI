# SGFI

PyTorch implementation of "Scribble-Supervised Multi-Modal Salient Object Detection via Semantics-Guided Feature Injection and Prototype Supervision".

This repository covers four modality settings:

- [RGB](./RGB)
- [RGB-D](./RGBD)
- [RGB-T](./RGBT)
- [V-D-T](./VDT)

## Resources

- Datasets: [Google Drive](https://drive.google.com/drive/folders/1t1sYnhiIXSaO-Xu2OpPESK7NLWGpXxqx?usp=drive_link)
- Trained checkpoints: [Google Drive](https://drive.google.com/drive/folders/1bBC0W-sl9r3bLRSATfq5Ie2yeRXXAg9h?usp=drive_link)
- Saliency maps: [Google Drive](https://drive.google.com/drive/folders/1aFDmHOWtE9AHHwceMUiN0M3SD9XYo1qw?usp=drive_link)
- Evaluation code: [Google Drive](https://drive.google.com/drive/folders/1kLwBIKJlMd70NOcfd8_VOjQ9zE5VL8fC?usp=drive_link)
- SAM2 pretrained weights: [Official SAM2 repository](https://github.com/facebookresearch/sam2)

## Environment

```bash
conda create -n sgfi python=3.10 -y
conda activate sgfi
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## Usage

See each subdirectory for training and testing instructions.

RGB/README.md
markdown# SGFI - RGB

## Setup

Place files as follows:
```
checkpoints/
├── sam2.1_hiera_large.pt
└── RGB_model.pth

data/
├── train_data/
│   ├── img/
│   ├── gt/
│   ├── mask/
│   └── gray/
└── test_data/
    ├── DUTS-TE/
    │   ├── images/
    │   └── segmentations/
    └── ...
```

## Training
```bash
python train.py \
    --sam2_cfg sam2.1/sam2.1_hiera_l \
    --sam2_ckpt ./checkpoints/sam2.1_hiera_large.pt \
    --train_RGBD_dir ./data \
    --val_RGBD_dir ./data \
    --val_sets DUTS-TE \
    --save_path ./output \
    --use_high_res_features \
    --use_edge_fusion_step1 \
    --use_edge_fusion_step2
```

## Testing
```bash
python test.py \
    --sam2_cfg sam2.1/sam2.1_hiera_l \
    --data_dir ./data \
    --save_dir ./output \
    --sam2_ckpt ./checkpoints/sam2.1_hiera_large.pt \
    --weights ./checkpoints/RGB_model.pth \
    --use_high_res_features \
    --use_edge_fusion_step1 \
    --use_edge_fusion_step2
```
