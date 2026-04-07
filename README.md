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

conda create -n sgfi python=3.10 -y
conda activate sgfi
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt


## Usage

See each subdirectory for training and testing instructions.
