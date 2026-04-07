import os
from PIL import Image
import torch.utils.data as data
import numpy as np
import torch
import torchvision.transforms as transforms
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_prompt(x, y, points_num, imgsize):
    point_list = []
    label_list = []
    t = 0
    while len(point_list) < points_num and t < 10000:
        t = t + 1
        random_x = np.random.randint(0, imgsize)
        random_y = np.random.randint(0, imgsize)
        x_value = x.getpixel((random_x, random_y))
        y_value = y.getpixel((random_x, random_y))
  
        if x_value == 255 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(1)
        elif x_value == 0 and y_value == 255:
            point_list.append([random_x, random_y])
            label_list.append(0)

    if len(point_list) < 1:
        point_list.append([0, 0])
        label_list.append(-1)

    input_point = torch.tensor(point_list)
    input_label = torch.tensor(label_list)
    return input_point, input_label


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, depth_root, gt_root, mask_root, gray_root, trainsize):
        self.trainsize = trainsize
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.depths = sorted([os.path.join(depth_root, f) for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.masks = sorted([os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.grays = sorted([os.path.join(gray_root, f) for f in os.listdir(gray_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.size = len(self.images)

        self.resize_transform = transforms.Resize((self.trainsize, self.trainsize))
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.up = transforms.Resize((self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST)
        self.points_num = 10

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        depth = self.rgb_loader(self.depths[index])
        gt = self.binary_loader(self.gts[index])
        mask = self.binary_loader(self.masks[index])
        gray = self.binary_loader(self.grays[index])

        gt_map = self.up(gt)
        mask_map = self.up(mask)
        input_point, input_label = get_prompt(gt_map, mask_map, self.points_num, self.trainsize)

        image = self.normalize_transform(self.to_tensor_transform(self.resize_transform(image)))
        depth = self.to_tensor_transform(self.resize_transform(depth))
        gt = self.to_tensor_transform(self.resize_transform(gt))
        mask = self.to_tensor_transform(self.resize_transform(mask))
        gray = self.to_tensor_transform(self.resize_transform(gray))

        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'): name = name.split('.jpg')[0] + '.png'

        return image, depth, gt, mask, gray, input_point, input_label, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# [수정] worker_init_fn 매개변수 추가
def get_loader(image_root, depth_root, gt_root, mask_root, gray_root, batchsize, trainsize,
               shuffle=True, num_workers=0, pin_memory=True, generator=None, worker_init_fn=None):
    dataset = SalObjDataset(image_root, depth_root, gt_root, mask_root, gray_root, trainsize)
    data_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=batchsize, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        generator=generator,      # 제너레이터 전달
        worker_init_fn=worker_init_fn # 전달받은 워커 시드 초기화 함수 사용
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = sorted([image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.gts = sorted([gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')])
        self.depths = sorted([depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp') or f.endswith('.jpg') or f.endswith('.png')])
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        depth = self.rgb_loader(self.depths[self.index])
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'): name = name.split('.jpg')[0] + '.png'
        self.index = (self.index + 1) % self.size
        return image, gt, depth, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')