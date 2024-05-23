# 预处理数据
# 将雾天图像与对应的清晰图像以及目标检测标签进行预处理

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import xml.etree.ElementTree as ET


class RESIDEDetectionDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, ann_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.hazy_images = list(sorted(os.listdir(hazy_dir)))
        self.clear_images = list(sorted(os.listdir(clear_dir)))
        self.annotations = list(sorted(os.listdir(ann_dir)))

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clear_image_path = os.path.join(self.clear_dir, self.clear_images[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        hazy_image = Image.open(hazy_image_path).convert("RGB")
        clear_image = Image.open(clear_image_path).convert("RGB")

        boxes = []
        labels = []

        tree = ET.parse(ann_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)  # 确保标签转换为数值格式

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)  # 确保标签为int64类型

        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return hazy_image, clear_image, target


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

dataset = RESIDEDetectionDataset(hazy_dir='datasets/RESIDE/ITS/hazy', clear_dir='datasets/RESIDE/ITS/clear',
                                 ann_dir='datasets/VOC/Annotations', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
