import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms


class DetectionDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.images = list(sorted(os.listdir(img_dir)))
        self.annotations = list(sorted(os.listdir(ann_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        image = Image.open(img_path).convert("RGB")

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
            image = self.transform(image)

        return image, target


# 数据预处理
detection_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建目标检测数据集和数据加载器
detection_dataset = DetectionDataset(img_dir='datasets/VOC/JPEGImages', ann_dir='datasets/VOC/Annotations',
                                     transform=detection_transform)
detection_dataloader = DataLoader(detection_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
