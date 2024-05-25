import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from data.dataset import RESIDEDetectionDataset
from models.dsnet import DSNet
import torchvision.transforms as transforms

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def evaluate(model, dataloader, device):
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for hazy_images, clear_images, targets in dataloader:
            hazy_images = [image.to(device) for image in hazy_images]
            clear_images = [image.to(device) for image in clear_images]
            clear_preds, detection_results = model(hazy_images)

            for clear_pred, clear_image in zip(clear_preds, clear_images):
                psnr = calculate_psnr(clear_pred, clear_image)
                psnr_values.append(psnr)

            # 在此处可以添加mAP计算逻辑
            # for output, target in zip(detection_results, targets):
            #     pred_boxes = output['boxes'].cpu()
            #     pred_labels = output['labels'].cpu()
            #     true_boxes = target['boxes'].cpu()
            #     true_labels = target['labels'].cpu()
            #     ious = box_iou(pred_boxes, true_boxes)
            #     # 计算mAP等指标

    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 数据预处理
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 创建验证数据集和数据加载器
val_dataset = RESIDEDetectionDataset(hazy_dir='datasets/RESIDE/ITS/hazy', clear_dir='datasets/RESIDE/ITS/clear', ann_dir='datasets/VOC/Annotations', transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 初始化模型并加载训练好的参数
model = DSNet(num_classes=21)
model.load_state_dict(torch.load('path_to_trained_model.pth'))
model.to(device)

# 评估模型
evaluate(model, val_dataloader, device)
