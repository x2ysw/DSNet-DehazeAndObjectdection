# 定义整个模型的训练逻辑的文件，包括命令行参数读取（利用argparse）,数据读取(Dataloader)，模型训练，损失与梯度回传，打印训练信息和保存模型
# 模型训练

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataset import RESIDEDetectionDataset
from models.dsnet import DSNet
import torchvision.transforms as transforms

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# 创建数据集和数据加载器
train_dataset = RESIDEDetectionDataset(hazy_dir='datasets/RESIDE/ITS/hazy', clear_dir='datasets/RESIDE/ITS/clear',
                                       ann_dir='datasets/VOC/Annotations', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# 初始化模型
model = DSNet(num_classes=21)
model.to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# 训练过程
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for hazy_images, clear_images, targets in train_dataloader:
        hazy_images = [image.to(device) for image in hazy_images]
        clear_images = [image.to(device) for image in clear_images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        clear_preds, detection_loss = model(hazy_images, clear_images, targets)

        # 去雾损失（例如MSE损失）
        dehazing_loss = sum([nn.MSELoss()(clear_pred, clear_image)
                             for clear_pred, clear_image in zip(clear_preds, clear_images)])

        # 总损失
        loss = dehazing_loss + detection_loss['loss_classifier'] + detection_loss['loss_box_reg'] + detection_loss[
            'loss_objectness'] + detection_loss['loss_rpn_box_reg']
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

print("Training complete")

##----------------------------------------------
# import torch.optim as optim
# from torch.utils.data import dataloader
# import torch.nn as nn
# import torch
#
# from model import DSNet
#
# model = DSNet(num_classes=21)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device)
#
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 10
#
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#
#     for hazy_images, clear_images, targets in dataloader:
#         hazy_images = list(image.to(device) for image in hazy_images)
#         clear_images = list(image.to(device) for image in clear_images)
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#
#         optimizer.zero_grad()
#         clear_pred, detection_loss = model(hazy_images, clear_images, targets)
#
#         # 去雾损失（例如MSE损失）
#         dehazing_loss = nn.MSELoss()(clear_pred, clear_images)
#
#         # 总损失
#         loss = dehazing_loss + detection_loss['loss_classifier'] + detection_loss['loss_box_reg'] + detection_loss[
#             'loss_objectness'] + detection_loss['loss_rpn_box_reg']
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
#
# print("Training complete")
