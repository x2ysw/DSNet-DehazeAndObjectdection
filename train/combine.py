import torch
import torch.nn as nn
from torch import optim

from model import DehazingNet
from models.detection_net import get_detection_net
from train import train_dataloader
# from train.traindehaze import criterion
from utils import device


class DSNet(nn.Module):
    def __init__(self, num_classes):
        super(DSNet, self).__init__()
        self.dehazing_net = DehazingNet()
        self.detection_net = get_detection_net(num_classes)

    def forward(self, hazy_images, clear_images, targets):
        clear_preds = [self.dehazing_net(hazy_image) for hazy_image in hazy_images]
        detection_loss = self.detection_net(clear_preds, targets)
        return clear_preds, detection_loss


# 初始化DSNet模型
dsnet = DSNet(num_classes=21)
dsnet.to(device)

# 定义优化器
optimizer = optim.Adam(dsnet.parameters(), lr=0.001)
num_epochs = 10

# 联合训练过程
for epoch in range(num_epochs):
    dsnet.train()
    running_loss = 0.0

    for hazy_images, clear_images, targets in train_dataloader:
        hazy_images = [image.to(device) for image in hazy_images]
        clear_images = [image.to(device) for image in clear_images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        clear_preds, detection_loss = dsnet(hazy_images, clear_images, targets)

        # 去雾损失
        dehazing_loss = sum(
            [criterion(clear_pred, clear_image) for clear_pred, clear_image in zip(clear_preds, clear_images)])

        # 总损失
        total_loss = dehazing_loss + sum(detection_loss.values())
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

print("Joint training complete")
