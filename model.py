# 主模型 数据输入 模型输出
# 定义去雾子网和目标检测网
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn
import torch

class DehazingNet(nn.Module):
    def __init__(self):
        super(DehazingNet, self).__init__()
        # 定义去雾子网络（例如UNet架构）
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 添加更多层...
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Tanh()
            # 添加更多层...
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DSNet(nn.Module):
    def __init__(self, num_classes):
        super(DSNet, self).__init__()
        self.dehazing_net = DehazingNet()
        self.detection_net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.detection_net.roi_heads.box_predictor.cls_score.in_features
        self.detection_net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, hazy_image, clear_image=None, target=None):
        clear_pred = self.dehazing_net(hazy_image)
        if self.training:
            detection_loss = self.detection_net(clear_pred, target)
            return clear_pred, detection_loss
        else:
            detection_result = self.detection_net(clear_pred)
            return clear_pred, detection_result
