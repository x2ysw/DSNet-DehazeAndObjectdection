# 定义DSNet模型

import torch.nn as nn
from .dehazing_net import DehazingNet
from .detection_net import get_detection_net

class DSNet(nn.Module):
    def __init__(self, num_classes):
        super(DSNet, self).__init__()
        self.dehazing_net = DehazingNet()
        self.detection_net = get_detection_net(num_classes)

    def forward(self, hazy_images, clear_images=None, targets=None):
        clear_preds = [self.dehazing_net(image) for image in hazy_images]
        if self.training:
            detection_loss = sum([self.detection_net(clear_pred.unsqueeze(0), [target])
                                  for clear_pred, target in zip(clear_preds, targets)])
            return clear_preds, detection_loss
        else:
            detection_results = [self.detection_net(clear_pred.unsqueeze(0))
                                 for clear_pred in clear_preds]
            return clear_preds, detection_results
