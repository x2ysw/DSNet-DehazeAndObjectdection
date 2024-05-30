# 定义检测子网

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils import device


def get_detection_net(num_classes):
    # 加载预训练的Faster R-CNN模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换预训练的头部为自定义的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# 初始化检测模型
detection_net = get_detection_net(num_classes=21)
detection_net.to(device)
