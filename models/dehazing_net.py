# 定义去雾子网

import torch
import torch.nn as nn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DehazingNet(nn.Module):
    def __init__(self):
        super(DehazingNet, self).__init__()
        # 定义UNet或其他去雾网络结构
        # 这里使用简单的卷积层作为示例
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 初始化去雾模型
dehazing_net = DehazingNet()
dehazing_net.to(device)
