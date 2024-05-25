# 定义去雾子网

import torch.nn as nn

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
