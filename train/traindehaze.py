import torch.optim as optim
import torch.nn as nn
from models import dehazing_net
from models.dehazing_net import device
from data.predehaze import dehaze_dataloader
"""
    去雾训练
"""

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(dehazing_net.parameters(), lr=0.001)
num_epochs = 10

# 训练过程
for epoch in range(num_epochs):
    dehazing_net.train()
    running_loss = 0.0

    for hazy_images, clear_images in dehaze_dataloader:
        hazy_images = hazy_images.to(device)
        clear_images = clear_images.to(device)

        optimizer.zero_grad()
        outputs = dehazing_net(hazy_images)
        loss = criterion(outputs, clear_images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dehaze_dataloader):.4f}")

print("Dehazing training complete")
