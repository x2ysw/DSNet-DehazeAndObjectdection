# 定义优化器
from torch.distributed import optim

from data.predetect import detection_dataloader
from models import detection_net
from utils import device

optimizer = optim.Adam(detection_net.parameters(), lr=0.001)
num_epochs = 10

# 训练过程
for epoch in range(num_epochs):
    detection_net.train()
    running_loss = 0.0

    for images, targets in detection_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = detection_net(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(detection_dataloader):.4f}")

print("Detection training complete")
