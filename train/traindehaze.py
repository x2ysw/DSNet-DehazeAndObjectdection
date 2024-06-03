import torch.optim as optim
import torch.nn as nn
from models.dehazing_net import DehazingNet
from data.predehaze import dehaze_dataloader
import torch

# 显示进度条
from tqdm import tqdm

# 设置设备
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 初始化模型
model = DehazingNet().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

log_file = open('training_log.txt', 'w')

# 训练过程
for epoch in range(num_epochs):

    model.train()
    running_loss = 0.0
    for hazy_images, clear_images in dehaze_dataloader:
        for inputs, labels in tqdm(dehaze_dataloader, desc=f'Epoch {epoch + 1}', total=len(dehaze_dataloader)):
            print("\033c", end="")

            hazy_images = hazy_images.to(device)
            clear_images = clear_images.to(device)

            optimizer.zero_grad()
            outputs = model(hazy_images)
            loss = criterion(outputs, clear_images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    log_file.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dehaze_dataloader):.4f}\n")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dehaze_dataloader):.4f}")


# 关闭日志文件
log_file.close()

# 保存模型参数
torch.save(model.state_dict(), 'dehazing_model.pth')
print("Model saved to dehazing_model.pth")

print("Dehazing training complete")
