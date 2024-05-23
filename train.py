# 定义整个模型的训练逻辑的文件，包括命令行参数读取（利用argparse）,数据读取(Dataloader)，模型训练，损失与梯度回传，打印训练信息和保存模型
# 模型训练

import torch.optim as optim

model = DSNet(num_classes=21)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for hazy_images, clear_images, targets in dataloader:
        hazy_images = list(image.to(device) for image in hazy_images)
        clear_images = list(image.to(device) for image in clear_images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        clear_pred, detection_loss = model(hazy_images, clear_images, targets)

        # 去雾损失（例如MSE损失）
        dehazing_loss = nn.MSELoss()(clear_pred, clear_images)

        # 总损失
        loss = dehazing_loss + detection_loss['loss_classifier'] + detection_loss['loss_box_reg'] + detection_loss[
            'loss_objectness'] + detection_loss['loss_rpn_box_reg']
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training complete")
