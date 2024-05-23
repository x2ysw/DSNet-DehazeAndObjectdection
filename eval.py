# 模型验证评估

import numpy as np
import math
from torchvision.ops import box_iou

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def evaluate(model, dataloader, device):
    model.eval()
    psnr_values = []
    with torch.no_grad():
        for hazy_images, clear_images, targets in dataloader:
            hazy_images = list(image.to(device) for image in hazy_images)
            clear_images = list(image.to(device) for image in clear_images)
            clear_preds, detection_results = model(hazy_images)

            for i, (clear_pred, clear_image) in enumerate(zip(clear_preds, clear_images)):
                psnr = calculate_psnr(clear_pred, clear_image)
                psnr_values.append(psnr)

            # 在此处可以添加mAP计算逻辑

    avg_psnr = np.mean(psnr_values)
    print(f"Average PSNR: {avg_psnr:.2f} dB")

# 评估模型
evaluate(model, val_dataloader, device)
