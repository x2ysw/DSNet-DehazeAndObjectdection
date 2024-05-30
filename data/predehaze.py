import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

"""
去雾预处理
"""
class DehazingDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.hazy_images = list(sorted(os.listdir(hazy_dir)))
        self.clear_images = list(sorted(os.listdir(clear_dir)))

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        clear_image_path = os.path.join(self.clear_dir, self.clear_images[idx])

        hazy_image = Image.open(hazy_image_path).convert("RGB")
        clear_image = Image.open(clear_image_path).convert("RGB")

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)
        print("__getitem__ finish")
        return hazy_image, clear_image


# 数据预处理
dehaze_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = "C:\\Users\\DELL\\Desktop\\ITS_v2"
hazy_dir = os.path.join(data_dir, "hazy\\hazy")
clear_dir = os.path.join(data_dir, "clear\\clear")

# 创建去雾数据集和数据加载器
# dehaze_dataset = DehazingDataset(hazy_dir='datasets/RESIDE/ITS/hazy', clear_dir='datasets/RESIDE/ITS/clear',
#                                  transform=dehaze_transform)
dehaze_dataset = DehazingDataset(hazy_dir, clear_dir,
                                 transform=dehaze_transform)
dehaze_dataloader = DataLoader(dehaze_dataset, batch_size=4, shuffle=True)
