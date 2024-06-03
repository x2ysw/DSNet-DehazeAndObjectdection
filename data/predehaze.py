import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.enabled = False

class DehazingDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, transform=None):
        self.hazy_dir = hazy_dir
        self.clear_dir = clear_dir
        self.transform = transform
        self.hazy_images = list(sorted(os.listdir(hazy_dir)))
        self.clear_images = list(sorted(os.listdir(clear_dir)))

        print("Number of hazy images:", len(self.hazy_images))
        print("Number of clear images:", len(self.clear_images))

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        # 从 hazy image 文件名中提取编号
        hazy_image_name = self.hazy_images[idx]
        clear_image_name = hazy_image_name.split('_')[0] + '.png'

        hazy_image_path = os.path.join(self.hazy_dir, hazy_image_name)
        clear_image_path = os.path.join(self.clear_dir, clear_image_name)

        hazy_image = Image.open(hazy_image_path).convert("RGB")
        clear_image = Image.open(clear_image_path).convert("RGB")

        if self.transform:
            hazy_image = self.transform(hazy_image)
            clear_image = self.transform(clear_image)

        return hazy_image, clear_image

# 数据预处理
dehaze_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 创建去雾数据集和数据加载器
dehaze_dataset = DehazingDataset(
    hazy_dir='C:\\Users\\DELL\\Desktop\\ITS_v2\\hazy\\hazy',
    clear_dir='C:\\Users\\DELL\\Desktop\\ITS_v2\\clear\\clear',
    transform=dehaze_transform
)

dehaze_dataloader = DataLoader(dehaze_dataset, batch_size=4, shuffle=True)
# batch_size=4 改过了更低的值，仍然提示错误，不是batch_size太大导致的显存不足
