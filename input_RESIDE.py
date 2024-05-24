# 数据集输入
import os
import cv2
import numpy as np

def load_reside_dataset(data_dir, resize_shape=(256, 256)):
    """
    加载 RESIDE 数据集，并进行简单的预处理
    Args:
    - data_dir: RESIDE 数据集的根目录
    - resize_shape: 调整图像大小的目标尺寸，默认为 (256, 256)

    Returns:
    - haze_images: 雾天图像数组
    - clear_images: 非雾天图像数组
    """

    haze_images = []
    clear_images = []

    haze_dir = os.path.join(data_dir, "hazy")
    clear_dir = os.path.join(data_dir, "gt")

    haze_files = os.listdir(haze_dir)
    clear_files = os.listdir(clear_dir)

    for haze_file, clear_file in zip(haze_files, clear_files):
        haze_path = os.path.join(haze_dir, haze_file)
        clear_path = os.path.join(clear_dir, clear_file)

        # 读取雾天图像和非雾天图像
        haze_image = cv2.imread(haze_path)
        clear_image = cv2.imread(clear_path)

        # 调整图像大小
        haze_image = cv2.resize(haze_image, resize_shape)
        clear_image = cv2.resize(clear_image, resize_shape)

        # 将图像添加到列表中
        haze_images.append(haze_image)
        clear_images.append(clear_image)

    return np.array(haze_images), np.array(clear_images)

# 设置 RESIDE 数据集的根目录
reside_data_dir = "C:\\Users\DELL\Desktop\SOTS\SOTS\outdoor\outdoor"


# 加载 RESIDE 数据集
haze_images, clear_images = load_reside_dataset(reside_data_dir)

# 打印数据集的形状
print("雾天图像数组形状：", haze_images.shape)
print("非雾天图像数组形状：", clear_images.shape)


load_reside_dataset(reside_data_dir)

print("雾天图像数组形状：", haze_images.shape)
print("非雾天图像数组形状：", clear_images.shape)