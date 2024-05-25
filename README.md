# DSNet-DehazeAndObjectdection
dehaze and object dectection

# 项目结构
```
dsnet_project/
├── data/
│   ├── prepare_data.py    # 数据准备脚本
│   ├── dataset.py         # 自定义数据集类
│   └── transforms.py      # 数据增强和变换
├── models/
│   ├── dsnet.py           # DSNet模型定义
│   ├── dehazing_net.py    # 去雾子网络定义
│   └── detection_net.py   # 目标检测网络定义
├── train.py               # 训练脚本
├── evaluate.py            # 评估脚本
├── utils.py               # 辅助函数
└── requirements.md       # 所需依赖库

```
