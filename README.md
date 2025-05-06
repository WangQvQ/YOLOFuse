

# YOLOFuse

**YOLOFuse** 是在 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 框架基础上进行的改进版本，专为多模态目标检测任务设计。该版本新增了对双模态图像输入（RGB + 红外 IR）的支持，在复杂环境下提升检测鲁棒性和准确率，适用于夜间监控、安全巡检、灾难救援等应用场景。

---

## ✨ 项目特点

* 🚀 **双模态输入**：支持同时加载 RGB 和 IR 图像，实现信息融合。
* 🔧 **兼容 YOLOv8 接口**：基于 Ultralytics YOLOv8 修改，使用习惯一致，易于上手。
* 🔍 **模块可扩展**：可灵活替换模态融合方式（如特征级融合、注意力机制等）。
* 📦 **推理/训练全流程支持**：支持训练、验证、推理、导出等完整功能。

---


## 🧩 模态输入说明

请将 RGB 和 IR 图像分别命名一致，并存放于不同目录下，例如：

```
assets/LLVIP/images/190003.jpg
assets/LLVIP/imagesIR/190003.jpg
```

系统将自动匹配相同文件名的 RGB 和 IR 图像进行融合处理。

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
git clone https://github.com/WangQvQ/YOLOFuse.git
cd YOLOFuse
pip install -e .
```

### 2️⃣ 训练模型

```bash
python train_dual.py \
  --data your_dualmodal_data.yaml \
  --imgsz 640 \
  --epochs 100 \
  --weights yolov8n.pt \
  --ch 6
```

### 3️⃣ 推理示例

```bash
python infer_dual.py \
  --weights path/to/your/best.pt \
  --imgsz 640 \
  --conf 0.25
```

---

## 📊 数据集格式

* 数据格式需符合 YOLO 格式（txt 标签），确保 RGB(images) 与 IR(imagesIR) 图像一一对应。
* 标签可统一标注于 RGB 图像下，IR 图像使用相同标签文件名。

```bash
datasets
├── images
│   ├── test
│   └── train
├── imagesIR     # 只需和 images 同级、同层次、同文件名
│   ├── test
│   └── train
└── labels
    ├── test
    └── train
```

---


## AudoDL 镜像 (开箱即用)

```bash
conda activate Ultralytics-RGB-IR
cd YOLOFuse

训练：
python train_dual.py

推理:
python infer_dual.py
```