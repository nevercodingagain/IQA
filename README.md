# 基于Vision Transformer的图像质量评估(IQA)项目

本项目使用Vision Transformer (ViT) 模型对图像质量进行评估，使用KonIQ-10k数据集进行训练和评估。

## 项目结构

- `dataset_utils.py`: 数据集加载和预处理工具
- `models.py`: ViT模型定义，包含基础ViT和带注意力机制的ViT两种模型
- `train.py`: 模型训练脚本
- `evaluate.py`: 模型评估脚本
- `main.py`: 主程序入口

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- scipy

## 数据集

本项目使用KonIQ-10k数据集，该数据集包含10,073张带有主观质量评分的自然图像。

数据集结构：
- 图像文件位于 `dataset/koniq-10k/1024x768/` 目录下
- 标签文件位于 `data/koniq-10k.txt`

## 使用方法

### 训练模型

```bash
python main.py --mode train --model_type vit --batch_size 32 --num_epochs 50 --learning_rate 1e-4
```

可选参数：
- `--model_type`: 选择模型类型，可选 `vit` 或 `vit_attention`
- `--freeze_backbone`: 是否冻结backbone参数
- `--batch_size`: 批大小
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--weight_decay`: 权重衰减

### 评估模型

```bash
python main.py --mode evaluate --model_type vit --model_path outputs/best_model.pth --visualize
```

可选参数：
- `--model_path`: 训练好的模型路径（必需）
- `--visualize`: 是否可视化预测结果

## 模型说明

本项目提供两种模型：

1. **基础ViT模型 (ViTForIQA)**：使用预训练的ViT-B/16作为特征提取器，添加回归头进行质量评分预测。

2. **带注意力机制的ViT模型 (ViTWithAttentionForIQA)**：在基础ViT模型的基础上添加注意力机制，可以更好地关注图像中与质量相关的区域。

## 评估指标

模型使用以下指标评估性能：

- **SRCC (Spearman Rank Correlation Coefficient)**: 衡量预测分数与真实分数之间的单调关系
- **PLCC (Pearson Linear Correlation Coefficient)**: 衡量预测分数与真实分数之间的线性关系
- **RMSE (Root Mean Square Error)**: 衡量预测分数与真实分数之间的误差大小

## 结果可视化

评估时添加 `--visualize` 参数可以生成以下可视化结果：

1. 预测分数与真实分数的散点图
2. 预测误差分布直方图

这些可视化结果保存在 `results/` 目录下。