# 图像质量评估 (IQA) 项目

这是一个基于Vision Transformer (ViT) 的图像质量评估项目，支持单机多GPU分布式训练。

## 功能特点

- 基于ViT的图像质量评估模型
- 支持多种模型架构（ViT、带注意力机制的ViT、ResNet+ViT混合模型）
- 支持单GPU和多GPU分布式训练
- 使用PyTorch DistributedDataParallel (DDP) 实现并行训练
- 支持KonIQ-10k数据集

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchvision
- transformers
- numpy
- pandas
- scipy
- matplotlib
- tqdm

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 选择模型类型

本项目支持三种不同的模型架构，可以通过`--model_type`参数进行选择：

1. **ViT (Vision Transformer)** - 默认模型，纯Transformer架构
   ```bash
   python main.py --mode train --model_type vit
   ```

2. **ViT with Attention** - 带有额外注意力机制的ViT模型，可以更好地关注图像质量相关区域
   ```bash
   python main.py --mode train --model_type vit_attention
   ```

3. **ResNet+ViT** - 结合ResNet和ViT的混合模型，利用CNN提取低级特征，ViT捕获全局关系
   ```bash
   python main.py --mode train --model_type resnet_vit
   ```

> **注意**：数据目录和标签文件路径需要在`config.py`中设置，详见[配置文件说明](#配置文件)

### 单GPU训练

```bash
python main.py --mode train
```

### 多GPU分布式训练

```bash
python main.py --mode train --world_size -1
```

### 指定GPU进行训练

```bash
python main.py --mode train --gpu_ids "0,1" --world_size 2
```

### 评估模型

```bash
python main.py --mode evaluate --model_path /path/to/model.pth --model_type vit
```

> **注意**：评估时需要指定与训练时相同的模型类型

## 分布式训练参数

- `--world_size`: 分布式训练的进程数量，-1表示使用所有可用GPU
- `--dist_master_addr`: 分布式训练的主节点地址，默认为'localhost'
- `--dist_master_port`: 分布式训练的主节点端口，默认为'12355'
- `--gpu_ids`: 指定要使用的GPU ID列表，例如"0,1,2"，用于设置CUDA_VISIBLE_DEVICES环境变量

## 配置文件

本项目使用`config.py`文件集中管理所有配置参数，包括数据路径和标签文件位置。您需要在此文件中设置以下关键参数：

```python
# 在config.py中设置数据目录和标签文件路径
self.data_dir = '../IQA_dataset/koniq10k/1024x768/'  # 修改为您的图像数据目录
self.label_file = 'data/koniq-10k.txt'  # 修改为您的标签文件路径
```

其他可配置参数包括：
- `batch_size`: 训练批大小
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `output_dir`: 输出目录，用于保存模型和结果

您可以通过修改`config.py`文件来调整这些参数，而不需要在命令行中指定。

## 注意事项

1. 分布式训练需要CUDA支持，请确保您的系统已安装CUDA
2. 如果只有一个GPU可用，系统会自动回退到单GPU训练模式
3. 分布式训练时，只有主进程(rank=0)会保存模型和输出日志
4. 为获得最佳性能，建议将batch_size设置为单GPU batch_size的倍数
5. 使用`--gpu_ids`参数可以指定要使用的GPU，这将设置CUDA_VISIBLE_DEVICES环境变量
6. 当使用`--gpu_ids`参数时，`--world_size`参数应该与指定的GPU数量一致
7. 数据目录和标签文件路径需要在`config.py`中设置，不要通过命令行参数传递