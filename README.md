# 图像质量评估 (IQA) 项目

这是一个基于Vision Transformer (ViT) 的图像质量评估项目，支持单机多GPU分布式训练。

## 功能特点

- 基于ViT的图像质量评估模型
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

### 单GPU训练

```bash
python main.py --mode train --data_dir /path/to/images --label_file /path/to/labels.txt
```

### 多GPU分布式训练

```bash
python main.py --mode train --data_dir /path/to/images --label_file /path/to/labels.txt --world_size -1
```

### 指定GPU进行训练

```bash
    python main.py --mode train --data_dir /path/to/images --label_file /path/to/labels.txt --gpu_ids "0,1" --world_size 2
```

### 评估模型

```bash
python main.py --mode evaluate --data_dir /path/to/images --label_file /path/to/labels.txt --model_path /path/to/model.pth
```

## 分布式训练参数

- `--world_size`: 分布式训练的进程数量，-1表示使用所有可用GPU
- `--dist_master_addr`: 分布式训练的主节点地址，默认为'localhost'
- `--dist_master_port`: 分布式训练的主节点端口，默认为'12355'
- `--gpu_ids`: 指定要使用的GPU ID列表，例如"0,1,2"，用于设置CUDA_VISIBLE_DEVICES环境变量

## 注意事项

1. 分布式训练需要CUDA支持，请确保您的系统已安装CUDA
2. 如果只有一个GPU可用，系统会自动回退到单GPU训练模式
3. 分布式训练时，只有主进程(rank=0)会保存模型和输出日志
4. 为获得最佳性能，建议将batch_size设置为单GPU batch_size的倍数
5. 使用`--gpu_ids`参数可以指定要使用的GPU，这将设置CUDA_VISIBLE_DEVICES环境变量
6. 当使用`--gpu_ids`参数时，`--world_size`参数应该与指定的GPU数量一致