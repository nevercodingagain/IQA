import os
import argparse
import torch

from train import train
from evaluate import evaluate
from config import get_train_config, get_eval_config

def main():
    # 创建命令行参数解析器（保留向后兼容性）
    parser = argparse.ArgumentParser(description='ViT图像质量评估项目')
    
    # 通用参数
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],
                        help='运行模式: train或evaluate')
    parser.add_argument('--data_dir', type=str, help='图像数据目录')
    parser.add_argument('--label_file', type=str, help='标签文件路径')
    parser.add_argument('--output_dir', type=str, help='输出目录，用于保存模型和结果')
    parser.add_argument('--model_type', type=str, choices=['vit', 'vit_attention', 'resnet_vit'],
                        help='模型类型: vit、vit_attention或resnet_vit')
    parser.add_argument('--batch_size', type=int, help='批大小')
    parser.add_argument('--num_workers', type=int, help='数据加载的工作线程数')
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')
    
    # 训练特定参数
    parser.add_argument('--num_epochs', type=int, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--freeze_backbone', action='store_true', help='是否冻结backbone参数')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    # 分布式训练参数
    parser.add_argument('--world_size', type=int, help='分布式训练的进程数量，-1表示使用所有可用GPU')
    parser.add_argument('--dist_master_addr', type=str, help='分布式训练的主节点地址')
    parser.add_argument('--dist_master_port', type=str, help='分布式训练的主节点端口')
    parser.add_argument('--gpu_ids', type=str, help='指定要使用的GPU ID列表，例如"0,1,2"，用于设置CUDA_VISIBLE_DEVICES环境变量')
    
    # 评估特定参数
    parser.add_argument('--model_path', type=str, help='训练好的模型路径，仅在evaluate模式下需要')
    parser.add_argument('--visualize', action='store_true', help='是否可视化预测结果，仅在evaluate模式下有效')
    
    args = parser.parse_args()
    
    # 根据模式获取配置
    if args.mode == 'train':
        config = get_train_config()
    elif args.mode == 'evaluate':
        config = get_eval_config()
        if args.model_path is None:
            parser.error("evaluate模式需要指定--model_path参数")
    
    # 从命令行参数更新配置（如果提供）
    config_updates = {}
    for key, value in vars(args).items():
        if key != 'mode' and value is not None:
            if key == 'no_cuda':
                config_updates['use_cuda'] = not value
            else:
                config_updates[key] = value
    
    config.update(**config_updates)
    
    # 设置CUDA_VISIBLE_DEVICES环境变量（如果指定了gpu_ids）
    if hasattr(config, 'gpu_ids') and config.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_ids
        print(f"已设置CUDA_VISIBLE_DEVICES={config.gpu_ids}")
    
    # 检查CUDA可用性
    if config.use_cuda and not torch.cuda.is_available():
        print("警告: 没有可用的CUDA设备，将使用CPU")
        config.use_cuda = False
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 打印配置信息
    print(config)
    
    # 根据模式运行相应的功能
    if config.mode == 'train':
        train(config)
    elif config.mode == 'evaluate':
        evaluate(config)

if __name__ == '__main__':
    main()