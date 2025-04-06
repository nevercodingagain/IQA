import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from dataset_utils import get_dataloaders
from model.models import ViTForIQA, ViTWithAttentionForIQA, ResNetViTForIQA


def evaluate(config):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 获取测试数据加载器
    dataloaders = get_dataloaders(
        root_dir=config.data_dir,
        label_file=config.label_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    test_loader = dataloaders['test']
    
    # 初始化模型
    if config.model_type == 'vit':
        model = ViTForIQA(pretrained=False)  # 不需要预训练权重，因为会加载训练好的模型
    elif config.model_type == 'vit_attention':
        model = ViTWithAttentionForIQA(pretrained=False)
    elif config.model_type == 'resnet_vit':
        model = ResNetViTForIQA(pretrained=False)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    
    # 加载训练好的模型
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 在测试集上评估
    all_preds = []
    all_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            # 获取数据和标签
            images = batch['image'].to(device)
            targets = batch['mos'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            test_loss += loss.item() * images.size(0)
            
            # 收集预测和目标值
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算平均损失
    test_loss = test_loss / len(test_loader.dataset)
    
    # 计算评估指标
    srcc, _ = spearmanr(all_targets, all_preds)
    plcc, _ = pearsonr(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    # 打印结果
    print(f"测试结果 | Loss: {test_loss:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, RMSE: {rmse:.4f}")
    
    # 可视化预测结果
    if config.visualize:
        visualize_predictions(all_targets, all_preds, config.output_dir)


def visualize_predictions(targets, predictions, output_dir):
    """
    可视化预测结果
    
    参数:
        targets (list): 真实MOS值列表
        predictions (list): 预测MOS值列表
        output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # 添加对角线（理想情况）
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加标题和标签
    plt.title('预测MOS值 vs 真实MOS值')
    plt.xlabel('真实MOS值')
    plt.ylabel('预测MOS值')
    
    # 添加SRCC和PLCC信息
    srcc, _ = spearmanr(targets, predictions)
    plcc, _ = pearsonr(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    plt.text(min_val + 0.1 * (max_val - min_val), max_val - 0.1 * (max_val - min_val),
             f'SRCC: {srcc:.4f}\nPLCC: {plcc:.4f}\nRMSE: {rmse:.4f}')
    
    # 保存图像
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    plt.close()
    
    # 绘制误差分布直方图
    errors = np.array(predictions) - np.array(targets)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.75)
    plt.title('预测误差分布')
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'))
    plt.close()


# 如果直接运行此脚本，使用默认配置进行评估
if __name__ == '__main__':
    from config import get_eval_config
    import sys
    
    if len(sys.argv) < 2:
        print("错误: 需要提供模型路径作为参数")
        print("用法: python evaluate.py <model_path>")
        sys.exit(1)
    
    config = get_eval_config(model_path=sys.argv[1])
    evaluate(config)