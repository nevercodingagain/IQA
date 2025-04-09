import os
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from dataset_utils import get_dataloaders
from model.models import ViTForIQA, ResNetViTForIQA, ResNetViTConcatForIQA, SwinForIQA


def evaluate(config, logger=None):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"使用设备: {device}")
    if logger:
        logger.info(f"使用设备: {device}")
    
    # 获取验证数据加载器
    dataloaders = get_dataloaders(
        root_dir=config.data_dir,
        label_file=config.label_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    val_loader = dataloaders['val']
    
    if logger:
        logger.info(f"验证数据集大小: {len(val_loader.dataset)}")
        logger.info(f"批大小: {config.batch_size}")
        logger.info(f"总批次数: {len(val_loader)}")
        logger.info(f"标签文件: {config.label_file}")
        logger.info(f"数据目录: {config.data_dir}")
        logger.info(f"工作线程数: {config.num_workers}")
        logger.info("开始加载模型...")
    
    # 初始化模型
    if config.model_type == 'vit':
        model = ViTForIQA(pretrained=False)  # 不需要预训练权重，因为会加载训练好的模型
        if logger:
            logger.info("初始化ViT模型")
    elif config.model_type == 'resnet_vit':
        model = ResNetViTForIQA(pretrained=False)
        if logger:
            logger.info("初始化ResNet+ViT模型")
    elif config.model_type == 'resnet_vit_concat':
        model = ResNetViTConcatForIQA(pretrained=False)
        if logger:
            logger.info("初始化ResNet+ViT级联模型")
    elif config.model_type == 'swin':  
        model = SwinForIQA(  
            pretrained=True,   
            freeze_backbone=config.freeze_backbone,  
            model_size=getattr(config, 'swin_size', 'tiny')  # 允许通过配置选择模型大小  
        )  
        if logger:
            logger.info(f"初始化Swin Transformer模型，大小: {getattr(config, 'swin_size', 'tiny')}")
    else:
        error_msg = f"不支持的模型类型: {config.model_type}"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)
    
    # 加载训练好的模型
    if logger:
        logger.info(f"从 {config.model_path} 加载模型权重")
    
    checkpoint = torch.load(config.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    if logger:
        logger.info("模型加载完成，开始评估...")
    
    # 定义损失函数
    criterion = nn.MSELoss()
    
    # 在验证集上评估
    all_preds = []
    all_targets = []
    val_loss = 0.0
    
    if logger:
        logger.info("开始在验证集上评估模型...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # 获取数据和标签
            images = batch['image'].to(device)
            targets = batch['mos'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            val_loss += loss.item() * images.size(0)
            
            # 收集预测和目标值
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 每处理10个批次记录一次进度
            if logger and (batch_idx + 1) % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1}/{len(val_loader)} 批次")
    
    # 计算平均损失
    val_loss = val_loss / len(val_loader.dataset)
    
    if logger:
        logger.info(f"评估完成，验证集大小: {len(val_loader.dataset)}")
        logger.info(f"平均损失: {val_loss:.4f}")
    
    # 计算评估指标
    srcc, _ = spearmanr(all_targets, all_preds)
    plcc, _ = pearsonr(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    
    # 打印结果
    print(f"验证结果 | Loss: {val_loss:.4f}, SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, RMSE: {rmse:.4f}")
    
    if logger:
        logger.info("计算评估指标:")
        logger.info(f"  损失 (Loss): {val_loss:.4f}")
        logger.info(f"  斯皮尔曼等级相关系数 (SRCC): {srcc:.4f}")
        logger.info(f"  皮尔逊线性相关系数 (PLCC): {plcc:.4f}")
        logger.info(f"  均方根误差 (RMSE): {rmse:.4f}")
    
    # 可视化预测结果
    if config.visualize:
        if logger:
            logger.info(f"正在生成可视化结果到 {config.output_dir}...")
        visualize_predictions(all_targets, all_preds, config.output_dir)
        if logger:
            logger.info("可视化结果生成完成")
    
    # 返回评估结果字典，以便在main.py中记录
    results = {
        'loss': val_loss,
        'srcc': srcc,
        'plcc': plcc,
        'rmse': rmse
    }
    
    return results


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
    
    # 创建可视化子目录
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(vis_dir, 'prediction_scatter.png'))
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
    plt.savefig(os.path.join(vis_dir, 'error_histogram.png'))
    plt.close()


# 如果直接运行此脚本，使用默认配置进行评估
if __name__ == '__main__':
    from config import get_eval_config
    import sys
    from logger_utils import setup_logger, log_section_start, log_section_end
    
    if len(sys.argv) < 2:
        print("错误: 需要提供模型路径作为参数")
        print("用法: python evaluate.py <model_path>")
        sys.exit(1)
    
    # 设置配置
    config = get_eval_config(model_path=sys.argv[1])
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logger(config.output_dir, f"eval_{timestamp}.log")
    
    # 记录评估开始
    log_section_start(logger, "独立评估开始")
    logger.info(f"模型路径: {config.model_path}")
    
    # 执行评估
    results = evaluate(config, logger)
    
    # 记录评估结束
    log_section_end(logger, "独立评估结束")
    
    # 打印最终结果
    print("\n最终评估结果:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")