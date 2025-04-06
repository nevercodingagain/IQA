import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from dataset_utils import get_dataloaders
from models import ViTForIQA, ViTWithAttentionForIQA

# 设置随机种子，确保结果可复现
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# 训练一个epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    # 使用tqdm包装数据加载器，显示进度条
    for batch in tqdm(dataloader, desc="训练中", leave=False):
        # 获取数据和标签
        images = batch['image'].to(device)
        targets = batch['mos'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 累加损失
        epoch_loss += loss.item() * images.size(0)
        
        # 收集预测和目标值，用于计算相关系数
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
    
    # 计算平均损失
    epoch_loss = epoch_loss / len(dataloader.dataset)
    
    # 计算SRCC和PLCC
    srcc, _ = spearmanr(all_targets, all_preds)
    plcc, _ = pearsonr(all_targets, all_preds)
    
    return epoch_loss, srcc, plcc

# 验证一个epoch
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        # 使用tqdm包装数据加载器，显示进度条
        for batch in tqdm(dataloader, desc="验证中", leave=False):
            # 获取数据和标签
            images = batch['image'].to(device)
            targets = batch['mos'].to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 累加损失
            epoch_loss += loss.item() * images.size(0)
            
            # 收集预测和目标值，用于计算相关系数
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算平均损失
    epoch_loss = epoch_loss / len(dataloader.dataset)
    
    # 计算SRCC和PLCC
    srcc, _ = spearmanr(all_targets, all_preds)
    plcc, _ = pearsonr(all_targets, all_preds)
    
    return epoch_loss, srcc, plcc

# 主训练函数
def train(config):
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.use_cuda else "cpu")
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.output_dir, 'logs'))
    
    # 获取数据加载器
    dataloaders = get_dataloaders(
        root_dir=config.data_dir,
        label_file=config.label_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    # 初始化模型
    if config.model_type == 'vit':
        model = ViTForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)
    elif config.model_type == 'vit_attention':
        model = ViTWithAttentionForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # 训练循环
    best_val_srcc = -1.0
    best_epoch = 0
    
    # 使用tqdm包装epoch循环，显示总体训练进度
    for epoch in tqdm(range(config.num_epochs), desc="训练进度"):
        # 训练
        train_loss, train_srcc, train_plcc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # 验证
        val_loss, val_srcc, val_plcc = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('SRCC/train', train_srcc, epoch)
        writer.add_scalar('SRCC/val', val_srcc, epoch)
        writer.add_scalar('PLCC/train', train_plcc, epoch)
        writer.add_scalar('PLCC/val', val_plcc, epoch)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f}, SRCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f} | "
              f"Val Loss: {val_loss:.4f}, SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}")
        
        # 保存最佳模型
        if val_srcc > best_val_srcc:
            best_val_srcc = val_srcc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_srcc': val_srcc,
                'val_plcc': val_plcc,
            }, os.path.join(config.output_dir, 'best_model.pth'))
            print(f"保存最佳模型，验证SRCC: {val_srcc:.4f}")
    
    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(config.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在测试集上评估
    test_loss, test_srcc, test_plcc = validate_epoch(model, dataloaders['test'], criterion, device)
    print(f"\n测试结果 (来自第{best_epoch+1}个epoch的最佳模型) | "
          f"Loss: {test_loss:.4f}, SRCC: {test_srcc:.4f}, PLCC: {test_plcc:.4f}")
    
    # 关闭TensorBoard writer
    writer.close()

# 如果直接运行此脚本，使用默认配置进行训练
if __name__ == '__main__':
    from config import get_train_config
    config = get_train_config()
    train(config)