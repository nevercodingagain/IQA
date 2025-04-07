import os
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from dataset_utils import get_dataloaders, KonIQ10kDataset, get_data_transforms
from model.models import ViTForIQA, ResNetViTForIQA, ResNetViTConcatForIQA

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
    rank = dist.get_rank()
    dataset_size = len(dataloader.dataset)
    
    
    # 只在主进程显示训练进度条
    if rank == 0:
        dataloader = tqdm(dataloader, desc="训练中", leave=False, position=1)
    for batch in dataloader:
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
    epoch_loss = epoch_loss / dataset_size
    
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
    rank = dist.get_rank()
    dataset_size = len(dataloader.dataset)
    
    with torch.no_grad():
        # 只在主进程显示验证进度条
        if rank == 0:
            dataloader = tqdm(dataloader, desc="验证中", leave=False, position=1)
        for batch in dataloader:
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
    epoch_loss = epoch_loss / dataset_size
    
    # 计算SRCC和PLCC
    srcc, _ = spearmanr(all_targets, all_preds)
    plcc, _ = pearsonr(all_targets, all_preds)
    
    return epoch_loss, srcc, plcc

# 设置分布式训练环境
def setup(rank, world_size, config):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = config.dist_master_addr
    os.environ['MASTER_PORT'] = config.dist_master_port
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)

# 清理分布式训练环境
def cleanup():
    """清理分布式训练环境"""
    dist.destroy_process_group()

# 获取分布式数据加载器
def get_distributed_dataloaders(rank, world_size, config):
    """创建分布式数据加载器"""
    transforms_dict = get_data_transforms()
    
    # 创建数据集
    train_dataset = KonIQ10kDataset(
        root_dir=config.data_dir,
        label_file=config.label_file,
        transform=transforms_dict['train'],
        split='train'
    )
    
    val_dataset = KonIQ10kDataset(
        root_dir=config.data_dir,
        label_file=config.label_file,
        transform=transforms_dict['test'],
        split='val'
    )
    
    test_dataset = KonIQ10kDataset(
        root_dir=config.data_dir,
        label_file=config.label_file,
        transform=transforms_dict['test'],
        split='test'
    )
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # 使用DistributedSampler时必须为False
        num_workers=config.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    
    # 验证和测试集不需要分布式采样器
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}, train_sampler

# 分布式训练函数
def train_distributed(rank, world_size, config):
    """在单个进程中执行分布式训练"""
    # 设置随机种子
    set_seed(config.seed + rank)  # 每个进程使用不同的种子
    
    # 设置分布式环境
    setup(rank, world_size, config)
    
    # 设置设备
    device = torch.device(f"cuda:{rank}")
    
    # 创建实验输出目录
    if rank == 0:  # 只在主进程中创建目录和初始化TensorBoard
        # 创建实验名称目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir_name = f"{config.experiment_name}_{config.model_type}_{config.dataset}_{config.batch_size}_{timestamp}"
        experiment_dir = os.path.join(config.output_dir, experiment_dir_name)
        
        # 创建子目录
        logs_dir = os.path.join(experiment_dir, 'logs')
        models_dir = os.path.join(experiment_dir, 'models')
        tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
        
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        
        # 初始化TensorBoard
        writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # 保存当前配置到日志目录
        with open(os.path.join(logs_dir, 'config.txt'), 'w') as f:
            f.write(str(config))
            
        print(f"使用设备: {device} (主进程)")
        print(f"实验目录: {experiment_dir}")
    else:
        writer = None
        experiment_dir = None
        models_dir = None
        print(f"使用设备: {device} (进程 {rank})")
    
    # 获取分布式数据加载器
    dataloaders, train_sampler = get_distributed_dataloaders(rank, world_size, config)
    
    # 初始化模型
    if config.model_type == 'vit':  
        model = ViTForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)  
    elif config.model_type == 'resnet_vit':  
        model = ResNetViTForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)  
    elif config.model_type == 'resnet_vit_concat':  # 新增模型类型  
        model = ResNetViTConcatForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)  
    else:  
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    
    # 将模型移动到当前设备
    model = model.to(device)
    
    # 使用DistributedDataParallel包装模型
    model = DDP(model, device_ids=[rank])
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=(rank==0))  # 只在主进程中显示详细信息
    
    # 训练循环
    best_val_metric = -1.0  # 使用SRCC+PLCC作为最佳模型指标
    best_epoch = 0
    
    # 使用tqdm包装epoch循环，只在主进程中显示进度
    epoch_iterator = tqdm(range(config.num_epochs), desc="训练进度", position=0, leave=True) if rank == 0 else range(config.num_epochs)
    for epoch in epoch_iterator:
        # 设置采样器的epoch，确保每个epoch的数据顺序不同
        train_sampler.set_epoch(epoch)
        
        # 训练
        train_loss, train_srcc, train_plcc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        
        # 验证
        val_loss, val_srcc, val_plcc = validate_epoch(model, dataloaders['val'], criterion, device)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 只在主进程中记录到TensorBoard和打印进度
        if rank == 0:
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
            
            # 计算综合指标(SRCC+PLCC)
            val_metric = val_srcc + val_plcc
            
            # 保存最佳模型 - 只在SRCC+PLCC总和超过历史最高时保存
            if val_metric > best_val_metric:  
                best_val_metric = val_metric  
                best_epoch = epoch  
                
                # 保存最佳模型文件
                torch.save({  
                    'epoch': epoch,  
                    'model_state_dict': model.module.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'val_srcc': val_srcc,  
                    'val_plcc': val_plcc,  
                    'val_metric': val_metric,  
                }, os.path.join(models_dir, 'best_model.pth'))
                
                print(f"保存最佳模型，验证SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, 总和: {val_metric:.4f}")
                
                # 记录更好的结果到日志文件（追加模式）
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(os.path.join(logs_dir, 'test_results.txt'), 'a') as f:
                    f.write(f"[{current_time}] Epoch {epoch+1}/{config.num_epochs} - 新的最佳结果\n")
                    f.write(f"验证 SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, 总和: {val_metric:.4f}\n\n")
            
            # 按照保存频率保存带有epoch信息的模型文件
            if (epoch + 1) % config.save_frequency == 0:
                torch.save({  
                    'epoch': epoch,  
                    'model_state_dict': model.module.state_dict(),  
                    'optimizer_state_dict': optimizer.state_dict(),  
                    'val_srcc': val_srcc,  
                    'val_plcc': val_plcc,  
                    'val_metric': val_metric,  
                }, os.path.join(models_dir, f'model_epoch_{epoch+1}.pth'))  
                print(f"按照保存频率保存模型，epoch: {epoch+1}")
    
    # 只在主进程中进行测试评估
    if rank == 0:
        # 在测试集上评估最佳模型
        print(f"\n训练完成，最佳模型来自第{best_epoch+1}个epoch，验证指标总和: {best_val_metric:.4f}")
        
        # 记录最终训练结果到日志文件（追加模式）
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(logs_dir, 'test_results.txt'), 'a') as f:
            f.write(f"\n[{current_time}] 训练完成 - 最佳模型来自第{best_epoch+1}个epoch\n")
            f.write(f"最终验证指标总和: {best_val_metric:.4f}\n\n")
        
        # 关闭TensorBoard writer
        writer.close()
    
    # 清理分布式环境
    cleanup()

# 主训练函数
def train(config):
    """启动分布式训练"""
    # 设置随机种子（主进程）
    set_seed(config.seed)
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: 没有可用的CUDA设备，无法进行分布式训练")
        return
    
    # 获取可用的GPU数量
    if config.world_size == -1:
        world_size = torch.cuda.device_count()
    else:
        world_size = config.world_size
    
    if world_size <= 1:
        print(f"警告: 只有{world_size}个GPU可用，将使用单GPU训练")
        # 回退到非分布式训练
        # 这里可以实现单GPU训练逻辑，或者仍然使用DDP但只有一个进程
    
    print(f"启动分布式训练，使用{world_size}个GPU")
    
    # 使用spawn启动多个进程
    mp.spawn(
        train_distributed,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

# 如果直接运行此脚本，使用默认配置进行训练
if __name__ == '__main__':
    from config import get_train_config
    config = get_train_config()
    train(config)