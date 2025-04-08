import torch  
import torch.distributed as dist  
from scipy.stats import spearmanr, pearsonr  
from tqdm import tqdm  

# 训练一个epoch  
def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None, num_epochs=None):  
    model.train()  
    epoch_loss = 0.0  
    all_preds = []  
    all_targets = []  
    
    # 分布式处理  
    is_distributed = dist.is_initialized()  
    if is_distributed:  
        rank = dist.get_rank()  
        world_size = dist.get_world_size()  
    else:  
        rank = 0  
        world_size = 1  
    
    dataset_size = len(dataloader.dataset)  
    
    # 只在主进程显示训练进度条  
    if rank == 0:  
        # 添加epoch信息到进度条描述
        desc = "训练中"
        if epoch is not None and num_epochs is not None:
            desc = f"Epoch {epoch+1}/{num_epochs} 训练中"
        dataloader = tqdm(dataloader, desc=desc, leave=False, position=1)  
    
    for batch in dataloader:  
        # 获取数据和标签  
        images = batch['image'].to(device, non_blocking=True)  
        targets = batch['mos'].to(device, non_blocking=True)  
        
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
        
        # 收集预测和目标值  
        all_preds.extend(outputs.detach().cpu().numpy())  
        all_targets.extend(targets.detach().cpu().numpy())  
    
    # 计算平均损失  
    epoch_loss = epoch_loss / dataset_size  
    
    # 计算SRCC和PLCC  
    srcc, _ = spearmanr(all_targets, all_preds)  
    plcc, _ = pearsonr(all_targets, all_preds)  
    
    return epoch_loss, srcc, plcc  

# 验证一个epoch  
def validate_epoch(model, dataloader, criterion, device, epoch=None, num_epochs=None):  
    model.eval()  
    epoch_loss = 0.0  
    all_preds = []  
    all_targets = []  
    
    # 分布式处理  
    is_distributed = dist.is_initialized()  
    if is_distributed:  
        rank = dist.get_rank()  
        world_size = dist.get_world_size()  
    else:  
        rank = 0  
        world_size = 1  
    
    dataset_size = len(dataloader.dataset)  
    
    with torch.no_grad():  
        # 只在主进程显示验证进度条  
        if rank == 0:  
            # 添加epoch信息到进度条描述
            desc = "验证中"
            if epoch is not None and num_epochs is not None:
                desc = f"Epoch {epoch+1}/{num_epochs} 验证中"
            dataloader = tqdm(dataloader, desc=desc, leave=False, position=1)  
        
        for batch in dataloader:  
            # 获取数据和标签  
            images = batch['image'].to(device, non_blocking=True)  
            targets = batch['mos'].to(device, non_blocking=True)  
            
            # 前向传播  
            outputs = model(images)  
            
            # 计算损失  
            loss = criterion(outputs, targets)  
            
            # 累加损失  
            epoch_loss += loss.item() * images.size(0)  
            
            # 收集预测和目标值  
            all_preds.extend(outputs.cpu().numpy())  
            all_targets.extend(targets.cpu().numpy())  
    
    # 计算平均损失  
    epoch_loss = epoch_loss / dataset_size  
    
    # 计算SRCC和PLCC  
    srcc, _ = spearmanr(all_targets, all_preds)  
    plcc, _ = pearsonr(all_targets, all_preds)  
    
    return epoch_loss, srcc, plcc