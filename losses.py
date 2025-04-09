import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):
    """
    均方误差损失函数
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class AdaptiveBoundaryRankingLoss(nn.Module):
    """
    自适应边界排序损失函数
    
    L_rank = (1/K) * Σ max(0, (λ·|y_i - y_j|) - (y_pred_i - y_pred_j)·sign(y_i - y_j))
    
    其中：
    - K: 所有图像对的数量
    - λ: 基于Weber-Fechner定律的自适应边界因子，λ = β / (1 + γ * |y_i - y_j|)
    - y_i, y_j: 真实质量分数
    - y_pred_i, y_pred_j: 模型预测的质量分数
    - sign(y_i - y_j): 排序方向项
    """
    def __init__(self, beta=0.3, gamma=0.1):
        super(AdaptiveBoundaryRankingLoss, self).__init__()
        self.beta = beta    # 边界强度控制因子
        self.gamma = gamma  # 非线性调整因子
    
    def forward(self, pred, target):
        # 获取批次大小
        batch_size = pred.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        # 展平预测和真实分数
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 创建所有预测分数对的差异矩阵
        pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # [batch_size, batch_size]
        
        # 创建所有真实分数对的差异矩阵
        true_diff = target.unsqueeze(0) - target.unsqueeze(1)  # [batch_size, batch_size]
        
        # 计算真实分数差异的绝对值
        abs_true_diff = torch.abs(true_diff)
        
        # 计算自适应边界系数 λ = β / (1 + γ * |y_i - y_j|)
        lambda_values = self.beta / (1 + self.gamma * abs_true_diff)
        
        # 计算每对的边界大小
        margins = lambda_values * abs_true_diff
        
        # 计算排序方向的符号
        sign_true_diff = torch.sign(true_diff)
        
        # 计算预测差异与正确排序方向的乘积
        pred_diff_signed = pred_diff * sign_true_diff
        
        # 应用hinge损失: max(0, margin - pred_diff_signed)
        hinge_loss = F.relu(margins - pred_diff_signed)
        
        # 创建掩码，排除对角线和重复对
        mask = torch.ones_like(hinge_loss, device=pred.device) - torch.eye(batch_size, device=pred.device)
        mask = torch.triu(mask, diagonal=1)  # 只取上三角矩阵
        
        # 应用掩码并求和
        valid_pairs = torch.sum(mask)
        if valid_pairs > 0:
            loss = torch.sum(hinge_loss * mask) / valid_pairs
        else:
            loss = torch.tensor(0.0, device=pred.device)
            
        return loss

class ExponentialRankingLoss(nn.Module):  
    """  
    使用矩阵运算优化的指数形式排序损失函数  
    
    公式: lossrank = (2/N) * Σ{e^(yˆi−yˆi+1), if yi < yi+1; 0, others}  
    """  
    def __init__(self):  
        super(ExponentialRankingLoss, self).__init__()  
    
    def forward(self, pred, target):  
        batch_size = pred.size(0)  
        if batch_size <= 1:  
            return torch.tensor(0.0, device=pred.device)  
        
        # 展平预测和真实分数  
        pred = pred.view(-1)  
        target = target.view(-1)  
        
        # 直接筛选相邻对(步长为2)  
        loss = torch.tensor(0.0, device=pred.device)  
        
        # 创建索引  
        even_indices = torch.arange(0, batch_size-1, 2, device=pred.device)  
        
        # 提取对应位置的值  
        even_targets = target[even_indices]  
        odd_targets = target[even_indices + 1]  
        even_preds = pred[even_indices]  
        odd_preds = pred[even_indices + 1]  
        
        # 掩码和损失计算  
        mask = even_targets < odd_targets  
        if mask.sum() > 0:  
            # 只对满足条件的对计算损失  
            loss = torch.exp(even_preds[mask] - odd_preds[mask]).sum()  
            # 乘以2/N系数  
            loss = (2.0 / batch_size) * loss  
        
        return loss  

class CombinedLoss(nn.Module):
    """
    结合MSE损失和排序损失的组合损失函数
    
    可以选择使用自适应边界排序损失或指数形式的排序损失
    """
    def __init__(self, mse_weight=1.0, rank_weight=0.2, beta=0.3, gamma=0.1, rank_type='adaptive'):
        super(CombinedLoss, self).__init__()
        self.mse_loss = MSELoss()
        
        # 根据指定的类型选择排序损失函数
        if rank_type == 'adaptive':
            self.rank_loss = AdaptiveBoundaryRankingLoss(beta=beta, gamma=gamma)
        elif rank_type == 'exponential':
            self.rank_loss = ExponentialRankingLoss()
        else:
            raise ValueError(f"不支持的排序损失类型: {rank_type}")
            
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        rank = self.rank_loss(pred, target)
        
        # 组合损失
        combined = self.mse_weight * mse + self.rank_weight * rank
        
        return combined, mse, rank