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
        batch_size = pred.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=pred.device)
        
        # 计算所有可能的图像对数量
        K = batch_size * (batch_size - 1) // 2
        
        # 初始化损失
        loss = 0.0
        
        # 计算所有图像对的损失
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # 获取真实质量分数和预测分数
                y_i, y_j = target[i], target[j]
                y_pred_i, y_pred_j = pred[i], pred[j]
                
                # 计算真实质量分数差异的绝对值
                y_diff_abs = torch.abs(y_i - y_j)
                
                # 计算自适应边界因子 λ = β / (1 + γ * |y_i - y_j|)
                lambda_factor = self.beta / (1 + self.gamma * y_diff_abs)
                
                # 计算边界项
                boundary = lambda_factor * y_diff_abs
                
                # 计算排序方向项
                sign_term = torch.sign(y_i - y_j)
                pred_diff = (y_pred_i - y_pred_j) * sign_term
                
                # 计算hinge损失
                pair_loss = torch.max(torch.tensor(0.0, device=pred.device), boundary - pred_diff)
                
                # 累加损失
                loss += pair_loss
        
        # 归一化损失
        loss = loss / K
        
        return loss

class CombinedLoss(nn.Module):
    """
    结合MSE损失和自适应边界排序损失的组合损失函数
    """
    def __init__(self, mse_weight=1.0, rank_weight=1.0, beta=0.3, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.mse_loss = MSELoss()
        self.rank_loss = AdaptiveBoundaryRankingLoss(beta=beta, gamma=gamma)
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
    
    def forward(self, pred, target):
        mse = self.mse_loss(pred, target)
        rank = self.rank_loss(pred, target)
        
        # 组合损失
        combined = self.mse_weight * mse + self.rank_weight * rank
        
        return combined, mse, rank