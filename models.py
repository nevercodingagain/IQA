import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTForIQA(nn.Module):
    """
    基于Vision Transformer的图像质量评估模型
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        初始化ViT IQA模型
        
        参数:
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结backbone参数
        """
        super(ViTForIQA, self).__init__()
        
        # 加载预训练的ViT模型
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16()
        
        # 获取ViT的特征维度
        vit_hidden_dim = self.vit.heads.head.in_features
        
        # 移除分类头
        self.vit.heads = nn.Identity()
        
        # 添加IQA回归头
        self.regression_head = nn.Sequential(
            nn.Linear(vit_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # 如果需要，冻结backbone参数
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, 224, 224]
            
        返回:
            torch.Tensor: 预测的图像质量分数，形状为 [batch_size, 1]
        """
        # 提取ViT特征
        features = self.vit(x)
        
        # 通过回归头预测质量分数
        quality_score = self.regression_head(features)
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]


class ViTWithAttentionForIQA(nn.Module):
    """
    带有注意力机制的ViT图像质量评估模型
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        初始化带有注意力机制的ViT IQA模型
        
        参数:
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结backbone参数
        """
        super(ViTWithAttentionForIQA, self).__init__()
        
        # 加载预训练的ViT模型
        if pretrained:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16()
        
        # 获取ViT的特征维度
        vit_hidden_dim = self.vit.heads.head.in_features
        
        # 移除分类头
        self.vit.heads = nn.Identity()
        
        # 添加注意力层
        self.attention = nn.Sequential(
            nn.Linear(vit_hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # 添加IQA回归头
        self.regression_head = nn.Sequential(
            nn.Linear(vit_hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # 如果需要，冻结backbone参数
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, 224, 224]
            
        返回:
            torch.Tensor: 预测的图像质量分数，形状为 [batch_size, 1]
        """
        # 提取ViT特征
        features = self.vit(x)
        
        # 应用注意力机制
        attention_weights = F.softmax(self.attention(features), dim=1)
        weighted_features = attention_weights * features
        
        # 通过回归头预测质量分数
        quality_score = self.regression_head(weighted_features)
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]