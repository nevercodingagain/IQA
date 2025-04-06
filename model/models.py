import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from model.resnet import resnet50_backbone


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
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # 获取ViT的特征维度
        vit_hidden_dim = self.vit.config.hidden_size
        
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
        outputs = self.vit(x)
        # 使用池化后的特征 (CLS token)
        features = outputs.pooler_output
        
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
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # 获取ViT的特征维度
        vit_hidden_dim = self.vit.config.hidden_size
        
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
        outputs = self.vit(x)
        # 使用池化后的特征 (CLS token)
        features = outputs.pooler_output
        
        # 应用注意力机制
        attention_weights = F.softmax(self.attention(features), dim=1)
        weighted_features = attention_weights * features
        
        # 通过回归头预测质量分数
        quality_score = self.regression_head(weighted_features)
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]


class ResNetViTForIQA(nn.Module):
    """
    结合ResNet50和ViT的图像质量评估模型
    """
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        初始化ResNet50+ViT IQA模型
        
        参数:
            pretrained (bool): 是否使用预训练权重
            freeze_backbone (bool): 是否冻结backbone参数
        """
        super(ResNetViTForIQA, self).__init__()
        
        # 加载预训练的ResNet50模型作为特征提取器
        self.resnet = resnet50_backbone()
        
        # 加载预训练的ViT模型
        if pretrained:
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        # 获取ViT的特征维度
        vit_hidden_dim = self.vit.config.hidden_size
        
        # ResNet50输出特征图的通道数为2048
        resnet_out_channels = 2048
        
        # 添加适配层，将ResNet特征转换为ViT可接受的格式
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),  # 调整特征图大小
            nn.Conv2d(resnet_out_channels, 768, kernel_size=1, stride=1),  # 调整通道数与ViT匹配
            nn.Flatten(2),  # 将特征图展平为序列
            nn.Transpose(1, 2)  # 调整维度顺序以适应ViT输入
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
            for param in self.resnet.parameters():
                param.requires_grad = False
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
        # 通过ResNet50提取特征
        resnet_features = self.resnet(x)  # [batch_size, 2048, 7, 7]
        
        # 将ResNet特征转换为ViT可接受的格式
        adapted_features = self.adapter(resnet_features)  # [batch_size, 196, 768]
        
        # 通过ViT处理特征
        vit_outputs = self.vit(inputs_embeds=adapted_features)
        vit_features = vit_outputs.pooler_output  # [batch_size, 768]
        
        # 通过回归头预测质量分数
        quality_score = self.regression_head(vit_features)
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]