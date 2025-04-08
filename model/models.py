import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
from transformers import SwinModel, SwinConfig
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
        
        # 使用自定义的ViT模型，直接接收ResNet特征
        from model.vit import ViT
        
        # ResNet50输出特征图的通道数为2048，特征图大小为7x7
        resnet_out_channels = 2048
        feature_size = 7  # ResNet50最后一层输出的特征图大小
        
        # 创建ViT模型，直接处理ResNet特征
        self.vit = ViT(
            feature_size=feature_size,  # ResNet50输出特征图大小
            num_classes=1,  # 回归任务，输出一个质量分数
            dim=768,  # 特征维度
            depth=12,  # Transformer深度
            heads=12,  # 注意力头数
            mlp_dim=3072,  # MLP隐藏层维度
            dropout=0.1,
            emb_dropout=0.1
        )
        
        # 不再需要适配层，因为修改后的ViT可以直接处理ResNet特征
        
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
        
        # 直接将ResNet特征输入到修改后的ViT中
        # 修改后的ViT已经包含了特征投影和质量分数预测
        quality_score = self.vit(resnet_features)
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]
    
    
class ResNetViTConcatForIQA(nn.Module):  
    """  
    结合ResNet50和ViT的图像质量评估模型，通过特征拼接融合两种模型的输出  
    """  
    def __init__(self, pretrained=True, freeze_backbone=False):  
        """  
        初始化ResNet50+ViT特征拼接IQA模型  
        
        参数:  
            pretrained (bool): 是否使用预训练权重  
            freeze_backbone (bool): 是否冻结backbone参数  
        """  
        super(ResNetViTConcatForIQA, self).__init__()  
        
        # 加载预训练的ResNet50模型作为特征提取器  
        self.resnet = resnet50_backbone()  
        
        # 全局平均池化层，用于将ResNet特征图转换为向量  
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        
        # 加载预训练的ViT模型  
        if pretrained:  
            self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')  
        else:  
            config = ViTConfig()  
            self.vit = ViTModel(config)  
        
        # 获取特征维度  
        resnet_dim = 2048  # ResNet50 最后一层输出的通道数  
        vit_dim = self.vit.config.hidden_size  # ViT 的特征维度 (768)  
        concat_dim = resnet_dim + vit_dim  # 拼接后的特征维度 (2048+768=2816)  
        
        # 特征融合和质量预测层  
        self.regression_head = nn.Sequential(  
            nn.Linear(concat_dim, 512),  
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
        
        # 全局池化将特征图转换为向量  
        resnet_pooled = self.global_pool(resnet_features).squeeze(-1).squeeze(-1)  # [batch_size, 2048]  
        
        # 通过ViT提取特征  
        vit_output = self.vit(x)  
        vit_features = vit_output.pooler_output  # [batch_size, 768]  
        
        # 特征拼接  
        concat_features = torch.cat([resnet_pooled, vit_features], dim=1)  # [batch_size, 2816]  
        
        # 通过回归头预测质量分数  
        quality_score = self.regression_head(concat_features)  
        
        return quality_score.squeeze(1)  # 移除最后一个维度，变为 [batch_size]


class SwinForIQA(nn.Module):  
    """  
    基于Swin Transformer的图像质量评估模型  
    """  
    def __init__(self, pretrained=True, freeze_backbone=False, model_size='tiny'):  
        """  
        初始化Swin Transformer IQA模型  
        
        参数:  
            pretrained (bool): 是否使用预训练权重  
            freeze_backbone (bool): 是否冻结backbone参数  
            model_size (str): 模型大小，可选'tiny', 'small', 'base', 'large'  
        """  
        super(SwinForIQA, self).__init__()  
        
        # 根据size选择合适的预训练模型  
        model_variants = {  
            'tiny': 'microsoft/swin-tiny-patch4-window7-224',  
            'small': 'microsoft/swin-small-patch4-window7-224',   
            'base': 'microsoft/swin-base-patch4-window7-224',  
            'large': 'microsoft/swin-large-patch4-window7-224'  
        }  
        
        model_name = model_variants.get(model_size, model_variants['tiny'])  
        
        # 加载预训练的Swin模型  
        if pretrained:  
            self.swin = SwinModel.from_pretrained(model_name)  
        else:  
            config = SwinConfig.from_pretrained(model_name)  
            self.swin = SwinModel(config)  
        
        # 获取Swin的特征维度  
        swin_hidden_dim = self.swin.config.hidden_size  
        
        # 添加IQA回归头  
        self.regression_head = nn.Sequential(  
            nn.Linear(swin_hidden_dim, 512),  
            nn.ReLU(),  
            nn.Dropout(0.2),  
            nn.Linear(512, 128),  
            nn.ReLU(),  
            nn.Dropout(0.2),  
            nn.Linear(128, 1)  
        )  
        
        # 如果需要，冻结backbone参数  
        if freeze_backbone:  
            for param in self.swin.parameters():  
                param.requires_grad = False  
    
    def forward(self, x):  
        """  
        前向传播  
        
        参数:  
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, 3, 224, 224]  
                
        返回:  
            torch.Tensor: 预测的图像质量分数，形状为 [batch_size]  
        """  
        # 提取Swin特征  
        outputs = self.swin(pixel_values=x)  
        
        # 获取pooled输出  
        features = outputs.pooler_output  
        
        # 通过回归头预测质量分数  
        quality_score = self.regression_head(features)  
        
        return quality_score.squeeze(1)  