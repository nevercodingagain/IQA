import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

class KonIQ10kDataset(Dataset):
    """
    KonIQ-10k数据集加载类
    """
    def __init__(self, root_dir, data=None, label_file=None, transform=None):  
        """  
        初始化KonIQ-10k数据集  
        
        参数:  
            root_dir (str): 图像文件所在的根目录  
            data (pandas.DataFrame, optional): 预先划分好的数据  
            label_file (str, optional): 标签文件路径 (当data为None时使用)  
            transform (callable, optional): 应用于图像的转换  
        """  
        self.root_dir = root_dir  
        self.transform = transform  
        
        # 直接使用传入的数据，或者读取标签文件  
        if data is not None:  
            self.data = data  
        elif label_file is not None:  
            # 读取标签文件  
            self.data = self._read_label_file(label_file)  
        else:  
            raise ValueError("必须提供data或label_file") 
    
    def _read_label_file(self, label_file):
        """
        读取标签文件
        
        参数:
            label_file (str): 标签文件路径
            
        返回:
            pandas.DataFrame: 包含图像文件名和质量评分的数据框
        """
        # 读取标签文件
        df = pd.read_csv(label_file, sep='\t', header=None)
        df.columns = ['id', 'image_name', 'mos']
        return df
    
    def __len__(self):
        """
        返回数据集大小
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            dict: 包含图像和质量评分的字典
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 获取图像文件名和质量评分
        img_name = self.data.iloc[idx]['image_name']
        mos = self.data.iloc[idx]['mos']
        
        # 构建图像路径
        img_path = os.path.join(self.root_dir, img_name)
        
        # 读取图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像 {img_path}: {e}")
            # 返回一个空图像和原始MOS值
            image = Image.new('RGB', (224, 224), color='black')
        
        # 应用转换
        if self.transform:
            image = self.transform(image)
        
        return {'image': image, 'mos': torch.tensor(mos, dtype=torch.float32)}


def get_data_transforms():
    """
    获取数据转换
    
    返回:
        dict: 包含训练和测试转换的字典
    """
    # 定义数据转换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ViT通常使用224x224的输入大小
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return {'train': train_transform, 'val': val_transform}


def get_dataloaders(root_dir, label_file, batch_size=32, num_workers=4, train_ratio=0.8, val_ratio=0.2, random_state=42):  
    """  
    获取数据加载器，确保训练集和验证集没有重叠  
    
    参数:  
        root_dir (str): 图像文件所在的根目录  
        label_file (str): 标签文件路径  
        batch_size (int): 批大小  
        num_workers (int): 数据加载的工作线程数  
        train_ratio (float): 训练集比例  
        val_ratio (float): 验证集比例  
        random_state (int): 随机种子，控制数据集划分的随机性  
        
    返回:  
        dict: 包含训练和验证数据加载器的字典  
    """  
    # 获取数据转换  
    transforms_dict = get_data_transforms()  
    
    # 读取所有数据并执行一次性划分  
    df = pd.read_csv(label_file, sep='\t', header=None)  
    df.columns = ['id', 'image_name', 'mos']  
    
    # 确保比例之和为1  
    assert abs(train_ratio + val_ratio - 1.0) < 1e-5, "比例之和必须为1"  
    
    # 打乱数据 - 使用传入的random_state  
    shuffled_data = df.sample(frac=1, random_state=random_state).reset_index(drop=True)  
    
    # 计算各集合的大小  
    n_samples = len(shuffled_data)  
    n_train = int(n_samples * train_ratio)  
    
    # 划分为训练集和验证集  
    train_data = shuffled_data[:n_train]  
    val_data = shuffled_data[n_train:]  
    
    # 验证没有重叠  
    # train_images = set(train_data['image_name'])  
    # val_images = set(val_data['image_name'])  
    # overlap = train_images.intersection(val_images)  
    # print(f"训练集大小: {len(train_images)}")  
    # print(f"验证集大小: {len(val_images)}")  
    # print(f"重叠数量: {len(overlap)}")  
    
    # 创建数据集  
    train_dataset = KonIQ10kDataset(  
        root_dir=root_dir,  
        data=train_data,  # 传入预先划分的数据  
        transform=transforms_dict['train']  
    )  
    
    val_dataset = KonIQ10kDataset(  
        root_dir=root_dir,  
        data=val_data,  # 传入预先划分的数据  
        transform=transforms_dict['val']  
    )  
    
    # 创建数据加载器  
    train_loader = DataLoader(  
        train_dataset,  
        batch_size=batch_size,  
        shuffle=True,  
        num_workers=num_workers,  
        pin_memory=True  
    )  
    
    val_loader = DataLoader(  
        val_dataset,  
        batch_size=batch_size,  
        shuffle=False,  
        num_workers=num_workers,  
        pin_memory=True  
    )  
    
    return {'train': train_loader, 'val': val_loader}


def visualize_samples(dataloader, num_samples=5):
    """
    可视化数据集中的样本
    
    参数:
        dataloader: 数据加载器
        num_samples (int): 要可视化的样本数量
    """
    # 获取一批数据
    batch = next(iter(dataloader))
    images = batch['image']
    scores = batch['mos']
    
    # 反标准化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # 可视化图像和评分
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        ax.set_title(f"MOS: {scores[i]:.2f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数，用于测试数据集加载
    """
    # 设置路径
    root_dir = 'dataset/koniq-10k/1024x768'
    label_file = 'data/koniq-10k.txt'
    
    # 获取数据加载器
    dataloaders = get_dataloaders(  
        root_dir=root_dir,   
        label_file=label_file,   
        batch_size=32,  
        random_state=42 
    )  
    
    # 打印数据集大小
    for split, dataloader in dataloaders.items():
        print(f"{split} dataset size: {len(dataloader.dataset)}")
    
    # 可视化样本
    visualize_samples(dataloaders['train'])


if __name__ == '__main__':
    main()