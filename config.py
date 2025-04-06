# config.py
# 集中管理所有配置参数

class Config:
    """
    配置类，用于集中管理所有参数
    """
    def __init__(self, mode='train'):
        """
        初始化配置
        
        参数:
            mode (str): 运行模式，'train'或'evaluate'
        """
        self.mode = mode
        
        # 通用参数
        self.data_dir = '../IQA_dataset/koniq10k/1024x768/'  # 图像数据目录
        self.label_file = 'data/koniq-10k.txt'  # 标签文件路径
        self.output_dir = 'outputs'  # 输出目录，用于保存模型和结果
        self.model_type = 'vit'  # 模型类型: vit、vit_attention或resnet_vit
        self.batch_size = 128  # 批大小
        self.num_workers = 12  # 数据加载的工作线程数
        self.use_cuda = True  # 是否使用CUDA
        
        # 分布式训练参数
        self.world_size = -1  # 进程数量，-1表示使用所有可用GPU
        self.dist_master_addr = 'localhost'  # 主节点地址
        self.dist_master_port = '12355'  # 主节点端口
        self.gpu_ids = None  # 指定要使用的GPU ID列表，用于设置CUDA_VISIBLE_DEVICES环境变量
        
        # 训练特定参数
        self.num_epochs = 50  # 训练轮数
        self.learning_rate = 1e-4  # 学习率
        self.weight_decay = 1e-5  # 权重衰减
        self.freeze_backbone = False  # 是否冻结backbone参数
        self.seed = 42  # 随机种子
        
        # 评估特定参数
        self.model_path = None  # 训练好的模型路径，仅在evaluate模式下需要
        self.visualize = False  # 是否可视化预测结果，仅在evaluate模式下有效
    
    def update(self, **kwargs):
        """
        更新配置参数
        
        参数:
            **kwargs: 键值对，用于更新配置
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"配置中不存在参数: {key}")
    
    def __str__(self):
        """
        返回配置的字符串表示
        """
        config_str = "配置参数:\n"
        for key, value in self.__dict__.items():
            config_str += f"  {key}: {value}\n"
        return config_str


# 默认训练配置
def get_train_config(**kwargs):
    """
    获取训练配置
    
    参数:
        **kwargs: 键值对，用于更新默认配置
        
    返回:
        Config: 训练配置对象
    """
    config = Config(mode='train')
    config.update(**kwargs)
    return config


# 默认评估配置
def get_eval_config(**kwargs):
    """
    获取评估配置
    
    参数:
        **kwargs: 键值对，用于更新默认配置
        
    返回:
        Config: 评估配置对象
    """
    config = Config(mode='evaluate')
    config.update(**kwargs)
    return config