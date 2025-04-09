import os
import logging
import sys
from datetime import datetime


def setup_logger(experiment_dir, log_file_name=None):
    """
    设置日志记录器
    
    参数:
        experiment_dir (str): 实验目录路径
        log_file_name (str, optional): 日志文件名，如果为None则自动生成
        
    返回:
        logging.Logger: 配置好的日志记录器
    """
    # 创建日志目录
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # 如果未提供日志文件名，则自动生成
    if log_file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"experiment_{timestamp}.log"
    
    # 完整的日志文件路径
    log_file_path = os.path.join(logs_dir, log_file_name)
    
    # 创建日志记录器
    logger = logging.getLogger('IQA')
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_section_start(logger, section_name):
    """
    记录章节开始
    
    参数:
        logger (logging.Logger): 日志记录器
        section_name (str): 章节名称
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"开始: {section_name}")
    logger.info("=" * 80)


def log_section_end(logger, section_name):
    """
    记录章节结束
    
    参数:
        logger (logging.Logger): 日志记录器
        section_name (str): 章节名称
    """
    logger.info("=" * 80)
    logger.info(f"结束: {section_name}")
    logger.info("=" * 80 + "\n")


def log_config(logger, config):
    """
    记录配置信息
    
    参数:
        logger (logging.Logger): 日志记录器
        config (Config): 配置对象
    """
    logger.info("配置参数:")
    for key, value in config.__dict__.items():
        logger.info(f"  {key}: {value}")


def log_dataset_info(logger, train_dataset=None, val_dataset=None):
    """
    记录数据集信息
    
    参数:
        logger (logging.Logger): 日志记录器
        train_dataset: 训练数据集
        val_dataset: 验证数据集
    """
    logger.info("数据集信息:")
    if train_dataset is not None:
        logger.info(f"  训练集大小: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"  验证集大小: {len(val_dataset)}")


def log_epoch_results(logger, epoch, num_epochs, train_results, val_results=None, is_best=False):
    """
    记录每个epoch的训练和验证结果
    
    参数:
        logger (logging.Logger): 日志记录器
        epoch (int): 当前epoch
        num_epochs (int): 总epoch数
        train_results (tuple): 训练结果，包含(loss, srcc, plcc)或(loss, srcc, plcc, mse_loss, rank_loss)
        val_results (tuple, optional): 验证结果，格式同train_results
        is_best (bool): 是否是最佳模型
    """
    # 处理训练结果
    if len(train_results) == 3:
        train_loss, train_srcc, train_plcc = train_results
        train_mse_loss, train_rank_loss = None, None
    else:
        train_loss, train_srcc, train_plcc, train_mse_loss, train_rank_loss = train_results
    
    # 构建训练结果字符串
    train_str = f"Epoch {epoch+1}/{num_epochs} - 训练: Loss: {train_loss:.4f}, SRCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f}"
    if train_mse_loss is not None and train_rank_loss is not None:
        train_str += f", MSE Loss: {train_mse_loss:.4f}, Rank Loss: {train_rank_loss:.4f}"
    
    logger.info(train_str)
    
    # 处理验证结果
    if val_results is not None:
        if len(val_results) == 3:
            val_loss, val_srcc, val_plcc = val_results
            val_mse_loss, val_rank_loss = None, None
        else:
            val_loss, val_srcc, val_plcc, val_mse_loss, val_rank_loss = val_results
        
        # 构建验证结果字符串
        val_str = f"Epoch {epoch+1}/{num_epochs} - 验证: Loss: {val_loss:.4f}, SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}"
        if val_mse_loss is not None and val_rank_loss is not None:
            val_str += f", MSE Loss: {val_mse_loss:.4f}, Rank Loss: {val_rank_loss:.4f}"
        
        # 如果是最佳模型，添加标记
        if is_best:
            val_str += " [最佳模型]"
        
        logger.info(val_str)


def log_best_model_info(logger, epoch, val_results):
    """
    记录最佳模型信息
    
    参数:
        logger (logging.Logger): 日志记录器
        epoch (int): 最佳模型的epoch
        val_results (tuple): 验证结果，包含(loss, srcc, plcc)或(loss, srcc, plcc, mse_loss, rank_loss)
    """
    logger.info("\n" + "-" * 50)
    logger.info(f"最佳模型 (Epoch {epoch+1}):")
    
    if len(val_results) == 3:
        val_loss, val_srcc, val_plcc = val_results
        logger.info(f"  验证 Loss: {val_loss:.4f}")
        logger.info(f"  验证 SRCC: {val_srcc:.4f}")
        logger.info(f"  验证 PLCC: {val_plcc:.4f}")
    else:
        val_loss, val_srcc, val_plcc, val_mse_loss, val_rank_loss = val_results
        logger.info(f"  验证 Loss: {val_loss:.4f}")
        logger.info(f"  验证 SRCC: {val_srcc:.4f}")
        logger.info(f"  验证 PLCC: {val_plcc:.4f}")
        logger.info(f"  验证 MSE Loss: {val_mse_loss:.4f}")
        logger.info(f"  验证 Rank Loss: {val_rank_loss:.4f}")
    
    logger.info("-" * 50 + "\n")


def log_evaluation_results(logger, results):
    """
    记录评估结果
    
    参数:
        logger (logging.Logger): 日志记录器
        results (dict): 评估结果字典
    """
    logger.info("\n" + "-" * 50)
    logger.info("评估结果:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.4f}")
    logger.info("-" * 50 + "\n")