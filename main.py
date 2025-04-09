import os  
import argparse  
import torch  
import torch.distributed as dist  
from datetime import datetime  
from tqdm import tqdm

from evaluate import evaluate  
from config import get_train_config, get_eval_config  
from dataset_utils import get_data_transforms, get_dataloaders  
from model.models import ViTForIQA, ResNetViTForIQA, ResNetViTConcatForIQA, SwinForIQA
from train import train_epoch, validate_epoch  
from losses import MSELoss, AdaptiveBoundaryRankingLoss, CombinedLoss  
from logger_utils import setup_logger, log_section_start, log_section_end, log_config, log_dataset_info, log_epoch_results, log_best_model_info  

def main():  
    # 创建命令行参数解析器  
    parser = argparse.ArgumentParser(description='ViT图像质量评估项目')  
    
    # 添加local_rank参数(torchrun需要)  
    parser.add_argument('--local_rank', type=int, default=os.getenv('LOCAL_RANK', -1),  
                      help='分布式训练的本地进程序号')  
    
    # 其他参数保持不变...  
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate'],  
                        help='运行模式: train或evaluate')  
    parser.add_argument('--data_dir', type=str, help='图像数据目录')  
    parser.add_argument('--label_file', type=str, help='标签文件路径')  
    parser.add_argument('--output_dir', type=str, help='输出目录，用于保存模型和结果')  
    parser.add_argument('--model_type', type=str,   
                    choices=['vit', 'resnet_vit', 'resnet_vit_concat', 'swin'],  
                    help='模型类型: vit、resnet_vit、resnet_vit_concat或swin')  
    parser.add_argument('--swin_size', type=str, default='tiny', choices=['tiny', 'small', 'base', 'large'],  
                    help='Swin Transformer模型大小: tiny, small, base, large')  
    parser.add_argument('--batch_size', type=int, help='批大小')  
    parser.add_argument('--num_workers', type=int, help='数据加载的工作线程数')  
    parser.add_argument('--no_cuda', action='store_true', help='禁用CUDA')  
    parser.add_argument('--num_epochs', type=int, help='训练轮数')  
    parser.add_argument('--learning_rate', type=float, help='学习率')  
    parser.add_argument('--weight_decay', type=float, help='权重衰减')  
    parser.add_argument('--freeze_backbone', action='store_true', help='是否冻结backbone参数')  
    parser.add_argument('--seed', type=int, help='随机种子')  
    parser.add_argument('--experiment_name', type=str, help='实验名称，用于创建实验目录')  
    parser.add_argument('--save_frequency', type=int, help='模型保存频率')  
    parser.add_argument('--model_path', type=str, help='训练好的模型路径，仅在evaluate模式下需要')  
    parser.add_argument('--visualize', action='store_true', help='是否可视化预测结果，仅在evaluate模式下有效')  
    # 添加损失函数类型选择参数
    parser.add_argument('--loss_type', type=str, default='combined', choices=['mse', 'combined'],
                    help='损失函数类型: mse(均方误差)或combined(MSE+排序损失的组合)')
    # 排序损失函数类型选择参数
    parser.add_argument('--rank_type', type=str, default='adaptive', choices=['adaptive', 'exponential'],
                    help='排序损失函数类型: adaptive(自适应边界排序损失)或exponential(指数形式排序损失)')
    # 损失函数相关参数
    parser.add_argument('--mse_weight', type=float, default=1.0, help='MSE损失的权重')
    parser.add_argument('--rank_weight', type=float, default=1.0, help='排序损失的权重')
    parser.add_argument('--beta', type=float, default=0.3, help='自适应边界强度控制因子(仅用于自适应边界排序损失)')
    parser.add_argument('--gamma', type=float, default=0.1, help='自适应边界非线性调整因子(仅用于自适应边界排序损失)')  
    
    args = parser.parse_args()  
    
    # 根据模式获取配置  
    if args.mode == 'train':  
        config = get_train_config()  
    elif args.mode == 'evaluate':  
        config = get_eval_config()  
        if args.model_path is None:  
            parser.error("evaluate模式需要指定--model_path参数")  
    
    # 更新配置  
    config_updates = {}  
    for key, value in vars(args).items():  
        if key != 'mode' and value is not None:  
            if key == 'no_cuda':  
                config_updates['use_cuda'] = not value  
            else:  
                config_updates[key] = value  
    
    config.update(**config_updates)  
    
    # 获取本地rank  
    local_rank = args.local_rank  
    
    # 检查是否为分布式模式  
    is_distributed = local_rank != -1  
    
    if is_distributed:  
        # 设置当前设备  
        torch.cuda.set_device(local_rank)  
        # 初始化分布式进程组  
        dist.init_process_group(backend='nccl', init_method='env://')  
        world_size = dist.get_world_size()  
        rank = dist.get_rank()  
    else:  
        world_size = 1  
        rank = 0  
    
    # 配置设备  
    device = torch.device(f'cuda:{local_rank}' if local_rank != -1 else 'cuda:0' if torch.cuda.is_available() else 'cpu')  
    
    # 设置随机种子  
    if config.seed is not None:  
        seed = config.seed + rank if is_distributed else config.seed  
        torch.manual_seed(seed)  
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True  
        
    if config.mode == 'train':  
        # 创建实验目录(只在主进程中)  
        if rank == 0:  
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
            experiment_dir_name = f"{config.experiment_name}_{config.model_type}_{os.path.basename(config.label_file).split('.')[0]}_{config.batch_size}_{timestamp}"  
            experiment_dir = os.path.join(config.output_dir, experiment_dir_name)  
            
            logs_dir = os.path.join(experiment_dir, 'logs')  
            models_dir = os.path.join(experiment_dir, 'models')  
            tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')  
            
            os.makedirs(logs_dir, exist_ok=True)  
            os.makedirs(models_dir, exist_ok=True)  
            os.makedirs(tensorboard_dir, exist_ok=True)  
            
            # 保存配置  
            with open(os.path.join(logs_dir, 'config.txt'), 'w') as f:  
                f.write(str(config))  
            
            # 设置日志记录器
            logger = setup_logger(experiment_dir)
            log_section_start(logger, "实验初始化")
            logger.info(f"实验目录: {experiment_dir}")
            log_config(logger, config)
            log_section_end(logger, "实验初始化")
                
            print(f"实验目录: {experiment_dir}")  
        else:  
            experiment_dir = None  
            logs_dir = None  
            models_dir = None  
            tensorboard_dir = None  
            logger = None  
        
        # 获取数据集和加载器，使用get_dataloaders函数  
        dataloaders = get_dataloaders(  
            root_dir=config.data_dir,  
            label_file=config.label_file,  
            batch_size=config.batch_size,  
            num_workers=config.num_workers,  
            random_state=config.seed  
        )  

        train_dataset = dataloaders['train'].dataset  
        val_dataset = dataloaders['val'].dataset  

        # 记录数据集信息
        if rank == 0:
            log_section_start(logger, "数据集信息")
            log_dataset_info(logger, train_dataset, val_dataset)
            log_section_end(logger, "数据集信息")  

        # 如果是分布式训练，重新创建train_loader  
        if is_distributed:  
            train_sampler = torch.utils.data.distributed.DistributedSampler(  
                train_dataset,  
                num_replicas=world_size,  
                rank=rank,  
                shuffle=True,  
                seed=config.seed  
            )  
            
            train_loader = torch.utils.data.DataLoader(  
                train_dataset,  
                batch_size=config.batch_size,  
                shuffle=False,  # 使用DistributedSampler时必须为False  
                num_workers=config.num_workers,  
                pin_memory=True,  
                sampler=train_sampler  
            )  
        else:  
            train_sampler = None  
            train_loader = dataloaders['train']  

        val_loader = dataloaders['val']  
        
        
        # 创建模型  
        if rank == 0:
            log_section_start(logger, "模型初始化")
            logger.info(f"创建模型类型: {config.model_type}")
            
        if config.model_type == 'vit':  
            model = ViTForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)  
        elif config.model_type == 'resnet_vit':  
            model = ResNetViTForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)  
        elif config.model_type == 'resnet_vit_concat':  
            model = ResNetViTConcatForIQA(pretrained=True, freeze_backbone=config.freeze_backbone)
        elif config.model_type == 'swin':  
            model = SwinForIQA(  
                pretrained=True,   
                freeze_backbone=config.freeze_backbone,  
                model_size=getattr(config, 'swin_size', 'tiny')  # 允许通过配置选择模型大小  
            )  
        else:  
            raise ValueError(f"不支持的模型类型: {config.model_type}")  
            
        if rank == 0:
            logger.info(f"模型创建完成: {config.model_type}")
            log_section_end(logger, "模型初始化")  
        
        # 将模型移动到设备  
        model = model.to(device)  
        
        # 使用DistributedDataParallel包装模型  
        if is_distributed:  
            model = torch.nn.parallel.DistributedDataParallel(  
                model,   
                device_ids=[local_rank],  
                output_device=local_rank  
            )  
        
        # 创建优化器和损失函数  
        if rank == 0:
            log_section_start(logger, "损失函数和优化器")
            
        if config.loss_type == 'combined':
            # 获取排序损失函数类型
            rank_type = getattr(config, 'rank_type', 'adaptive')
            
            # 使用组合损失函数（MSE + 排序损失）  
            criterion = CombinedLoss(
                mse_weight=getattr(config, 'mse_weight', 1.0),  
                rank_weight=getattr(config, 'rank_weight', 0.2),  
                beta=getattr(config, 'beta', 0.3),  
                gamma=getattr(config, 'gamma', 0.1),
                rank_type=rank_type
            )  
            print(f"使用组合损失函数: MSE权重={getattr(config, 'mse_weight', 1.0)}, 排序权重={getattr(config, 'rank_weight', 0.2)}")  
            
            if rank == 0:
                logger.info(f"使用组合损失函数: MSE权重={getattr(config, 'mse_weight', 1.0)}, 排序权重={getattr(config, 'rank_weight', 0.2)}")
            
            if rank_type == 'adaptive':
                print(f"使用自适应边界排序损失: beta={getattr(config, 'beta', 0.3)}, gamma={getattr(config, 'gamma', 0.1)}")  
                if rank == 0:
                    logger.info(f"使用自适应边界排序损失: beta={getattr(config, 'beta', 0.3)}, gamma={getattr(config, 'gamma', 0.1)}")
            elif rank_type == 'exponential':
                print("使用指数形式排序损失")
                if rank == 0:
                    logger.info("使用指数形式排序损失")
            
        else:  
            # 使用传统的MSE损失  
            criterion = MSELoss()  
            print("使用传统MSE损失函数")  
            if rank == 0:
                logger.info("使用传统MSE损失函数")  
            
        optimizer = torch.optim.Adam(  
            model.parameters(),   
            lr=config.learning_rate,   
            weight_decay=config.weight_decay  
        )  
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  
            optimizer,   
            mode='min',   
            factor=0.5,   
            patience=5,   
            verbose=(rank==0)  
        )  
        
        if rank == 0:
            logger.info(f"优化器: Adam, 学习率: {config.learning_rate}, 权重衰减: {config.weight_decay}")
            logger.info("学习率调度器: ReduceLROnPlateau, 模式: min, 因子: 0.5, 耐心值: 5")
            log_section_end(logger, "损失函数和优化器")  
        
        # 训练循环  
        best_val_metric = -1.0  
        best_epoch = 0  
        
        # 如果是主进程，初始化TensorBoard  
        if rank == 0:  
            from torch.utils.tensorboard import SummaryWriter  
            writer = SummaryWriter(log_dir=tensorboard_dir)  
        else:  
            writer = None  
        
        # 训练循环  
        # 使用tqdm包装epoch循环，只在主进程中显示进度
        epoch_iterator = tqdm(range(config.num_epochs), desc="训练进度", position=0, leave=True) if rank == 0 else range(config.num_epochs)
        
        if rank == 0:
            log_section_start(logger, "训练过程")
            logger.info(f"开始训练，总共 {config.num_epochs} 个epochs")
        
        for epoch in epoch_iterator:  
            if is_distributed:  
                train_sampler.set_epoch(epoch)  
            
            # 训练一个epoch  
            train_results = train_epoch(  
                model, train_loader, criterion, optimizer, device, epoch, config.num_epochs, logger  
            )
            
            # 处理返回结果
            if isinstance(criterion, CombinedLoss):
                train_loss, train_srcc, train_plcc, train_mse_loss, train_rank_loss = train_results
            else:
                train_loss, train_srcc, train_plcc = train_results
                train_mse_loss, train_rank_loss = train_loss, 0.0
            
            # 验证  
            val_results = validate_epoch(  
                model, val_loader, criterion, device, epoch, config.num_epochs, logger  
            )
            
            # 处理返回结果
            if isinstance(criterion, CombinedLoss):
                val_loss, val_srcc, val_plcc, val_mse_loss, val_rank_loss = val_results
            else:
                val_loss, val_srcc, val_plcc = val_results
                val_mse_loss, val_rank_loss = val_loss, 0.0
            
            
            # 更新学习率  
            scheduler.step(val_loss)  
            
            # 记录到TensorBoard(如果是主进程)  
            if rank == 0:  
                writer.add_scalar('Loss/train', train_loss, epoch)  
                writer.add_scalar('Loss/val', val_loss, epoch)  
                writer.add_scalar('SRCC/train', train_srcc, epoch)  
                writer.add_scalar('SRCC/val', val_srcc, epoch)  
                writer.add_scalar('PLCC/train', train_plcc, epoch)  
                writer.add_scalar('PLCC/val', val_plcc, epoch)  
                
                # 如果使用组合损失函数，记录额外的损失信息
                if isinstance(criterion, CombinedLoss):
                    writer.add_scalar('MSE_Loss/train', train_mse_loss, epoch)
                    writer.add_scalar('MSE_Loss/val', val_mse_loss, epoch)
                    writer.add_scalar('Rank_Loss/train', train_rank_loss, epoch)
                    writer.add_scalar('Rank_Loss/val', val_rank_loss, epoch)  
                
                print(f"Epoch {epoch+1}/{config.num_epochs} | "  
                      f"Train Loss: {train_loss:.4f}, SRCC: {train_srcc:.4f}, PLCC: {train_plcc:.4f} | "  
                      f"Val Loss: {val_loss:.4f}, SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}")  
                
                # 记录训练和验证结果到日志
                if isinstance(criterion, CombinedLoss):
                    train_results = (train_loss, train_srcc, train_plcc, train_mse_loss, train_rank_loss)
                    val_results = (val_loss, val_srcc, val_plcc, val_mse_loss, val_rank_loss)
                else:
                    train_results = (train_loss, train_srcc, train_plcc)
                    val_results = (val_loss, val_srcc, val_plcc)  
                
                # 记录epoch结果到日志
                is_best = False
                val_metric = val_srcc + val_plcc
                if val_metric > best_val_metric:
                    is_best = True
                    best_val_metric = val_metric
                    best_epoch = epoch
                
                # 记录到日志系统
                log_epoch_results(logger, epoch, config.num_epochs, train_results, val_results, is_best)
                
                # 保存最佳模型  
                if is_best:  
                    # 保存模型  
                    model_to_save = model.module if hasattr(model, 'module') else model  
                    torch.save({  
                        'epoch': epoch,  
                        'model_state_dict': model_to_save.state_dict(),  
                        'optimizer_state_dict': optimizer.state_dict(),  
                        'val_srcc': val_srcc,  
                        'val_plcc': val_plcc,  
                        'val_metric': val_metric,  
                    }, os.path.join(models_dir, 'best_model.pth'))  
                    
                    print(f"保存最佳模型，验证SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, 总和: {val_metric:.4f}")  
                    logger.info(f"保存最佳模型，验证SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, 总和: {val_metric:.4f}")
                    
                    # 记录更好的结果到日志文件（追加模式）
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(os.path.join(logs_dir, 'val_results.txt'), 'a') as f:
                        f.write(f"[{current_time}] Epoch {epoch+1}/{config.num_epochs} - 新的最佳结果\n")
                        f.write(f"验证 SRCC: {val_srcc:.4f}, PLCC: {val_plcc:.4f}, 总和: {val_metric:.4f}\n\n")  
                
                # 按频率保存模型  
                if (epoch + 1) % config.save_frequency == 0:  
                    model_to_save = model.module if hasattr(model, 'module') else model  
                    torch.save({  
                        'epoch': epoch,  
                        'model_state_dict': model_to_save.state_dict(),  
                        'optimizer_state_dict': optimizer.state_dict(),  
                        'val_srcc': val_srcc,  
                        'val_plcc': val_plcc,  
                        'val_metric': val_metric,  
                    }, os.path.join(models_dir, f'model_epoch_{epoch+1}.pth'))  
                    print(f"按照保存频率保存模型，epoch: {epoch+1}")  
        
        # 训练结束后，在主进程上记录最终结果
        if rank == 0:
            # 记录最佳模型信息
            print(f"\n训练完成，最佳模型来自第{best_epoch+1}个epoch，验证指标总和: {best_val_metric:.4f}")
            
            # 记录最终训练结果到日志文件（追加模式）
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(logs_dir, 'val_results.txt'), 'a') as f:
                f.write(f"\n[{current_time}] 训练完成 - 最佳模型来自第{best_epoch+1}个epoch\n")
                f.write(f"最终验证指标总和: {best_val_metric:.4f}\n\n")
            
            # 记录到日志系统
            log_section_end(logger, "训练过程")
            log_section_start(logger, "训练总结")
            logger.info(f"训练完成，共 {config.num_epochs} 个epochs")
            
            # 记录最佳模型信息
            if isinstance(criterion, CombinedLoss):
                best_val_results = (val_loss, val_srcc, val_plcc, val_mse_loss, val_rank_loss)
            else:
                best_val_results = (val_loss, val_srcc, val_plcc)
            log_best_model_info(logger, best_epoch, best_val_results)
            log_section_end(logger, "训练总结")
            
            # 关闭TensorBoard
            writer.close()  
    
    elif config.mode == 'evaluate':  
        # 创建评估日志目录
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir, exist_ok=True)
        
        # 设置评估日志
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = os.path.join(config.output_dir, f"eval_{timestamp}")
        os.makedirs(eval_dir, exist_ok=True)
        
        # 设置日志记录器
        logger = setup_logger(eval_dir, f"eval_{timestamp}.log")
        log_section_start(logger, "评估开始")
        logger.info(f"模型路径: {config.model_path}")
        log_config(logger, config)
        
        # 调用原始评估函数并获取结果
        results = evaluate(config, logger)
        
        # 记录评估结果
        if results:
            log_section_start(logger, "评估结果")
            for key, value in results.items():
                logger.info(f"{key}: {value:.4f}")
            log_section_end(logger, "评估结果")
        
        log_section_end(logger, "评估结束")  

if __name__ == '__main__':  
    main()