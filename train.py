"""
模型训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime

from config import *
from preprocessing import DataManager, get_dataloader
from models import FingerprintEnhancedGAT, count_parameters
from utils.metrics import MetricsCalculator
from utils.early_stopping import EarlyStopping
from utils.logger import setup_logger


class Trainer:
    """模型训练器"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader,
                 val_loader,
                 test_loader,
                 config: dict,
                 device: torch.device,
                 logger: logging.Logger):
        """
        初始化训练器
        
        参数:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器
            config: 训练配置
            device: 计算设备
            logger: 日志记录器
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # 损失函数
        if config['loss_function'] == 'MSE':
            self.criterion = nn.MSELoss()
        elif config['loss_function'] == 'MAE':
            self.criterion = nn.L1Loss()
        elif config['loss_function'] == 'Huber':
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # 优化器
        if config['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0)
            )
        elif config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0)
            )
        elif config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config.get('weight_decay', 0)
            )
        
        # 学习率调度器
        if config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['lr_scheduler']['factor'],
                patience=config['lr_scheduler']['patience'],
                min_lr=config['lr_scheduler']['min_lr'],
                verbose=True
            )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            verbose=True,
            path=CHECKPOINT_DIR / 'best_model.pt',
            logger=logger
        )
        
        # 指标计算器
        self.metrics_calculator = MetricsCalculator()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            batch = self._to_device(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['label'])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy().flatten())
            all_labels.extend(batch['label'].detach().cpu().numpy().flatten())
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        return avg_loss, metrics
    
    @torch.no_grad()
    def validate(self, loader, desc='Val'):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f'[{desc}]')
        
        for batch in pbar:
            batch = self._to_device(batch)
            
            outputs = self.model(batch)
            loss = self.criterion(outputs, batch['label'])
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(batch['label'].cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_preds)
        )
        
        return avg_loss, metrics, np.array(all_preds), np.array(all_labels)
    
    def train(self, num_epochs: int):
        """完整训练流程"""
        self.logger.info("开始训练...")
        self.logger.info(f"模型参数数量: {count_parameters(self.model):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_metrics, _, _ = self.validate(self.val_loader, 'Val')
            
            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['learning_rates'].append(current_lr)
            
            # 日志
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            self.logger.info(f"Train Metrics: {train_metrics}")
            self.logger.info(f"Val Metrics: {val_metrics}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
            
            # 定期保存
            if epoch % self.config['save_frequency'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # 早停检查
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                self.logger.info(f"早停触发于第 {epoch} 轮")
                break
        
        # 加载最佳模型
        self.logger.info("加载最佳模型进行最终测试...")
        self.load_checkpoint(CHECKPOINT_DIR / 'best_model.pt')
        
        # 最终测试
        test_loss, test_metrics, test_preds, test_labels = self.validate(
            self.test_loader, 'Test'
        )
        
        self.logger.info(f"最终测试 - Loss: {test_loss:.4f}")
        self.logger.info(f"测试集指标: {test_metrics}")
        
        # 保存训练历史
        self.save_history()
        
        return test_metrics, test_preds, test_labels
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        if is_best:
            path = CHECKPOINT_DIR / 'best_model.pt'
        else:
            path = CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        self.logger.info(f"检查点已保存: {path}")
    
    def load_checkpoint(self, path: Path):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"模型已加载: {path}")
    
    def save_history(self):
        """保存训练历史"""
        history_path = LOGS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            # 转换numpy类型为Python原生类型
            history_serializable = {}
            for key, value in self.history.items():
                if isinstance(value, list):
                    history_serializable[key] = [
                        {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                         for k, v in item.items()} if isinstance(item, dict) else float(item)
                        for item in value
                    ]
                else:
                    history_serializable[key] = value
            
            json.dump(history_serializable, f, indent=2)
        
        self.logger.info(f"训练历史已保存: {history_path}")
    
    def _to_device(self, batch):
        """将批次数据移到设备"""
        batch['graph'] = batch['graph'].to(self.device)
        batch['fingerprint'] = batch['fingerprint'].to(self.device)
        batch['label'] = batch['label'].to(self.device)
        return batch


def main():
    """主训练函数"""
    # 设置日志
    logger = setup_logger('training', LOGS_DIR / 'training.log')
    
    # 打印配置
    print_config()
    logger.info(f"实验: {EXPERIMENT_NAME} {VERSION}")
    
    # 设置设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # 加载数据
    logger.info("加载数据...")
    data_manager = DataManager(
        data_path=VEGFR2_DATA_PATH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
        fingerprint_type='morgan'
    )
    
    # 创建数据集
    train_dataset, val_dataset, test_dataset = data_manager.create_datasets(
        cache_dir=PROCESSED_DATA_DIR
    )
    
    # 创建数据加载器
    train_loader = get_dataloader(
        train_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=True
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False
    )
    test_loader = get_dataloader(
        test_dataset,
        batch_size=TRAIN_CONFIG['batch_size'],
        shuffle=False
    )
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")
    
    # 创建模型
    logger.info("创建模型...")
    model_config = {
        'node_feature_dim': NODE_FEATURE_DIM,
        'edge_feature_dim': EDGE_FEATURE_DIM,
        'fingerprint_dim': FINGERPRINT_CONFIG['morgan']['n_bits'],
        'gat_hidden_channels': GAT_CONFIG['hidden_channels'],
        'gat_num_layers': GAT_CONFIG['num_layers'],
        'gat_num_heads': GAT_CONFIG['num_heads'],
        'gat_dropout': GAT_CONFIG['dropout'],
        'fingerprint_hidden_dims': FINGERPRINT_ENCODER_CONFIG['hidden_dims'],
        'fingerprint_dropout': FINGERPRINT_ENCODER_CONFIG['dropout'],
        'fusion_dims': FUSION_CONFIG['fusion_dims'],
        'fusion_dropout': FUSION_CONFIG['dropout'],
        'output_dim': OUTPUT_DIM
    }
    
    model = FingerprintEnhancedGAT(**model_config)
    logger.info(f"模型参数数量: {count_parameters(model):,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=TRAIN_CONFIG,
        device=device,
        logger=logger
    )
    
    # 开始训练
    start_time = datetime.now()
    test_metrics, test_preds, test_labels = trainer.train(TRAIN_CONFIG['num_epochs'])
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds()
    logger.info(f"训练总时间: {training_time:.2f}秒 ({training_time/60:.2f}分钟)")
    
    # 保存最终结果
    results = {
        'experiment': EXPERIMENT_NAME,
        'version': VERSION,
        'test_metrics': test_metrics,
        'training_time': training_time,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = RESULTS_DIR / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"最终结果已保存: {results_path}")
    logger.info("训练完成!")


if __name__ == "__main__":
    main()

