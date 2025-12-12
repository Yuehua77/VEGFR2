"""
数据集模块
处理数据加载、预处理和数据集创建
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .mol_features import smiles_to_features


# ============================================================================
# 主数据集类
# ============================================================================

class VEGFR2Dataset(Dataset):
    """VEGFR2抑制剂数据集"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 fingerprint_type: str = 'morgan',
                 cache_dir: Optional[Path] = None):
        """
        初始化数据集
        
        参数:
            data: DataFrame，必须包含'canonical_smiles'和'pActivity'列
            fingerprint_type: 指纹类型
            cache_dir: 缓存目录
        """
        self.data = data.reset_index(drop=True)
        self.fingerprint_type = fingerprint_type
        self.cache_dir = cache_dir
        
        # 预处理数据
        self._preprocess()
    
    def _preprocess(self):
        """预处理数据"""
        print("正在预处理数据...")
        
        self.graphs = []
        self.fingerprints = []
        self.labels = []
        self.valid_indices = []
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            smiles = row['canonical_smiles']
            y = row['pActivity']
            
            # 提取特征
            features = smiles_to_features(
                smiles,
                include_graph=True,
                include_fingerprint=True,
                fingerprint_type=self.fingerprint_type,
                y=y
            )
            
            if features is not None:
                self.graphs.append(features['graph'])
                self.fingerprints.append(features['fingerprint'])
                self.labels.append(y)
                self.valid_indices.append(idx)
        
        print(f"成功处理 {len(self.valid_indices)}/{len(self.data)} 个化合物")
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        返回:
            graph: 分子图
            fingerprint: 分子指纹
            label: 活性值
        """
        return {
            'graph': self.graphs[idx],
            'fingerprint': self.fingerprints[idx],
            'label': torch.tensor([self.labels[idx]], dtype=torch.float)
        }
    
    def save_cache(self, path: Path):
        """保存处理后的数据"""
        cache_data = {
            'graphs': self.graphs,
            'fingerprints': self.fingerprints,
            'labels': self.labels,
            'valid_indices': self.valid_indices
        }
        with open(path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"缓存已保存到: {path}")
    
    def load_cache(self, path: Path):
        """加载缓存的数据"""
        with open(path, 'rb') as f:
            cache_data = pickle.load(f)
        self.graphs = cache_data['graphs']
        self.fingerprints = cache_data['fingerprints']
        self.labels = cache_data['labels']
        self.valid_indices = cache_data['valid_indices']
        print(f"从缓存加载: {path}")


# ============================================================================
# 数据加载和划分
# ============================================================================

class DataManager:
    """数据管理器"""
    
    def __init__(self,
                 data_path: Path,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 random_seed: int = 42,
                 fingerprint_type: str = 'morgan'):
        """
        初始化数据管理器
        
        参数:
            data_path: 数据文件路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
            fingerprint_type: 指纹类型
        """
        self.data_path = Path(data_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.fingerprint_type = fingerprint_type
        
        # 加载数据
        self.df = pd.read_csv(data_path)
        print(f"加载数据: {len(self.df)} 个化合物")
        
        # 划分数据集
        self._split_data()
    
    def _split_data(self):
        """划分训练集、验证集和测试集"""
        print("\n划分数据集...")
        
        # 第一次划分：分离出测试集
        train_val_df, test_df = train_test_split(
            self.df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            shuffle=True
        )
        
        # 第二次划分：从训练+验证集中分离出验证集
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=self.random_seed,
            shuffle=True
        )
        
        print(f"训练集: {len(train_df)} 个化合物")
        print(f"验证集: {len(val_df)} 个化合物")
        print(f"测试集: {len(test_df)} 个化合物")
        
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
    
    def create_datasets(self, cache_dir: Optional[Path] = None):
        """创建数据集对象"""
        print("\n创建数据集...")
        
        # 训练集
        self.train_dataset = VEGFR2Dataset(
            self.train_df,
            fingerprint_type=self.fingerprint_type,
            cache_dir=cache_dir
        )
        
        # 验证集
        self.val_dataset = VEGFR2Dataset(
            self.val_df,
            fingerprint_type=self.fingerprint_type,
            cache_dir=cache_dir
        )
        
        # 测试集
        self.test_dataset = VEGFR2Dataset(
            self.test_df,
            fingerprint_type=self.fingerprint_type,
            cache_dir=cache_dir
        )
        
        print("数据集创建完成")
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_activity_distribution(self):
        """获取活性分布统计"""
        print("\n活性分布统计:")
        
        for name, df in [('训练集', self.train_df), 
                         ('验证集', self.val_df), 
                         ('测试集', self.test_df)]:
            print(f"\n{name}:")
            if 'activity_class' in df.columns:
                print(df['activity_class'].value_counts())
            
            print(f"  平均活性: {df['pActivity'].mean():.2f}")
            print(f"  标准差: {df['pActivity'].std():.2f}")
            print(f"  最小值: {df['pActivity'].min():.2f}")
            print(f"  最大值: {df['pActivity'].max():.2f}")


# ============================================================================
# 自定义批次整理函数
# ============================================================================

def custom_collate_fn(batch):
    """
    自定义批次整理函数，处理图数据和指纹数据
    
    参数:
        batch: 批次数据列表
    
    返回:
        batch_data: 整理后的批次数据
    """
    # 提取图、指纹和标签
    graphs = [item['graph'] for item in batch]
    fingerprints = torch.stack([item['fingerprint'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # 使用PyG的Batch类整理图数据
    batch_graph = Batch.from_data_list(graphs)
    
    return {
        'graph': batch_graph,
        'fingerprint': fingerprints,
        'label': labels
    }


def get_dataloader(dataset: Dataset,
                   batch_size: int = 32,
                   shuffle: bool = True,
                   num_workers: int = 0):
    """
    创建数据加载器
    
    参数:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
    
    返回:
        DataLoader对象
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )


# ============================================================================
# 数据标准化
# ============================================================================

class ActivityScaler:
    """活性值标准化器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, y: np.ndarray):
        """拟合标准化器"""
        y = np.array(y).reshape(-1, 1)
        self.scaler.fit(y)
        self.fitted = True
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """标准化"""
        if not self.fitted:
            raise ValueError("需要先调用fit方法")
        y = np.array(y).reshape(-1, 1)
        return self.scaler.transform(y).flatten()
    
    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """反标准化"""
        if not self.fitted:
            raise ValueError("需要先调用fit方法")
        y = np.array(y).reshape(-1, 1)
        return self.scaler.inverse_transform(y).flatten()
    
    def save(self, path: Path):
        """保存标准化器"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, path: Path):
        """加载标准化器"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.fitted = True


# ============================================================================
# 辅助函数
# ============================================================================

def analyze_dataset(dataset: VEGFR2Dataset):
    """分析数据集"""
    print("\n数据集分析:")
    print(f"样本数: {len(dataset)}")
    
    # 活性值统计
    labels = np.array(dataset.labels)
    print(f"\n活性值统计:")
    print(f"  平均值: {labels.mean():.2f}")
    print(f"  标准差: {labels.std():.2f}")
    print(f"  最小值: {labels.min():.2f}")
    print(f"  最大值: {labels.max():.2f}")
    
    # 图大小统计
    num_nodes = [g.x.shape[0] for g in dataset.graphs]
    num_edges = [g.edge_index.shape[1] for g in dataset.graphs]
    
    print(f"\n图结构统计:")
    print(f"  平均节点数: {np.mean(num_nodes):.1f}")
    print(f"  平均边数: {np.mean(num_edges):.1f}")
    print(f"  最大节点数: {np.max(num_nodes)}")
    print(f"  最大边数: {np.max(num_edges)}")


def save_splits(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                test_df: pd.DataFrame,
                output_dir: Path):
    """保存数据划分"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    print(f"数据划分已保存到: {output_dir}")


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试数据加载
    data_path = "../data_collection/data/raw/vegfr2_processed.csv"
    
    if Path(data_path).exists():
        print("测试数据加载和处理...")
        
        # 创建数据管理器
        data_manager = DataManager(
            data_path=data_path,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42,
            fingerprint_type='morgan'
        )
        
        # 查看活性分布
        data_manager.get_activity_distribution()
        
        # 创建数据集（测试前10个样本）
        print("\n测试数据预处理（仅处理10个样本）...")
        test_df = data_manager.train_df.head(10)
        test_dataset = VEGFR2Dataset(test_df, fingerprint_type='morgan')
        
        # 分析数据集
        analyze_dataset(test_dataset)
        
        # 测试数据加载器
        print("\n测试数据加载器...")
        test_loader = get_dataloader(test_dataset, batch_size=4, shuffle=False)
        
        for batch_idx, batch in enumerate(test_loader):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"  图节点数: {batch['graph'].x.shape[0]}")
            print(f"  图边数: {batch['graph'].edge_index.shape[1]}")
            print(f"  指纹形状: {batch['fingerprint'].shape}")
            print(f"  标签形状: {batch['label'].shape}")
            
            if batch_idx >= 1:  # 只测试2个batch
                break
        
        print("\n测试完成!")
    else:
        print(f"数据文件不存在: {data_path}")
        print("请先运行数据收集脚本")

