"""
配置文件
包含所有超参数和路径配置
"""

import os
from pathlib import Path

# ============================================================================
# 路径配置
# ============================================================================

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROTEIN_STRUCTURE_DIR = DATA_DIR / "protein_structures"

# 模型目录
MODEL_DIR = ROOT_DIR / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOGS_DIR = MODEL_DIR / "logs"

# 结果目录
RESULTS_DIR = ROOT_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
VISUALIZATION_DIR = RESULTS_DIR / "visualizations"

# 创建所有必要的目录
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, PROTEIN_STRUCTURE_DIR,
                 CHECKPOINT_DIR, LOGS_DIR, PREDICTIONS_DIR, VISUALIZATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 数据配置
# ============================================================================

# 数据文件路径
VEGFR2_DATA_PATH = RAW_DATA_DIR / "vegfr2_processed.csv"

# 数据划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子
RANDOM_SEED = 42

# ============================================================================
# 分子特征配置
# ============================================================================

# 分子指纹配置
FINGERPRINT_CONFIG = {
    'morgan': {
        'radius': 2,
        'n_bits': 1024
    },
    'maccs': {
        'n_bits': 167
    },
    'rdkit': {
        'n_bits': 2048
    }
}

# 图节点特征维度
NODE_FEATURE_DIM = 44  # 原子特征维度

# 图边特征维度
EDGE_FEATURE_DIM = 14  # 键特征维度

# 最大原子数（用于padding）
MAX_NUM_ATOMS = 150

# ============================================================================
# 模型配置
# ============================================================================

# 图注意力网络配置
GAT_CONFIG = {
    'hidden_channels': 44,  # 修改点：44 * 10 heads = 440，与论文一致
    'num_layers': 2,        # 论文主要提到了GAT层和GCN层，通常是2层结构
    'num_heads': 10,
    'dropout': 0.2,
    'concat_heads': True
}

FINGERPRINT_ENCODER_CONFIG = {
    'input_dim': 1024,
    'hidden_dims': [100, 25],  # 论文指定的特殊维度
    'output_dim': 10,          # 论文最后输出了10维向量
    'dropout': 0.3
}

# 图分支后处理配置 (对应 Figure 2A)
GRAPH_POST_PROCESS_CONFIG = {
    'flatten_dim': 880,      # GAT/GCN output 440 * 2 (Max+Mean Pooling)
    'hidden_dims': [1500, 50],
    'output_dim': 10         # 最终输出10维，与指纹分支对齐
}

# 融合层配置
FUSION_CONFIG = {
    'graph_dim': 10,        # 图分支输出维度
    'fingerprint_dim': 10,  # 指纹分支输出维度
    'fusion_dims': [],      # 原文似乎是直接 Concat -> Dense(1)，没有隐藏层，或者只有一层
    'dropout': 0.0          # 最后一层通常不需要 Dropout
}
# 输出层配置
OUTPUT_DIM = 1  # 

# ============================================================================
# 训练配置
# ============================================================================

TRAIN_CONFIG = {
    # 批次大小
    'batch_size': 32,
    
    # 学习率
    'learning_rate': 0.0001,
    
    # 训练轮数
    'num_epochs': 100,
    
    # 早停参数
    'early_stopping_patience': 18, #18或28
    
    # 学习率调度
    'lr_scheduler': {
        'type': 'ReduceLROnPlateau',
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6
    },
    
    # 权重衰减
    'weight_decay': 1e-5,
    
    # 梯度裁剪
    'gradient_clip': 1.0,
    
    # 损失函数
    'loss_function': 'BCE',  # 'MSE', 'MAE', 'Huber'
    
    # 优化器
    'optimizer': 'Adam',  # 'Adam', 'AdamW', 'SGD'
    
    # 是否使用GPU
    'use_gpu': True,
    
    # 保存检查点的频率
    'save_frequency': 10,
    
    # 验证频率
    'val_frequency': 1
}

# ============================================================================
# 评估指标配置
# ============================================================================

METRICS = ['AUC', 'BA', 'F1', 'MCC']

# ============================================================================
# 虚拟筛选配置
# ============================================================================

VIRTUAL_SCREENING_CONFIG = {
    # 活性阈值
    'high_activity_threshold': 7.0,  # pIC50 >= 7.0
    
    # Top-K筛选
    'top_k': 100,
    
    # 类药性过滤
    'druglikeness_filter': {
        'MW': (150, 500),  # 分子量
        'LogP': (-2, 5),   # 脂水分配系数
        'HBA': (0, 10),    # 氢键受体
        'HBD': (0, 5),     # 氢键供体
        'TPSA': (0, 140),  # 极性表面积
        'RotBonds': (0, 10)  # 可旋转键数
    }
}

# ============================================================================
# 分子对接配置
# ============================================================================

DOCKING_CONFIG = {
    # 蛋白质结构文件
    'receptor_pdb': '3WZE',  # VEGFR2与Sorafenib复合物
    
    # 对接软件
    'software': 'AutoDock Vina',  # 'AutoDock Vina', 'Glide', 'GOLD'
    
    # 对接盒子配置
    'box_center': [0, 0, 0],  # 需要根据实际结构调整
    'box_size': [25, 25, 25],  # Å
    
    # 对接参数
    'exhaustiveness': 8,
    'num_modes': 10,
    
    # 能量范围
    'energy_range': 3.0
}

# ============================================================================
# 分子动力学模拟配置
# ============================================================================

MD_CONFIG = {
    # 模拟软件
    'software': 'GROMACS',  # 'GROMACS', 'AMBER', 'NAMD'
    
    # 力场
    'force_field': 'AMBER99SB-ILDN',
    
    # 水模型
    'water_model': 'TIP3P',
    
    # 盒子类型
    'box_type': 'dodecahedron',
    
    # 盒子边界距离
    'box_distance': 1.0,  # nm
    
    # 离子浓度
    'ion_concentration': 0.15,  # M (生理盐浓度)
    
    # 能量最小化
    'minimization': {
        'method': 'steepest descent',
        'steps': 50000
    },
    
    # NVT平衡
    'nvt_equilibration': {
        'temperature': 300,  # K
        'time': 100,  # ps
    },
    
    # NPT平衡
    'npt_equilibration': {
        'temperature': 300,  # K
        'pressure': 1.0,  # bar
        'time': 100,  # ps
    },
    
    # 生产模拟
    'production': {
        'temperature': 300,  # K
        'pressure': 1.0,  # bar
        'time': 50,  # ns (可根据需要调整)
    },
    
    # 分析参数
    'analysis': {
        'rmsd': True,
        'rmsf': True,
        'rg': True,  # 回旋半径
        'sasa': True,  # 溶剂可及表面积
        'hbond': True,  # 氢键分析
        'binding_energy': 'MM-PBSA'  # 或 'MM-GBSA'
    }
}

# ============================================================================
# 可视化配置
# ============================================================================

VISUALIZATION_CONFIG = {
    'figure_dpi': 300,
    'figure_format': 'png',
    'color_scheme': {
        'train': '#2E86AB',
        'val': '#A23B72',
        'test': '#F18F01',
        'high_activity': '#06A77D',
        'medium_activity': '#F77F00',
        'low_activity': '#D62828'
    },
    'font_family': 'Arial',
    'font_size': 12
}

# ============================================================================
# 日志配置
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'training.log'),
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# ============================================================================
# 实验配置
# ============================================================================

EXPERIMENT_NAME = "VEGFR2_Fingerprint_GAT"
VERSION = "v1.0"

# ============================================================================
# 辅助函数
# ============================================================================

def get_device():
    """获取计算设备"""
    import torch
    if TRAIN_CONFIG['use_gpu'] and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def print_config():
    """打印配置信息"""
    print("=" * 80)
    print(f"实验名称: {EXPERIMENT_NAME} {VERSION}")
    print("=" * 80)
    print(f"数据路径: {VEGFR2_DATA_PATH}")
    print(f"模型保存路径: {CHECKPOINT_DIR}")
    print(f"日志保存路径: {LOGS_DIR}")
    print(f"结果保存路径: {RESULTS_DIR}")
    print(f"计算设备: {get_device()}")
    print("=" * 80)

