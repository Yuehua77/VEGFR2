"""
虚拟筛选脚本
使用训练好的模型进行虚拟筛选
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import warnings
warnings.filterwarnings('ignore')

from config import *
from preprocessing import smiles_to_features
from models import FingerprintEnhancedGAT
from utils.logger import setup_logger


class VirtualScreener:
    """虚拟筛选器"""
    
    def __init__(self,
                 model_path: Path,
                 device: torch.device,
                 logger):
        """
        初始化虚拟筛选器
        
        参数:
            model_path: 训练好的模型路径
            device: 计算设备
            logger: 日志记录器
        """
        self.device = device
        self.logger = logger
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.logger.info("虚拟筛选器初始化完成")
    
    def _load_model(self, model_path: Path):
        """加载训练好的模型"""
        self.logger.info(f"加载模型: {model_path}")
        
        # 创建模型实例
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
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def predict(self, smiles_list: list) -> np.ndarray:
        """
        预测化合物活性
        
        参数:
            smiles_list: SMILES列表
        
        返回:
            预测的活性值数组
        """
        predictions = []
        
        for smiles in tqdm(smiles_list, desc="预测中"):
            # 提取特征
            features = smiles_to_features(smiles, fingerprint_type='morgan')
            
            if features is None:
                predictions.append(np.nan)
                continue
            
            # 准备批次数据
            batch = {
                'graph': features['graph'].to(self.device),
                'fingerprint': features['fingerprint'].unsqueeze(0).to(self.device)
            }
            
            # 预测
            output = self.model(batch)
            pred = output.cpu().item()
            predictions.append(pred)
        
        return np.array(predictions)
    
    def screen_compounds(self,
                         input_file: Path,
                         output_file: Path,
                         top_k: int = 100,
                         activity_threshold: float = 7.0,
                         apply_druglikeness: bool = True):
        """
        筛选化合物
        
        参数:
            input_file: 输入文件（CSV，包含SMILES列）
            output_file: 输出文件路径
            top_k: 返回Top-K个化合物
            activity_threshold: 活性阈值
            apply_druglikeness: 是否应用类药性过滤
        """
        self.logger.info(f"开始虚拟筛选: {input_file}")
        
        # 读取数据
        df = pd.read_csv(input_file)
        self.logger.info(f"读取 {len(df)} 个化合物")
        
        # 检查SMILES列
        smiles_col = None
        for col in ['SMILES', 'smiles', 'canonical_smiles', 'Canonical_SMILES']:
            if col in df.columns:
                smiles_col = col
                break
        
        if smiles_col is None:
            raise ValueError("未找到SMILES列")
        
        # 预测活性
        self.logger.info("预测化合物活性...")
        predictions = self.predict(df[smiles_col].tolist())
        df['predicted_pIC50'] = predictions
        
        # 移除预测失败的
        df = df.dropna(subset=['predicted_pIC50'])
        self.logger.info(f"成功预测 {len(df)} 个化合物")
        
        # 应用类药性过滤
        if apply_druglikeness:
            self.logger.info("应用类药性过滤...")
            df = self._apply_druglikeness_filter(df, smiles_col)
            self.logger.info(f"类药性过滤后剩余 {len(df)} 个化合物")
        
        # 筛选高活性化合物
        df_high_activity = df[df['predicted_pIC50'] >= activity_threshold]
        self.logger.info(f"高活性化合物（pIC50 >= {activity_threshold}）: {len(df_high_activity)} 个")
        
        # 按预测活性排序
        df = df.sort_values('predicted_pIC50', ascending=False)
        
        # 选择Top-K
        df_top = df.head(top_k)
        
        # 保存结果
        df_top.to_csv(output_file, index=False)
        self.logger.info(f"筛选结果已保存: {output_file}")
        self.logger.info(f"Top-{top_k} 化合物平均预测活性: {df_top['predicted_pIC50'].mean():.2f}")
        
        return df_top
    
    def _apply_druglikeness_filter(self, df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
        """应用类药性过滤"""
        druglike_mask = []
        
        for smiles in tqdm(df[smiles_col], desc="类药性过滤"):
            mol = Chem.MolFromSmiles(smiles)
            
            if mol is None:
                druglike_mask.append(False)
                continue
            
            # 计算性质
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            tpsa = Descriptors.TPSA(mol)
            rot_bonds = Descriptors.NumRotatableBonds(mol)
            
            # 检查Lipinski规则
            druglike = True
            
            druglike_config = VIRTUAL_SCREENING_CONFIG['druglikeness_filter']
            
            if not (druglike_config['MW'][0] <= mw <= druglike_config['MW'][1]):
                druglike = False
            if not (druglike_config['LogP'][0] <= logp <= druglike_config['LogP'][1]):
                druglike = False
            if not (druglike_config['HBA'][0] <= hba <= druglike_config['HBA'][1]):
                druglike = False
            if not (druglike_config['HBD'][0] <= hbd <= druglike_config['HBD'][1]):
                druglike = False
            if not (druglike_config['TPSA'][0] <= tpsa <= druglike_config['TPSA'][1]):
                druglike = False
            if not (druglike_config['RotBonds'][0] <= rot_bonds <= druglike_config['RotBonds'][1]):
                druglike = False
            
            druglike_mask.append(druglike)
        
        return df[druglike_mask]


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger('virtual_screening', LOGS_DIR / 'virtual_screening.log')
    
    # 设置设备
    device = get_device()
    logger.info(f"使用设备: {device}")
    
    # 模型路径
    model_path = CHECKPOINT_DIR / 'best_model.pt'
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        logger.info("请先运行训练脚本")
        return
    
    # 创建虚拟筛选器
    screener = VirtualScreener(
        model_path=model_path,
        device=device,
        logger=logger
    )
    
    # 输入文件（这里使用训练数据作为示例，实际应用中应该是新的化合物库）
    input_file = VEGFR2_DATA_PATH
    output_file = PREDICTIONS_DIR / 'virtual_screening_results.csv'
    
    # 执行虚拟筛选
    results = screener.screen_compounds(
        input_file=input_file,
        output_file=output_file,
        top_k=VIRTUAL_SCREENING_CONFIG['top_k'],
        activity_threshold=VIRTUAL_SCREENING_CONFIG['high_activity_threshold'],
        apply_druglikeness=True
    )
    
    logger.info("虚拟筛选完成!")
    logger.info(f"\nTop-10 化合物:")
    logger.info(results.head(10)[['canonical_smiles', 'predicted_pIC50']])


if __name__ == "__main__":
    main()

