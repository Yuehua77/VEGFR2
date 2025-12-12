#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据验证和可视化脚本
用于检查收集的数据质量并生成统计图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataValidator:
    """数据验证器"""
    
    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.df = None
        self.report = {}
        
        # 创建输出目录
        self.output_dir = self.data_file.parent / "validation_results"
        self.output_dir.mkdir(exist_ok=True)
    
    def load_data(self) -> bool:
        """加载数据文件"""
        try:
            print(f"正在加载数据: {self.data_file}")
            self.df = pd.read_csv(self.data_file)
            print(f"✓ 成功加载 {len(self.df)} 条记录")
            return True
        except Exception as e:
            print(f"✗ 加载失败: {e}")
            return False
    
    def check_basic_info(self):
        """检查基本信息"""
        print("\n" + "=" * 60)
        print("基本信息检查")
        print("=" * 60)
        
        print(f"数据形状: {self.df.shape}")
        print(f"列名: {list(self.df.columns)}")
        print(f"\n前5行数据:")
        print(self.df.head())
        
        self.report['total_records'] = len(self.df)
        self.report['columns'] = list(self.df.columns)
    
    def check_missing_values(self):
        """检查缺失值"""
        print("\n" + "=" * 60)
        print("缺失值检查")
        print("=" * 60)
        
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        
        print(missing_df[missing_df['缺失数量'] > 0])
        
        self.report['missing_values'] = missing_df.to_dict()
    
    def check_smiles_validity(self):
        """检查SMILES有效性"""
        print("\n" + "=" * 60)
        print("SMILES有效性检查")
        print("=" * 60)
        
        try:
            from rdkit import Chem
            
            smiles_col = self.get_smiles_column()
            if smiles_col is None:
                print("未找到SMILES列")
                return
            
            valid_count = 0
            invalid_smiles = []
            
            for idx, smiles in enumerate(self.df[smiles_col]):
                if pd.isna(smiles):
                    invalid_smiles.append((idx, smiles, "缺失"))
                    continue
                
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    valid_count += 1
                else:
                    invalid_smiles.append((idx, smiles, "无效"))
            
            total = len(self.df)
            valid_pct = (valid_count / total * 100) if total > 0 else 0
            
            print(f"总SMILES数: {total}")
            print(f"有效SMILES: {valid_count} ({valid_pct:.2f}%)")
            print(f"无效SMILES: {len(invalid_smiles)}")
            
            if invalid_smiles[:5]:
                print("\n前5个无效SMILES示例:")
                for idx, smi, reason in invalid_smiles[:5]:
                    print(f"  行{idx}: {smi} ({reason})")
            
            self.report['smiles_valid'] = valid_count
            self.report['smiles_invalid'] = len(invalid_smiles)
            
        except ImportError:
            print("⚠ 未安装RDKit，跳过SMILES验证")
    
    def check_activity_distribution(self):
        """检查活性值分布"""
        print("\n" + "=" * 60)
        print("活性值分布检查")
        print("=" * 60)
        
        activity_col = self.get_activity_column()
        if activity_col is None:
            print("未找到活性值列")
            return
        
        activities = self.df[activity_col].dropna()
        
        print(f"\n统计信息:")
        print(activities.describe())
        
        # 检查活性类别分布
        if 'activity_class' in self.df.columns:
            print(f"\n活性类别分布:")
            print(self.df['activity_class'].value_counts())
        
        self.report['activity_stats'] = activities.describe().to_dict()
    
    def visualize_data(self):
        """生成可视化图表"""
        print("\n" + "=" * 60)
        print("生成可视化图表")
        print("=" * 60)
        
        activity_col = self.get_activity_column()
        smiles_col = self.get_smiles_column()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('VEGFR2抑制剂数据分析', fontsize=16, fontweight='bold')
        
        # 1. 活性值分布直方图
        if activity_col:
            activities = self.df[activity_col].dropna()
            axes[0, 0].hist(activities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('pActivity (pIC50 or pKi)')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].set_title('活性值分布')
            axes[0, 0].axvline(activities.median(), color='red', linestyle='--', 
                              label=f'中位数: {activities.median():.2f}')
            axes[0, 0].legend()
        
        # 2. 活性类别饼图
        if 'activity_class' in self.df.columns:
            class_counts = self.df['activity_class'].value_counts()
            colors = {'high': '#2ecc71', 'medium': '#f39c12', 'low': '#e74c3c'}
            pie_colors = [colors.get(c, 'gray') for c in class_counts.index]
            
            axes[0, 1].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                          colors=pie_colors, startangle=90)
            axes[0, 1].set_title('活性类别分布')
        
        # 3. 分子性质分布（如果有）
        if 'molecular_weight' in self.df.columns:
            mw = self.df['molecular_weight'].dropna()
            axes[1, 0].hist(mw, bins=40, color='coral', edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('分子量 (Da)')
            axes[1, 0].set_ylabel('频数')
            axes[1, 0].set_title('分子量分布')
            axes[1, 0].axvline(500, color='red', linestyle='--', label='Lipinski限制 (500)')
            axes[1, 0].legend()
        
        # 4. 活性与分子量关系（如果两者都有）
        if activity_col and 'molecular_weight' in self.df.columns:
            df_plot = self.df[[activity_col, 'molecular_weight']].dropna()
            axes[1, 1].scatter(df_plot['molecular_weight'], df_plot[activity_col], 
                             alpha=0.5, s=30, c='purple')
            axes[1, 1].set_xlabel('分子量 (Da)')
            axes[1, 1].set_ylabel('pActivity')
            axes[1, 1].set_title('活性 vs 分子量')
        
        plt.tight_layout()
        
        # 保存图表
        output_file = self.output_dir / "data_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ 图表已保存: {output_file}")
        
        plt.close()
    
    def check_druglikeness(self):
        """检查类药性"""
        print("\n" + "=" * 60)
        print("类药性检查 (Lipinski's Rule of Five)")
        print("=" * 60)
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            
            smiles_col = self.get_smiles_column()
            if smiles_col is None:
                return
            
            violations = []
            
            for smiles in self.df[smiles_col].dropna():
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    continue
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                v = 0
                if mw > 500: v += 1
                if logp > 5: v += 1
                if hbd > 5: v += 1
                if hba > 10: v += 1
                
                violations.append(v)
            
            violations = np.array(violations)
            druglike_count = (violations <= 1).sum()
            total = len(violations)
            
            print(f"总化合物数: {total}")
            print(f"类药性化合物 (≤1违规): {druglike_count} ({druglike_count/total*100:.1f}%)")
            print(f"\n违规分布:")
            for i in range(5):
                count = (violations == i).sum()
                print(f"  {i}个违规: {count} ({count/total*100:.1f}%)")
            
            self.report['druglike_compounds'] = int(druglike_count)
            
        except ImportError:
            print("⚠ 未安装RDKit，跳过类药性检查")
    
    def get_smiles_column(self) -> Optional[str]:
        """自动识别SMILES列"""
        possible_names = ['canonical_smiles', 'smiles', 'SMILES', 'Canonical_SMILES']
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def get_activity_column(self) -> Optional[str]:
        """自动识别活性值列"""
        possible_names = ['pActivity', 'pIC50', 'pKi', 'pic50', 'pki']
        for name in possible_names:
            if name in self.df.columns:
                return name
        return None
    
    def generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("数据验证报告")
        print("=" * 60)
        
        report_file = self.output_dir / "validation_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("VEGFR2抑制剂数据验证报告\n")
            f.write("=" * 60 + "\n\n")
            
            for key, value in self.report.items():
                f.write(f"{key}: {value}\n")
        
        print(f"✓ 验证报告已保存: {report_file}")
    
    def run_validation(self):
        """运行完整验证流程"""
        if not self.load_data():
            return
        
        self.check_basic_info()
        self.check_missing_values()
        self.check_smiles_validity()
        self.check_activity_distribution()
        self.check_druglikeness()
        self.visualize_data()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("✓ 验证完成!")
        print(f"结果保存在: {self.output_dir}")
        print("=" * 60)


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        # 默认文件路径
        data_file = "./data/raw/vegfr2_processed.csv"
    
    print(f"数据验证工具")
    print(f"数据文件: {data_file}\n")
    
    validator = DataValidator(data_file)
    validator.run_validation()


if __name__ == "__main__":
    main()

