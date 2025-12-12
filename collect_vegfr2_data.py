#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VEGFR2抑制剂数据收集脚本
从多个公共数据库收集VEGFR2相关的化合物和活性数据
"""

import pandas as pd
import requests
import time
from typing import List, Dict, Optional
import json
from pathlib import Path


class VEGFR2DataCollector:
    """VEGFR2抑制剂数据收集器"""
    
    def __init__(self, output_dir: str = "./data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ChEMBL API配置
        self.chembl_base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
        # VEGFR2的ChEMBL Target ID
        self.vegfr2_target_id = "CHEMBL279"  # KDR/VEGFR2
        # A549 Cell Line Target ID
        self.a549_target_id = "CHEMBL3307651"

        
        print(f"数据将保存到: {self.output_dir.absolute()}")
    
    def collect_from_chembl(self, target_id: str, limit: int = 10000) -> pd.DataFrame:
        """
        从ChEMBL数据库收集VEGFR2抑制剂数据
        
        参数:
            limit: 最大收集数量
        
        返回:
            DataFrame包含化合物结构和活性数据
        """
        print("\n=== 从ChEMBL收集数据 ===")
        
        all_data = []
        offset = 0
        batch_size = 1000
        
        while offset < limit:
            try:
                # 查询VEGFR2的活性数据
                url = f"{self.chembl_base_url}/activity.json"
                params = {
                    'target_chembl_id': target_id,
                    'limit': min(batch_size, limit - offset),
                    'offset': offset
                }
                
                print(f"正在获取数据 {offset} - {offset + batch_size}...")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    activities = data.get('activities', [])
                    
                    if not activities:
                        print("没有更多数据")
                        break
                    
                    for activity in activities:
                        # 提取关键信息
                        record = {
                            'molecule_chembl_id': activity.get('molecule_chembl_id'),
                            'canonical_smiles': activity.get('canonical_smiles'),
                            'standard_type': activity.get('standard_type'),
                            'standard_value': activity.get('standard_value'),
                            'standard_units': activity.get('standard_units'),
                            'standard_relation': activity.get('standard_relation'),
                            'assay_chembl_id': activity.get('assay_chembl_id'),
                            'assay_type': activity.get('assay_type'),
                            'assay_description': activity.get('assay_description'),
                            'target_organism': activity.get('target_organism'),
                        }
                        all_data.append(record)
                    
                    offset += len(activities)
                    print(f"已收集 {len(all_data)} 条记录")
                    
                    # 避免请求过快
                    time.sleep(0.5)
                    
                else:
                    print(f"请求失败: {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"错误: {e}")
                break
        
        df = pd.DataFrame(all_data)
        
        if not df.empty:
            output_file = self.output_dir / "chembl_vegfr2_raw.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n[OK] ChEMBL数据已保存: {output_file}")
            print(f"  总记录数: {len(df)}")
            print(f"  唯一化合物数: {df['molecule_chembl_id'].nunique()}")
        
        return df
    
    def get_molecule_properties(self, chembl_ids: List[str]) -> pd.DataFrame:
        """
        获取化合物的详细性质
        
        参数:
            chembl_ids: ChEMBL ID列表
        
        返回:
            包含分子性质的DataFrame
        """
        print("\n=== 获取分子性质 ===")
        
        properties = []
        
        for i, chembl_id in enumerate(chembl_ids):
            if i % 100 == 0:
                print(f"进度: {i}/{len(chembl_ids)}")
            
            try:
                url = f"{self.chembl_base_url}/molecule/{chembl_id}.json"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    mol_data = response.json()
                    
                    prop = {
                        'molecule_chembl_id': chembl_id,
                        'canonical_smiles': mol_data.get('molecule_structures', {}).get('canonical_smiles'),
                        'molecular_weight': mol_data.get('molecule_properties', {}).get('full_mwt'),
                        'alogp': mol_data.get('molecule_properties', {}).get('alogp'),
                        'hba': mol_data.get('molecule_properties', {}).get('hba'),  # H-bond acceptors
                        'hbd': mol_data.get('molecule_properties', {}).get('hbd'),  # H-bond donors
                        'psa': mol_data.get('molecule_properties', {}).get('psa'),  # Polar surface area
                        'rtb': mol_data.get('molecule_properties', {}).get('rtb'),  # Rotatable bonds
                        'ro3_pass': mol_data.get('molecule_properties', {}).get('ro3_pass'),
                        'num_ro5_violations': mol_data.get('molecule_properties', {}).get('num_ro5_violations'),
                    }
                    properties.append(prop)
                
                time.sleep(0.2)  # 避免请求过快
                
            except Exception as e:
                print(f"获取 {chembl_id} 失败: {e}")
                continue
        
        df = pd.DataFrame(properties)
        
        if not df.empty:
            output_file = self.output_dir / "molecule_properties.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n[OK] 分子性质已保存: {output_file}")
        
        return df
    
    def collect_from_pubchem(self, chembl_ids: List[str]) -> pd.DataFrame:
        """
        从PubChem补充数据（通过同义词搜索ChEMBL ID）
        
        参数:
            chembl_ids: ChEMBL ID列表
        
        返回:
            包含PubChem CID和额外信息的DataFrame
        """
        print("\n=== 从PubChem收集补充数据 ===")
        
        pubchem_data = []
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        
        for i, chembl_id in enumerate(chembl_ids[:100]):  # 限制数量避免过长时间
            if i % 10 == 0:
                print(f"进度: {i}/{min(100, len(chembl_ids))}")
            
            try:
                # 通过ChEMBL ID搜索PubChem CID
                url = f"{base_url}/compound/name/{chembl_id}/cids/JSON"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    cids = data.get('IdentifierList', {}).get('CID', [])
                    
                    if cids:
                        pubchem_data.append({
                            'molecule_chembl_id': chembl_id,
                            'pubchem_cid': cids[0]
                        })
                
                time.sleep(0.3)
                
            except Exception as e:
                continue
        
        df = pd.DataFrame(pubchem_data)
        
        if not df.empty:
            output_file = self.output_dir / "pubchem_mapping.csv"
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n[OK] PubChem映射已保存: {output_file}")
        
        return df
    
    def filter_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗和过滤
        
        参数:
            df: 原始数据DataFrame
        
        返回:
            清洗后的DataFrame
        """
        print("\n=== 数据清洗 ===")
        print(f"原始数据: {len(df)} 条")
        
        # 1. 移除缺失SMILES的记录
        df = df.dropna(subset=['canonical_smiles'])
        print(f"移除缺失SMILES后: {len(df)} 条")
        
        # 2. 只保留IC50和Ki数据
        df = df[df['standard_type'].isin(['IC50', 'Ki', 'EC50', 'Kd'])]
        print(f"筛选活性类型后: {len(df)} 条")
        
        # 3. 只保留nM单位的数据
        df = df[df['standard_units'] == 'nM']
        print(f"筛选单位后: {len(df)} 条")
        
        # 4. 移除异常值（活性值应在合理范围内）
        df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
        df = df[(df['standard_value'] > 0) & (df['standard_value'] < 1e8)]
        print(f"移除异常值后: {len(df)} 条")
        
        # 5. 计算pIC50/pKi (负对数浓度)
        df['pActivity'] = -df['standard_value'].apply(lambda x: pd.np.log10(x * 1e-9) if x > 0 else None)
        df = df.dropna(subset=['pActivity'])
        
        # 6. 对于同一化合物的多个测量值，取平均
        df_agg = df.groupby('molecule_chembl_id').agg({
            'canonical_smiles': 'first',
            'pActivity': 'mean',
            'standard_type': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        }).reset_index()
        
        print(f"聚合后唯一化合物: {len(df_agg)} 个")
        
        # 7. 标注活性类别
        df_agg['activity_class'] = df_agg['pActivity'].apply(
            lambda x: 'high' if x >= 7 else ('medium' if x >= 6 else 'low')
        )
        
        print("\n活性分布:")
        print(df_agg['activity_class'].value_counts())
        
        return df_agg
    
    def save_processed_data(self, df: pd.DataFrame):
        """保存处理后的数据"""
        output_file = self.output_dir / "vegfr2_processed.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 处理后数据已保存: {output_file}")
        
        # 生成统计报告
        report = {
            'total_compounds': len(df),
            'activity_distribution': df['activity_class'].value_counts().to_dict(),
            'pActivity_stats': df['pActivity'].describe().to_dict(),
        }
        
        report_file = self.output_dir / "data_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] 统计报告已保存: {report_file}")
    
    def run_full_collection(self):
        """执行完整的数据收集流程"""
        print("=" * 60)
        print("开始收集VEGFR2抑制剂数据")
        print("=" * 60)
        
        # 步骤1: 从ChEMBL收集活性数据 (VEGFR2)
        print("\n--- 收集 VEGFR2 数据 ---")
        df_vegfr2 = self.collect_from_chembl(target_id=self.vegfr2_target_id, limit=5000)
        if not df_vegfr2.empty:
            df_vegfr2['target_type'] = 'VEGFR2'

        # 步骤1.5: 从ChEMBL收集活性数据 (A549)
        print("\n--- 收集 A549 数据 ---")
        df_a549 = self.collect_from_chembl(target_id=self.a549_target_id, limit=5000)
        if not df_a549.empty:
            df_a549['target_type'] = 'A549'
        
        # 合并数据
        df_activity = pd.concat([df_vegfr2, df_a549], ignore_index=True)

        
        if df_activity.empty:
            print("[ERROR] 未能收集到数据，请检查网络连接或API状态")
            return
        
        # 步骤2: 获取唯一的ChEMBL ID
        unique_ids = df_activity['molecule_chembl_id'].dropna().unique()[:500]
        print(f"\n共有 {len(unique_ids)} 个唯一化合物")
        
        # 步骤3: 获取分子性质
        df_properties = self.get_molecule_properties(unique_ids.tolist())
        
        # 步骤4: 数据清洗
        df_cleaned = self.filter_and_clean_data(df_activity)
        
        # 步骤5: 合并性质数据
        if not df_properties.empty:
            df_final = df_cleaned.merge(
                df_properties,
                on='molecule_chembl_id',
                how='left',
                suffixes=('', '_prop')
            )
        else:
            df_final = df_cleaned
        
        # 步骤6: 保存最终数据
        self.save_processed_data(df_final)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 数据收集完成!")
        print("=" * 60)
        
        return df_final


def main():
    """主函数"""
    collector = VEGFR2DataCollector(output_dir="./data/raw")
    df = collector.run_full_collection()
    
    if df is not None:
        print("\n数据预览:")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        print(f"\n列名: {df.columns.tolist()}")


if __name__ == "__main__":
    main()

