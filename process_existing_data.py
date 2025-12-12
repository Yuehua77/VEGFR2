#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理已收集的ChEMBL数据
"""

import pandas as pd
import numpy as np
import json
import requests
from pathlib import Path


def process_data(input_path=None, output_path=None):
    """处理已下载的数据"""
    
    # 读取原始数据
    if input_path:
        input_file = Path(input_path)
    else:
        input_file = Path("./data/raw/chembl_vegfr2_raw.csv")
    
    if not input_file.exists():
        print("[ERROR] 找不到数据文件!")
        return
    
    print("正在读取数据...")
    df = pd.read_csv(input_file)
    print(f"原始数据: {len(df)} 条")
    
    # 数据清洗
    print("\n=== 数据清洗 ===")
    
    # 1. 移除缺失SMILES的记录
    df = df.dropna(subset=['canonical_smiles'])
    print(f"移除缺失SMILES后: {len(df)} 条")
    
    # 2. 只保留IC50和Ki数据
    df = df[df['standard_type'].isin(['IC50', 'Ki', 'EC50', 'Kd'])]
    print(f"筛选活性类型后: {len(df)} 条")
    
    # 3. 只保留nM单位的数据
    df = df[df['standard_units'] == 'nM']
    print(f"筛选单位后: {len(df)} 条")
    
    # 4. 移除异常值
    df['standard_value'] = pd.to_numeric(df['standard_value'], errors='coerce')
    df = df[(df['standard_value'] > 0) & (df['standard_value'] < 1e8)]
    print(f"移除异常值后: {len(df)} 条")
    
    # 5. 计算pActivity
    df['pActivity'] = -np.log10(df['standard_value'] * 1e-9)
    df = df.dropna(subset=['pActivity'])
    
    # 6. 对于同一化合物的多个测量值，取平均 (按靶点分组)
    # 如果没有target_type列，默认为VEGFR2
    if 'target_type' not in df.columns:
        df['target_type'] = 'VEGFR2'
        
    df_agg = df.groupby(['molecule_chembl_id', 'target_type']).agg({
        'canonical_smiles': 'first',
        'pActivity': 'mean',
    }).reset_index()
    
    print(f"聚合后数据: {len(df_agg)} 条 (化合物-靶点对)")
    
    # 7. 标注活性类别 (严格二分类)
    def get_label(p_activity):
        # IC50 < 1 uM => pActivity > 6.0 => Active (1)
        # IC50 > 10 uM => pActivity < 5.0 => Inactive (0)
        if p_activity > 6.0:
            return 1
        elif p_activity < 5.0:
            return 0
        else:
            return None # 丢弃中间值

    df_agg['label'] = df_agg['pActivity'].apply(get_label)
    df_agg = df_agg.dropna(subset=['label'])
    print(f"应用严格阈值后: {len(df_agg)} 条")
    
    # 转换长格式为宽格式 (Multi-task)
    df_pivot = df_agg.pivot(index='molecule_chembl_id', columns='target_type', values='label')
    df_smiles = df_agg.groupby('molecule_chembl_id')['canonical_smiles'].first()
    df_final = df_pivot.join(df_smiles).reset_index()
    
    # 重命名列
    df_final = df_final.rename(columns={'VEGFR2': 'label_vegfr2', 'A549': 'label_a549'})
    
    # 8. 负样本补充 (简易版: 从ChEMBL随机抽取非激酶靶点分子)
    print("\n=== 负样本补充 ===")
    df_decoys = fetch_negative_samples(limit=len(df_final) // 2) # 补充约50%的负样本
    if not df_decoys.empty:
        df_final = pd.concat([df_final, df_decoys], ignore_index=True)
        
    print("\n最终数据分布:")
    if 'label_vegfr2' in df_final.columns:
        print("VEGFR2 Labels:", df_final['label_vegfr2'].value_counts(dropna=False).to_dict())
    if 'label_a549' in df_final.columns:
        print("A549 Labels:", df_final['label_a549'].value_counts(dropna=False).to_dict())

  

    
    # 保存处理后的数据
    if output_path:
        output_file = Path(output_path)
    else:
        output_file = Path("./data/raw/vegfr2_processed.csv")
    df_final.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[OK] 处理后数据已保存: {output_file}")
    
    # 生成统计报告
    report = {
        'total_compounds': len(df_final),
        'columns': df_final.columns.tolist(),
    }
    
    report_file = Path("./data/raw/data_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 统计报告已保存: {report_file}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] 数据处理完成!")
    print("=" * 60)
    
    return df_final

def fetch_negative_samples(limit=100):
    """从ChEMBL获取负样本 (使用无关靶点，例如 Dopamine D2 receptor CHEMBL217)"""
    print(f"正在获取 {limit} 个负样本...")
    target_id = "CHEMBL217" 
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    
    params = {
        'target_chembl_id': target_id,
        'limit': limit,
        'standard_type': 'IC50',
        'standard_units': 'nM'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            activities = data.get('activities', [])
            
            decoys = []
            for act in activities:
                if act.get('canonical_smiles'):
                    decoys.append({
                        'molecule_chembl_id': act.get('molecule_chembl_id'),
                        'canonical_smiles': act.get('canonical_smiles'),
                        'label_vegfr2': 0, # 假定为不活跃
                        'label_a549': 0    # 假定为不活跃
                    })
            
            return pd.DataFrame(decoys)
    except Exception as e:
        print(f"获取负样本失败: {e}")
    
    return pd.DataFrame()

if __name__ == "__main__":
    df = process_data()
    if df is not None:
        print("\n数据预览:")
        print(df.head(10))
        print(f"\n列名: {df.columns.tolist()}")


