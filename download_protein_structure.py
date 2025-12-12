#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载VEGFR2蛋白质结构
从RCSB PDB数据库下载VEGFR2的3D结构
"""

import requests
from pathlib import Path
from typing import List


class ProteinStructureDownloader:
    """蛋白质结构下载器"""
    
    def __init__(self, output_dir: str = "./data/protein_structures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdb_base_url = "https://files.rcsb.org/download"
    
    def download_pdb(self, pdb_id: str) -> bool:
        """
        下载PDB结构文件
        
        参数:
            pdb_id: PDB ID (例如: '3VHE')
        
        返回:
            是否成功下载
        """
        try:
            url = f"{self.pdb_base_url}/{pdb_id}.pdb"
            print(f"正在下载 {pdb_id}...")
            
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                output_file = self.output_dir / f"{pdb_id}.pdb"
                with open(output_file, 'w') as f:
                    f.write(response.text)
                
                print(f"✓ 已保存: {output_file}")
                return True
            else:
                print(f"✗ 下载失败: {pdb_id} (状态码: {response.status_code})")
                return False
                
        except Exception as e:
            print(f"✗ 错误: {e}")
            return False
    
    def get_structure_info(self, pdb_id: str) -> dict:
        """
        获取PDB结构的详细信息
        
        参数:
            pdb_id: PDB ID
        
        返回:
            结构信息字典
        """
        try:
            url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            print(f"获取信息失败: {e}")
            return {}
    
    def download_vegfr2_structures(self):
        """下载常用的VEGFR2结构"""
        
        # VEGFR2相关的PDB结构（与不同抑制剂的复合物）
        vegfr2_structures = {
            '3VHE': 'VEGFR2激酶结构域 + Sorafenib',
            '4AGD': 'VEGFR2激酶结构域 + Axitinib',
            '3WZD': 'VEGFR2激酶结构域 + Tivozanib',
            '4ASE': 'VEGFR2激酶结构域 + Pazopanib',
            '3C7Q': 'VEGFR2激酶结构域 + Sunitinib',
            '2OH4': 'VEGFR2激酶结构域',
            '1YWN': 'VEGFR2激酶结构域',
            '3EFL': 'VEGFR2激酶结构域 + 抑制剂',
            '3CJG': 'VEGFR2激酶结构域 + 抑制剂',
            '4AGC': 'VEGFR2激酶结构域 + 抑制剂',
        }
        
        print("=" * 60)
        print("下载VEGFR2蛋白质结构")
        print("=" * 60)
        
        successful = []
        failed = []
        
        for pdb_id, description in vegfr2_structures.items():
            print(f"\n[{pdb_id}] {description}")
            
            # 获取结构信息
            info = self.get_structure_info(pdb_id)
            if info:
                resolution = info.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]
                if resolution:
                    print(f"  分辨率: {resolution} Å")
            
            # 下载结构
            if self.download_pdb(pdb_id):
                successful.append(pdb_id)
            else:
                failed.append(pdb_id)
        
        print("\n" + "=" * 60)
        print(f"✓ 成功下载: {len(successful)} 个结构")
        print(f"✗ 下载失败: {len(failed)} 个结构")
        
        if successful:
            print(f"\n成功下载的结构: {', '.join(successful)}")
        if failed:
            print(f"失败的结构: {', '.join(failed)}")
        
        print("=" * 60)
        
        # 保存推荐配置
        recommendations = self.output_dir / "recommended_structures.txt"
        with open(recommendations, 'w', encoding='utf-8') as f:
            f.write("VEGFR2蛋白质结构推荐\n")
            f.write("=" * 60 + "\n\n")
            f.write("用于分子对接的推荐结构:\n")
            f.write("  • 3VHE - 高分辨率结构，与Sorafenib复合\n")
            f.write("  • 4AGD - 与Axitinib复合，口袋清晰\n")
            f.write("  • 4ASE - 与Pazopanib复合\n\n")
            f.write("结构选择建议:\n")
            f.write("  1. 优先选择高分辨率结构 (<2.5 Å)\n")
            f.write("  2. 选择与已知抑制剂的复合物结构\n")
            f.write("  3. 检查配体结合口袋的完整性\n")
        
        print(f"✓ 推荐信息已保存: {recommendations}")


def main():
    downloader = ProteinStructureDownloader()
    downloader.download_vegfr2_structures()


if __name__ == "__main__":
    main()

