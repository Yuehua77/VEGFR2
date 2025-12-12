"""
分子对接脚本
使用AutoDock Vina进行分子对接
"""

import subprocess
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import *
from utils.logger import setup_logger


class MolecularDocker:
    """分子对接器"""
    
    def __init__(self, logger):
        """
        初始化分子对接器
        
        参数:
            logger: 日志记录器
        """
        self.logger = logger
        self.config = DOCKING_CONFIG
        self.logger.info("分子对接器初始化完成")
    
    def prepare_ligand(self, smiles: str, output_path: Path) -> bool:
        """
        准备配体文件
        
        参数:
            smiles: SMILES字符串
            output_path: 输出文件路径（.pdbqt）
        
        返回:
            是否成功
        """
        try:
            # SMILES转分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # 添加氢原子
            mol = Chem.AddHs(mol)
            
            # 生成3D坐标
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # 保存为PDB文件
            pdb_path = output_path.with_suffix('.pdb')
            Chem.MolToPDBFile(mol, str(pdb_path))
            
            # 转换为PDBQT格式（需要安装MGLTools）
            # 这里提供命令示例，实际执行需要安装相应工具
            cmd = f"prepare_ligand4.py -l {pdb_path} -o {output_path}"
            
            self.logger.info(f"配体准备命令: {cmd}")
            self.logger.info("注意: 需要安装MGLTools才能执行转换")
            
            return True
            
        except Exception as e:
            self.logger.error(f"配体准备失败: {e}")
            return False
    
    def run_docking(self,
                    ligand_pdbqt: Path,
                    receptor_pdbqt: Path,
                    output_pdbqt: Path,
                    center: List[float] = None,
                    size: List[float] = None) -> Tuple[bool, float]:
        """
        运行AutoDock Vina对接
        
        参数:
            ligand_pdbqt: 配体PDBQT文件
            receptor_pdbqt: 受体PDBQT文件
            output_pdbqt: 输出PDBQT文件
            center: 对接盒子中心坐标 [x, y, z]
            size: 对接盒子大小 [x, y, z]
        
        返回:
            (是否成功, 结合能)
        """
        if center is None:
            center = self.config['box_center']
        if size is None:
            size = self.config['box_size']
        
        # 构建Vina命令
        cmd = [
            'vina',
            '--receptor', str(receptor_pdbqt),
            '--ligand', str(ligand_pdbqt),
            '--out', str(output_pdbqt),
            '--center_x', str(center[0]),
            '--center_y', str(center[1]),
            '--center_z', str(center[2]),
            '--size_x', str(size[0]),
            '--size_y', str(size[1]),
            '--size_z', str(size[2]),
            '--exhaustiveness', str(self.config['exhaustiveness']),
            '--num_modes', str(self.config['num_modes']),
            '--energy_range', str(self.config['energy_range'])
        ]
        
        self.logger.info(f"对接命令: {' '.join(cmd)}")
        
        try:
            # 执行对接（需要安装AutoDock Vina）
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 从输出中提取结合能
                binding_energy = self._parse_vina_output(result.stdout)
                self.logger.info(f"对接成功，结合能: {binding_energy} kcal/mol")
                return True, binding_energy
            else:
                self.logger.error(f"对接失败: {result.stderr}")
                return False, None
                
        except FileNotFoundError:
            self.logger.error("未找到AutoDock Vina，请先安装")
            self.logger.info("安装方法: conda install -c conda-forge autodock-vina")
            return False, None
        except Exception as e:
            self.logger.error(f"对接错误: {e}")
            return False, None
    
    def _parse_vina_output(self, output: str) -> float:
        """解析Vina输出，提取最佳结合能"""
        for line in output.split('\n'):
            if line.strip().startswith('1'):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1])
        return None
    
    def batch_docking(self,
                      input_csv: Path,
                      receptor_pdbqt: Path,
                      output_csv: Path):
        """
        批量对接
        
        参数:
            input_csv: 输入CSV文件（包含SMILES和ID）
            receptor_pdbqt: 受体PDBQT文件
            output_csv: 输出CSV文件
        """
        self.logger.info(f"开始批量对接: {input_csv}")
        
        # 读取数据
        df = pd.read_csv(input_csv)
        self.logger.info(f"读取 {len(df)} 个化合物")
        
        # 创建临时目录
        temp_dir = RESULTS_DIR / 'docking_temp'
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 对接结果
        results = []
        
        for idx, row in df.iterrows():
            compound_id = row.get('molecule_chembl_id', f'compound_{idx}')
            smiles = row['canonical_smiles']
            
            self.logger.info(f"对接 {idx+1}/{len(df)}: {compound_id}")
            
            # 准备配体
            ligand_pdbqt = temp_dir / f"{compound_id}_ligand.pdbqt"
            
            if not self.prepare_ligand(smiles, ligand_pdbqt):
                self.logger.warning(f"配体准备失败: {compound_id}")
                results.append({
                    'compound_id': compound_id,
                    'smiles': smiles,
                    'docking_success': False,
                    'binding_energy': None
                })
                continue
            
            # 执行对接
            output_pdbqt = temp_dir / f"{compound_id}_docked.pdbqt"
            
            success, binding_energy = self.run_docking(
                ligand_pdbqt=ligand_pdbqt,
                receptor_pdbqt=receptor_pdbqt,
                output_pdbqt=output_pdbqt
            )
            
            results.append({
                'compound_id': compound_id,
                'smiles': smiles,
                'docking_success': success,
                'binding_energy': binding_energy
            })
        
        # 保存结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        self.logger.info(f"对接结果已保存: {output_csv}")
        
        # 统计
        success_count = results_df['docking_success'].sum()
        self.logger.info(f"成功对接: {success_count}/{len(df)}")
        
        if success_count > 0:
            avg_energy = results_df[results_df['docking_success']]['binding_energy'].mean()
            self.logger.info(f"平均结合能: {avg_energy:.2f} kcal/mol")


def main():
    """主函数"""
    # 设置日志
    logger = setup_logger('docking', LOGS_DIR / 'molecular_docking.log')
    
    logger.info("=" * 80)
    logger.info("分子对接模块")
    logger.info("=" * 80)
    logger.info("\n注意: 此脚本需要以下软件:")
    logger.info("  1. MGLTools (用于准备配体和受体)")
    logger.info("  2. AutoDock Vina (用于分子对接)")
    logger.info("\n安装方法:")
    logger.info("  conda install -c conda-forge mgltools")
    logger.info("  conda install -c conda-forge autodock-vina")
    logger.info("=" * 80)
    
    # 创建对接器
    docker = MolecularDocker(logger)
    
    # 示例：对接虚拟筛选的Top化合物
    input_file = PREDICTIONS_DIR / 'virtual_screening_results.csv'
    
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        logger.info("请先运行虚拟筛选脚本")
        return
    
    # 受体文件（需要预先准备PDBQT格式）
    receptor_file = PROTEIN_STRUCTURE_DIR / '3VHE_prepared.pdbqt'
    
    if not receptor_file.exists():
        logger.warning(f"受体文件不存在: {receptor_file}")
        logger.info("请先准备VEGFR2受体的PDBQT文件")
        logger.info("准备方法: prepare_receptor4.py -r 3VHE.pdb -o 3VHE_prepared.pdbqt")
        return
    
    # 输出文件
    output_file = RESULTS_DIR / 'docking_results.csv'
    
    # 执行批量对接
    docker.batch_docking(
        input_csv=input_file,
        receptor_pdbqt=receptor_file,
        output_csv=output_file
    )
    
    logger.info("分子对接完成!")


if __name__ == "__main__":
    main()

