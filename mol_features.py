"""
分子特征提取模块
包括分子指纹生成和分子图构建
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 分子指纹生成
# ============================================================================

class MolecularFingerprints:
    """分子指纹生成器"""
    
    @staticmethod
    def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """
        生成Morgan指纹 (ECFP)
        
        参数:
            mol: RDKit分子对象
            radius: 半径
            n_bits: 位数
        
        返回:
            numpy数组，形状为(n_bits,)
        """
        if mol is None:
            return np.zeros(n_bits)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    
    @staticmethod
    def maccs_fingerprint(mol: Chem.Mol) -> np.ndarray:
        """
        生成MACCS指纹
        
        参数:
            mol: RDKit分子对象
        
        返回:
            numpy数组，形状为(167,)
        """
        if mol is None:
            return np.zeros(167)
        
        fp = AllChem.GetMACCSKeysFingerprint(mol)
        return np.array(fp)
    
    @staticmethod
    def rdkit_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        """
        生成RDKit指纹
        
        参数:
            mol: RDKit分子对象
            n_bits: 位数
        
        返回:
            numpy数组，形状为(n_bits,)
        """
        if mol is None:
            return np.zeros(n_bits)
        
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return np.array(fp)
    
    @staticmethod
    def get_all_fingerprints(mol: Chem.Mol) -> dict:
        """
        生成所有类型的指纹
        
        参数:
            mol: RDKit分子对象
        
        返回:
            字典，包含不同类型的指纹
        """
        return {
            'morgan': MolecularFingerprints.morgan_fingerprint(mol),
            'maccs': MolecularFingerprints.maccs_fingerprint(mol),
            'rdkit': MolecularFingerprints.rdkit_fingerprint(mol)
        }


# ============================================================================
# 原子特征提取
# ============================================================================

class AtomFeaturizer:
    """原子特征提取器"""
    
    # 原子类型
    ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Unknown']
    
    # 杂化类型
    HYBRIDIZATION_TYPES = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ]
    
    @staticmethod
    def one_hot_encoding(value, allowed_set):
        """One-hot编码"""
        encoding = [0] * len(allowed_set)
        if value in allowed_set:
            encoding[allowed_set.index(value)] = 1
        else:
            encoding[-1] = 1  # Unknown
        return encoding
    
    @staticmethod
    def get_atom_features(atom: Chem.Atom) -> List[float]:
        """
        提取原子特征
        
        特征包括:
        - 原子类型 (one-hot, 13维)
        - 原子度数 (one-hot, 6维)
        - 形式电荷 (1维)
        - 杂化类型 (one-hot, 6维)
        - 是否为芳香原子 (1维)
        - 氢原子数 (one-hot, 5维)
        - 是否在环中 (1维)
        - 手性 (one-hot, 4维)
        
        总计: 13 + 6 + 1 + 6 + 1 + 5 + 1 + 4 = 37维
        
        参数:
            atom: RDKit原子对象
        
        返回:
            特征列表
        """
        features = []
        
        # 1. 原子类型 (13维)
        atom_type = atom.GetSymbol()
        features += AtomFeaturizer.one_hot_encoding(atom_type, AtomFeaturizer.ATOM_TYPES)
        
        # 2. 原子度数 (6维: 0, 1, 2, 3, 4, >4)
        degree = atom.GetDegree()
        features += AtomFeaturizer.one_hot_encoding(degree, [0, 1, 2, 3, 4, 5])
        
        # 3. 形式电荷 (1维)
        features.append(atom.GetFormalCharge())
        
        # 4. 杂化类型 (6维)
        hybridization = atom.GetHybridization()
        features += AtomFeaturizer.one_hot_encoding(hybridization, AtomFeaturizer.HYBRIDIZATION_TYPES)
        
        # 5. 是否为芳香原子 (1维)
        features.append(int(atom.GetIsAromatic()))
        
        # 6. 氢原子数 (5维: 0, 1, 2, 3, >3)
        num_hs = atom.GetTotalNumHs()
        features += AtomFeaturizer.one_hot_encoding(num_hs, [0, 1, 2, 3, 4])
        
        # 7. 是否在环中 (1维)
        features.append(int(atom.IsInRing()))
        
        # 8. 手性 (4维)
        try:
            chiral_tag = atom.GetChiralTag()
            features += AtomFeaturizer.one_hot_encoding(
                chiral_tag,
                [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                 Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                 Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                 Chem.rdchem.ChiralType.CHI_OTHER]
            )
        except:
            features += [1, 0, 0, 0]
        
        return features


# ============================================================================
# 键特征提取
# ============================================================================

class BondFeaturizer:
    """键特征提取器"""
    
    @staticmethod
    def get_bond_features(bond: Chem.Bond) -> List[float]:
        """
        提取键特征
        
        特征包括:
        - 键类型 (one-hot, 4维: SINGLE, DOUBLE, TRIPLE, AROMATIC)
        - 是否为共轭键 (1维)
        - 是否在环中 (1维)
        - 立体化学 (one-hot, 6维)
        
        总计: 4 + 1 + 1 + 6 = 12维
        
        参数:
            bond: RDKit键对象
        
        返回:
            特征列表
        """
        features = []
        
        # 1. 键类型 (4维)
        bond_type = bond.GetBondType()
        features += [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]
        
        # 2. 是否为共轭键 (1维)
        features.append(int(bond.GetIsConjugated()))
        
        # 3. 是否在环中 (1维)
        features.append(int(bond.IsInRing()))
        
        # 4. 立体化学 (6维)
        stereo = bond.GetStereo()
        features += [
            int(stereo == Chem.rdchem.BondStereo.STEREONONE),
            int(stereo == Chem.rdchem.BondStereo.STEREOANY),
            int(stereo == Chem.rdchem.BondStereo.STEREOZ),
            int(stereo == Chem.rdchem.BondStereo.STEREOE),
            int(stereo == Chem.rdchem.BondStereo.STEREOCIS),
            int(stereo == Chem.rdchem.BondStereo.STEREOTRANS)
        ]
        
        return features


# ============================================================================
# 分子图构建
# ============================================================================

class MolecularGraphBuilder:
    """分子图构建器"""
    
    @staticmethod
    def mol_to_graph(mol: Chem.Mol, y: Optional[float] = None) -> Data:
        """
        将RDKit分子对象转换为PyTorch Geometric图对象
        
        参数:
            mol: RDKit分子对象
            y: 目标值（可选）
        
        返回:
            PyTorch Geometric Data对象
        """
        if mol is None:
            return None
        
        # 提取原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = AtomFeaturizer.get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 提取边和边特征
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 无向图，添加双向边
            edge_indices.append([i, j])
            edge_indices.append([j, i])
            
            # 边特征
            bond_feat = BondFeaturizer.get_bond_features(bond)
            edge_features.append(bond_feat)
            edge_features.append(bond_feat)  # 双向边使用相同特征
        
        # 转换为张量
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            # 没有边的情况（单个原子）
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 12), dtype=torch.float)
        
        # 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        # 添加目标值
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.float)
        
        return data
    
    @staticmethod
    def smiles_to_graph(smiles: str, y: Optional[float] = None) -> Data:
        """
        从SMILES字符串构建分子图
        
        参数:
            smiles: SMILES字符串
            y: 目标值（可选）
        
        返回:
            PyTorch Geometric Data对象
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return MolecularGraphBuilder.mol_to_graph(mol, y)


# ============================================================================
# 分子描述符计算
# ============================================================================

class MolecularDescriptors:
    """分子描述符计算器"""
    
    @staticmethod
    def calculate_descriptors(mol: Chem.Mol) -> dict:
        """
        计算常用分子描述符
        
        参数:
            mol: RDKit分子对象
        
        返回:
            描述符字典
        """
        if mol is None:
            return {}
        
        descriptors = {
            'MW': Descriptors.MolWt(mol),  # 分子量
            'LogP': Descriptors.MolLogP(mol),  # 脂水分配系数
            'HBA': rdMolDescriptors.CalcNumHBA(mol),  # 氢键受体数
            'HBD': rdMolDescriptors.CalcNumHBD(mol),  # 氢键供体数
            'TPSA': Descriptors.TPSA(mol),  # 拓扑极性表面积
            'RotBonds': Descriptors.NumRotatableBonds(mol),  # 可旋转键数
            'AromaticRings': Descriptors.NumAromaticRings(mol),  # 芳香环数
            'HeteroAtoms': Descriptors.NumHeteroatoms(mol),  # 杂原子数
            'FractionCSP3': Descriptors.FractionCsp3(mol),  # SP3碳比例
            'MolMR': Descriptors.MolMR(mol),  # 摩尔折射率
        }
        
        return descriptors
    
    @staticmethod
    def check_lipinski_rule(mol: Chem.Mol) -> Tuple[bool, int]:
        """
        检查Lipinski五规则
        
        规则:
        - 分子量 ≤ 500
        - LogP ≤ 5
        - 氢键供体 ≤ 5
        - 氢键受体 ≤ 10
        
        参数:
            mol: RDKit分子对象
        
        返回:
            (是否通过, 违规数)
        """
        if mol is None:
            return False, 4
        
        violations = 0
        
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        
        if mw > 500:
            violations += 1
        if logp > 5:
            violations += 1
        if hbd > 5:
            violations += 1
        if hba > 10:
            violations += 1
        
        return violations <= 1, violations


# ============================================================================
# 批处理工具
# ============================================================================

def smiles_to_features(smiles: str, 
                       include_graph: bool = True,
                       include_fingerprint: bool = True,
                       fingerprint_type: str = 'morgan',
                       y: Optional[float] = None) -> dict:
    """
    从SMILES提取所有特征
    
    参数:
        smiles: SMILES字符串
        include_graph: 是否包含图表示
        include_fingerprint: 是否包含指纹
        fingerprint_type: 指纹类型
        y: 目标值
    
    返回:
        特征字典
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    features = {'smiles': smiles}
    
    # 添加图表示
    if include_graph:
        graph = MolecularGraphBuilder.mol_to_graph(mol, y)
        features['graph'] = graph
    
    # 添加指纹
    if include_fingerprint:
        if fingerprint_type == 'morgan':
            fp = MolecularFingerprints.morgan_fingerprint(mol)
        elif fingerprint_type == 'maccs':
            fp = MolecularFingerprints.maccs_fingerprint(mol)
        elif fingerprint_type == 'rdkit':
            fp = MolecularFingerprints.rdkit_fingerprint(mol)
        else:
            fp = MolecularFingerprints.morgan_fingerprint(mol)
        
        features['fingerprint'] = torch.tensor(fp, dtype=torch.float)
    
    # 添加分子描述符
    features['descriptors'] = MolecularDescriptors.calculate_descriptors(mol)
    
    # 添加类药性检查
    druglike, violations = MolecularDescriptors.check_lipinski_rule(mol)
    features['druglike'] = druglike
    features['lipinski_violations'] = violations
    
    return features


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试SMILES
    test_smiles = "CN1CCN(CC1)c1ccc(Nc2nccc(Nc3ccc(C)cc3)n2)cc1"  # 示例化合物
    
    print("测试分子特征提取...")
    print(f"SMILES: {test_smiles}")
    
    # 提取特征
    features = smiles_to_features(test_smiles, y=7.5)
    
    if features:
        print(f"\n图节点数: {features['graph'].x.shape[0]}")
        print(f"图边数: {features['graph'].edge_index.shape[1]}")
        print(f"节点特征维度: {features['graph'].x.shape[1]}")
        print(f"边特征维度: {features['graph'].edge_attr.shape[1]}")
        print(f"\n指纹维度: {features['fingerprint'].shape[0]}")
        print(f"\n分子描述符:")
        for key, value in features['descriptors'].items():
            print(f"  {key}: {value:.2f}")
        print(f"\n类药性: {features['druglike']}")
        print(f"Lipinski违规数: {features['lipinski_violations']}")
    else:
        print("特征提取失败")

