# VEGFR2抑制剂数据收集指南

本目录包含用于收集VEGFR2抑制剂研究所需数据的完整工具包。

## 📋 目录结构

```
data_collection/
├── collect_vegfr2_data.py          # 主数据收集脚本
├── download_protein_structure.py   # 蛋白质结构下载脚本
├── manual_download_guide.md        # 手动下载指南
├── requirements.txt                # Python依赖包
└── README.md                       # 本文件
```

---

## 🚀 快速开始

### 方法一：自动收集（推荐）

#### 步骤1：安装依赖
```bash
pip install -r requirements.txt
```

#### 步骤2：运行数据收集脚本
```bash
python collect_vegfr2_data.py
```

这将自动：
- 从ChEMBL下载VEGFR2抑制剂的活性数据
- 获取化合物的分子性质
- 清洗和过滤数据
- 保存处理后的数据到 `./data/raw/`

#### 步骤3：下载蛋白质结构
```bash
python download_protein_structure.py
```

这将下载10个常用的VEGFR2蛋白质结构到 `./data/protein_structures/`

---

### 方法二：手动收集

如果自动脚本遇到问题（如网络限制），请参考 [manual_download_guide.md](manual_download_guide.md) 进行手动下载。

---

## 📊 数据来源说明

### 1. ChEMBL数据库
- **网址**: https://www.ebi.ac.uk/chembl/
- **内容**: VEGFR2抑制剂的生物活性数据
- **Target ID**: CHEMBL279 (KDR/VEGFR2)
- **数据类型**: IC50, Ki, EC50等活性值

### 2. RCSB PDB
- **网址**: https://www.rcsb.org/
- **内容**: VEGFR2蛋白质3D结构
- **推荐结构**:
  - `3VHE`: 与Sorafenib复合物（推荐用于对接）
  - `4AGD`: 与Axitinib复合物
  - `4ASE`: 与Pazopanib复合物

### 3. PubChem（可选）
- **网址**: https://pubchem.ncbi.nlm.nih.gov/
- **内容**: 化合物额外信息和同义词

---

## 📁 输出文件说明

运行脚本后，将在 `./data/` 目录下生成以下文件：

```
data/
├── raw/
│   ├── chembl_vegfr2_raw.csv       # 原始ChEMBL数据
│   ├── molecule_properties.csv     # 分子性质数据
│   ├── vegfr2_processed.csv        # 清洗后的数据（主要使用）
│   └── data_report.json            # 数据统计报告
│
└── protein_structures/
    ├── 3VHE.pdb                    # VEGFR2结构文件
    ├── 4AGD.pdb
    ├── ...
    └── recommended_structures.txt   # 推荐使用说明
```

### 主要数据文件格式

**vegfr2_processed.csv** 包含以下列：
- `molecule_chembl_id`: ChEMBL化合物ID
- `canonical_smiles`: SMILES分子结构字符串
- `pActivity`: 负对数活性值 (-log10[M])
- `activity_class`: 活性分类 (high/medium/low)
- `molecular_weight`: 分子量
- `alogp`: 脂水分配系数
- `hba`: 氢键受体数
- `hbd`: 氢键供体数
- `psa`: 极性表面积
- `num_ro5_violations`: Lipinski五规则违反数

---

## 🛠️ 脚本使用详解

### 自定义数据收集

```python
from collect_vegfr2_data import VEGFR2DataCollector

# 创建收集器
collector = VEGFR2DataCollector(output_dir="./my_data")

# 收集指定数量的数据
df = collector.collect_from_chembl(limit=1000)

# 数据清洗
df_clean = collector.filter_and_clean_data(df)

# 保存
collector.save_processed_data(df_clean)
```

### 下载特定蛋白结构

```python
from download_protein_structure import ProteinStructureDownloader

downloader = ProteinStructureDownloader(output_dir="./structures")

# 下载单个结构
downloader.download_pdb('3VHE')

# 获取结构信息
info = downloader.get_structure_info('3VHE')
print(info)
```

---

## 📈 预期数据量

根据ChEMBL数据库（截至2024年）：
- **VEGFR2相关活性数据**: ~15,000-20,000条记录
- **唯一化合物数**: ~3,000-5,000个
- **清洗后高质量数据**: ~2,000-3,000个化合物

活性分布（参考）：
- **高活性** (pIC50 ≥ 7): ~30-40%
- **中等活性** (6 ≤ pIC50 < 7): ~30-40%
- **低活性** (pIC50 < 6): ~20-40%

---

## ⚠️ 注意事项

### 1. 网络要求
- 需要稳定的互联网连接
- 某些API可能有访问限制
- 建议使用机构网络或VPN

### 2. 时间估计
- **ChEMBL数据收集**: 5-15分钟（取决于数据量）
- **蛋白质结构下载**: 2-5分钟
- **总计**: 约10-20分钟

### 3. 数据使用许可
- ChEMBL数据：遵循CC-BY-SA 3.0许可
- PDB数据：可免费用于学术研究
- 商业用途请查阅相应数据库的使用条款

### 4. API限制
- ChEMBL API可能有请求频率限制
- 脚本已内置延时，避免请求过快
- 如遇429错误，请稍后重试

---

## 🔧 故障排除

### 问题1: 网络连接超时
**解决方案**:
- 检查网络连接
- 尝试使用代理或VPN
- 增加timeout参数

### 问题2: 没有收集到数据
**可能原因**:
- ChEMBL API临时不可用
- Target ID可能已更新

**解决方案**:
- 访问 https://www.ebi.ac.uk/chembl/ 确认API状态
- 手动查找VEGFR2的最新Target ID

### 问题3: 依赖包安装失败
**解决方案**:
```bash
# 使用conda安装RDKit
conda install -c conda-forge rdkit

# 或使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📞 获取帮助

如果遇到问题：
1. 查看 [manual_download_guide.md](manual_download_guide.md) 尝试手动下载
2. 检查网络连接和API状态
3. 查阅ChEMBL官方文档: https://chembl.gitbook.io/chembl-interface-documentation/

---

## 📚 参考资源

### 数据库文档
- [ChEMBL API文档](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
- [RCSB PDB文档](https://www.rcsb.org/docs/)
- [PubChem API](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest)

### 相关工具
- [RDKit文档](https://www.rdkit.org/docs/)
- [BioPython](https://biopython.org/)

---

## 📝 更新日志

- **v1.0** (2024-10-29): 初始版本
  - ChEMBL数据自动收集
  - 蛋白质结构下载
  - 数据清洗和过滤

