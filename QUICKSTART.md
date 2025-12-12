# 🚀 快速启动指南

5分钟快速上手VEGFR2抑制剂发现项目

---

## ✅ 前置条件

- Python 3.9+
- CUDA（可选，用于GPU加速）

---

## 📥 步骤1：安装环境（5分钟）

```bash
# 创建conda环境
conda create -n vegfr2 python=3.9 -y
conda activate vegfr2

# 安装核心依赖
conda install -c conda-forge rdkit pytorch torchvision pytorch-geometric -y

# 安装其他依赖
pip install pandas numpy scikit-learn matplotlib seaborn tqdm requests scipy
```

---

## 📊 步骤2：收集数据（10-15分钟）

```bash
cd data_collection
python collect_vegfr2_data.py
cd ..
```

**预期输出**：
- 收集约3,000个VEGFR2抑制剂化合物
- 数据保存在 `data_collection/data/raw/vegfr2_processed.csv`

---

## 🤖 步骤3：训练模型（1-2小时，取决于硬件）

```bash
python train.py
```

**预期输出**：
- 训练200个epoch（可能提前停止）
- 最佳模型保存在 `models/checkpoints/best_model.pt`
- 训练日志保存在 `models/logs/`

**训练进度示例**：
```
Epoch 1/200 - Train Loss: 1.2345, Val Loss: 1.3456
Epoch 2/200 - Train Loss: 1.1234, Val Loss: 1.2345
...
```

---

## 🔍 步骤4：虚拟筛选（5-10分钟）

```bash
python virtual_screening.py
```

**预期输出**：
- 筛选Top-100高活性化合物
- 结果保存在 `results/predictions/virtual_screening_results.csv`

---

## 🎉 完成！

你现在已经：
✅ 收集了3,000+个VEGFR2抑制剂数据  
✅ 训练了一个指纹增强图注意力网络模型  
✅ 筛选出了高活性候选化合物  

---

## 📋 一键运行（推荐）

如果你想一次性运行所有步骤：

```bash
python run_pipeline.py --all
```

---

## 🛠️ 常见问题

### Q1: 内存不足
**A**: 修改 `config.py` 中的 `batch_size`:
```python
TRAIN_CONFIG['batch_size'] = 16  # 从32降到16
```

### Q2: 训练太慢
**A**: 
- 确保使用GPU：检查 `torch.cuda.is_available()` 返回 `True`
- 或减少epoch数：`TRAIN_CONFIG['num_epochs'] = 50`

### Q3: 数据收集失败
**A**: 使用示例数据测试：
```bash
# 使用sample_data.csv（20个化合物）
cp data_collection/sample_data.csv data_collection/data/raw/vegfr2_processed.csv
```

---

## 📊 查看结果

### 训练结果
```bash
# 查看训练历史
cat models/logs/training_history.json

# 查看最终结果
cat results/final_results.json
```

### 筛选结果
```bash
# 查看Top-10化合物
head -11 results/predictions/virtual_screening_results.csv
```

---

## 🔬 下一步

### 1. 可视化结果
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('results/predictions/virtual_screening_results.csv')

# 绘制活性分布
plt.figure(figsize=(10, 6))
plt.hist(df['predicted_pIC50'], bins=50)
plt.xlabel('Predicted pIC50')
plt.ylabel('Frequency')
plt.title('Virtual Screening Results')
plt.savefig('activity_distribution.png')
```

### 2. 预测新化合物
```python
from preprocessing import smiles_to_features
from models import FingerprintEnhancedGAT
import torch

# 加载模型
model = FingerprintEnhancedGAT()
checkpoint = torch.load('models/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预测
smiles = "CN1CCN(CC1)c1ccc(Nc2nccc(n2)Nc2ccc(C)cc2)cc1"
features = smiles_to_features(smiles, fingerprint_type='morgan')

batch = {
    'graph': features['graph'],
    'fingerprint': features['fingerprint'].unsqueeze(0)
}

with torch.no_grad():
    pred = model(batch)
    print(f"Predicted pIC50: {pred.item():.2f}")
```

### 3. 分子对接（需要额外安装）
```bash
# 安装AutoDock Vina
conda install -c conda-forge autodock-vina

# 生成MD模拟脚本
python run_pipeline.py --md-guide
```

---

## 📚 更多信息

- 完整文档：[README.md](README.md)
- 数据收集：[data_collection/README.md](data_collection/README.md)
- 配置说明：[config.py](config.py)

---

## 🆘 获取帮助

遇到问题？

1. 查看 [README.md](README.md) 的故障排除部分
2. 查看日志文件 `models/logs/training.log`
3. 提交Issue

---

**祝实验顺利！** 🎯

