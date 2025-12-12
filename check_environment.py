"""
环境检查脚本
检查所有必需的依赖包是否正确安装
"""

import sys

def check_package(package_name, import_name=None, required=True):
    """检查单个包的安装情况"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        status = "[OK]"
        print(f"{status} {package_name:20s} 版本: {version}")
        return True
    except ImportError:
        if required:
            status = "[X]"
            print(f"{status} {package_name:20s} 未安装 (必需)")
        else:
            status = "[-]"
            print(f"{status} {package_name:20s} 未安装 (可选)")
        return False

def main():
    print("=" * 70)
    print("VEGFR2抑制剂发现项目 - 环境检查")
    print("=" * 70)
    
    print("\n【核心依赖】")
    core_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',
    }
    
    core_ok = True
    for pkg, import_name in core_packages.items():
        if not check_package(pkg, import_name, required=True):
            core_ok = False
    
    print("\n【深度学习框架】")
    dl_packages = {
        'torch': 'torch',
        'torch-geometric': 'torch_geometric',
    }
    
    dl_ok = True
    for pkg, import_name in dl_packages.items():
        if not check_package(pkg, import_name, required=True):
            dl_ok = False
    
    # 检查CUDA
    if 'torch' in sys.modules or dl_ok:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                print(f"[OK] {'CUDA':20s} 可用 (GPU: {torch.cuda.get_device_name(0)})")
            else:
                print(f"[-] {'CUDA':20s} 不可用 (将使用CPU)")
        except:
            pass
    
    print("\n【化学信息学】")
    rdkit_ok = check_package('rdkit', 'rdkit', required=True)
    
    print("\n【可视化】")
    viz_packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    for pkg, import_name in viz_packages.items():
        check_package(pkg, import_name, required=False)
    
    print("\n【其他工具】")
    other_packages = {
        'tqdm': 'tqdm',
        'requests': 'requests',
    }
    
    for pkg, import_name in other_packages.items():
        check_package(pkg, import_name, required=False)
    
    print("\n" + "=" * 70)
    print("检查结果总结")
    print("=" * 70)
    
    all_ok = core_ok and dl_ok and rdkit_ok
    
    if all_ok:
        print("[SUCCESS] 所有必需的包都已正确安装！")
        print("\n您可以开始使用项目了：")
        print("  python train.py")
    else:
        print("[ERROR] 存在缺失的必需包")
        print("\n需要安装的包：")
        
        if not core_ok:
            print("\n基础包:")
            print("  pip install numpy pandas scipy scikit-learn")
        
        if not dl_ok:
            print("\n深度学习包:")
            print("  # PyTorch (访问 https://pytorch.org 选择合适的版本)")
            print("  pip install torch torchvision torchaudio")
            print("  pip install torch-geometric")
        
        if not rdkit_ok:
            print("\n化学信息学包:")
            print("  conda install -c conda-forge rdkit")
            print("  # 或者创建新环境:")
            print("  conda create -n vegfr2_env python=3.9 rdkit -c conda-forge -y")
    
    print("\n" + "=" * 70)
    
    # 详细的RDKit检查
    print("\n【RDKit详细检查】")
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        
        # 测试基本功能
        test_smiles = "CCO"
        mol = Chem.MolFromSmiles(test_smiles)
        
        if mol is not None:
            print(f"[OK] RDKit分子解析正常")
            
            # 测试指纹生成
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            print(f"[OK] RDKit指纹生成正常")
            
            # 测试描述符计算
            mw = Descriptors.MolWt(mol)
            print(f"[OK] RDKit描述符计算正常")
            
            print(f"\n[SUCCESS] RDKit功能完全正常！")
        else:
            print(f"[X] RDKit分子解析失败")
            
    except ImportError as e:
        print(f"[X] RDKit未安装或导入失败")
        print(f"  错误信息: {e}")
        print("\n安装方法:")
        print("  方案1 (推荐): conda install -c conda-forge rdkit")
        print("  方案2: 创建新环境")
        print("    conda create -n vegfr2_env python=3.9 -y")
        print("    conda activate vegfr2_env")
        print("    conda install -c conda-forge rdkit -y")
    except Exception as e:
        print(f"[X] RDKit测试失败: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()

