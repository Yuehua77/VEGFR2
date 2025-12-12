"""
分子动力学模拟指南
生成GROMACS模拟脚本
"""

from pathlib import Path
from config import *
from utils.logger import setup_logger


class MDSimulationGuide:
    """MD模拟指南生成器"""
    
    def __init__(self, logger):
        self.logger = logger
        self.config = MD_CONFIG
    
    def generate_gromacs_scripts(self, output_dir: Path):
        """生成GROMACS模拟脚本"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"生成GROMACS脚本到: {output_dir}")
        
        # 1. 准备脚本
        self._generate_preparation_script(output_dir / "01_prepare.sh")
        
        # 2. 能量最小化脚本
        self._generate_minimization_script(output_dir / "02_minimization.sh")
        self._generate_minim_mdp(output_dir / "minim.mdp")
        
        # 3. NVT平衡脚本
        self._generate_nvt_script(output_dir / "03_nvt_equilibration.sh")
        self._generate_nvt_mdp(output_dir / "nvt.mdp")
        
        # 4. NPT平衡脚本
        self._generate_npt_script(output_dir / "04_npt_equilibration.sh")
        self._generate_npt_mdp(output_dir / "npt.mdp")
        
        # 5. 生产模拟脚本
        self._generate_production_script(output_dir / "05_production.sh")
        self._generate_md_mdp(output_dir / "md.mdp")
        
        # 6. 分析脚本
        self._generate_analysis_script(output_dir / "06_analysis.sh")
        
        # 7. README
        self._generate_readme(output_dir / "README_MD.md")
        
        self.logger.info("GROMACS脚本生成完成!")
    
    def _generate_preparation_script(self, path: Path):
        """生成准备脚本"""
        script = f"""#!/bin/bash
# GROMACS准备脚本

echo "========================================="
echo "VEGFR2-配体复合物MD模拟 - 准备阶段"
echo "========================================="

# 1. 准备蛋白质拓扑
echo "生成蛋白质拓扑..."
gmx pdb2gmx -f protein.pdb -o protein_processed.gro -water {self.config['water_model'].lower()} -ff {self.config['force_field'].lower()}

# 2. 准备配体拓扑（需要使用ACPYPE或其他工具）
echo "准备配体拓扑..."
echo "注意: 配体拓扑需要使用ACPYPE或LigParGen等工具生成"
echo "示例: acpype -i ligand.mol2 -o gmx"

# 3. 合并拓扑
echo "合并蛋白质和配体..."
# 需要手动编辑topol.top文件，添加配体拓扑

# 4. 定义盒子
echo "定义模拟盒子..."
gmx editconf -f complex.gro -o complex_box.gro -c -d {self.config['box_distance']} -bt {self.config['box_type']}

# 5. 添加溶剂
echo "添加溶剂分子..."
gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_solv.gro -p topol.top

# 6. 添加离子
echo "添加离子..."
gmx grompp -f ions.mdp -c complex_solv.gro -p topol.top -o ions.tpr
echo "SOL" | gmx genion -s ions.tpr -o complex_ions.gro -p topol.top -pname NA -nname CL -neutral -conc {self.config['ion_concentration']}

echo "准备完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_minimization_script(self, path: Path):
        """生成能量最小化脚本"""
        script = """#!/bin/bash
# 能量最小化

echo "========================================="
echo "能量最小化"
echo "========================================="

gmx grompp -f minim.mdp -c complex_ions.gro -p topol.top -o em.tpr
gmx mdrun -v -deffnm em

# 分析能量
echo "Potential" | gmx energy -f em.edr -o potential.xvg

echo "能量最小化完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_minim_mdp(self, path: Path):
        """生成能量最小化参数文件"""
        mdp_content = f"""
; 能量最小化参数
integrator  = {self.config['minimization']['method']}
emtol       = 1000.0
emstep      = 0.01
nsteps      = {self.config['minimization']['steps']}

; 输出控制
nstlog      = 1000
nstenergy   = 1000

; 邻区搜索
cutoff-scheme   = Verlet
nstlist         = 10
ns_type         = grid
rlist           = 1.0

; 静电相互作用
coulombtype     = PME
rcoulomb        = 1.0

; 范德华相互作用
vdwtype         = cutoff
rvdw            = 1.0

; 周期边界条件
pbc             = xyz
"""
        with open(path, 'w') as f:
            f.write(mdp_content)
    
    def _generate_nvt_script(self, path: Path):
        """生成NVT平衡脚本"""
        script = """#!/bin/bash
# NVT平衡

echo "========================================="
echo "NVT平衡（恒定体积）"
echo "========================================="

gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr
gmx mdrun -v -deffnm nvt

# 分析温度
echo "Temperature" | gmx energy -f nvt.edr -o temperature.xvg

echo "NVT平衡完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_nvt_mdp(self, path: Path):
        """生成NVT参数文件"""
        mdp_content = f"""
; NVT平衡参数
define      = -DPOSRES
integrator  = md
nsteps      = {int(self.config['nvt_equilibration']['time'] * 1000 / 0.002)}  ; {self.config['nvt_equilibration']['time']} ps
dt          = 0.002

; 输出控制
nstxout     = 5000
nstvout     = 5000
nstenergy   = 5000
nstlog      = 5000

; 键约束
constraints         = h-bonds
constraint_algorithm = lincs

; 邻区搜索
cutoff-scheme   = Verlet
nstlist         = 10
rlist           = 1.0

; 静电相互作用
coulombtype     = PME
rcoulomb        = 1.0

; 范德华相互作用
vdwtype         = cutoff
rvdw            = 1.0

; 温度耦合
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1    0.1
ref_t           = {self.config['nvt_equilibration']['temperature']} {self.config['nvt_equilibration']['temperature']}

; 周期边界条件
pbc             = xyz

; 速度生成
gen_vel         = yes
gen_temp        = {self.config['nvt_equilibration']['temperature']}
gen_seed        = -1
"""
        with open(path, 'w') as f:
            f.write(mdp_content)
    
    def _generate_npt_script(self, path: Path):
        """生成NPT平衡脚本"""
        script = """#!/bin/bash
# NPT平衡

echo "========================================="
echo "NPT平衡（恒定压力）"
echo "========================================="

gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr
gmx mdrun -v -deffnm npt

# 分析压力和密度
echo "Pressure" | gmx energy -f npt.edr -o pressure.xvg
echo "Density" | gmx energy -f npt.edr -o density.xvg

echo "NPT平衡完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_npt_mdp(self, path: Path):
        """生成NPT参数文件"""
        mdp_content = f"""
; NPT平衡参数
define      = -DPOSRES
integrator  = md
nsteps      = {int(self.config['npt_equilibration']['time'] * 1000 / 0.002)}  ; {self.config['npt_equilibration']['time']} ps
dt          = 0.002

; 输出控制
nstxout     = 5000
nstvout     = 5000
nstenergy   = 5000
nstlog      = 5000

; 键约束
constraints         = h-bonds
constraint_algorithm = lincs

; 邻区搜索
cutoff-scheme   = Verlet
nstlist         = 10
rlist           = 1.0

; 静电相互作用
coulombtype     = PME
rcoulomb        = 1.0

; 范德华相互作用
vdwtype         = cutoff
rvdw            = 1.0

; 温度耦合
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1    0.1
ref_t           = {self.config['npt_equilibration']['temperature']} {self.config['npt_equilibration']['temperature']}

; 压力耦合
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {self.config['npt_equilibration']['pressure']}
compressibility = 4.5e-5

; 周期边界条件
pbc             = xyz

; 速度生成
gen_vel         = no
"""
        with open(path, 'w') as f:
            f.write(mdp_content)
    
    def _generate_production_script(self, path: Path):
        """生成生产模拟脚本"""
        script = """#!/bin/bash
# 生产模拟

echo "========================================="
echo "生产模拟"
echo "========================================="

gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr
gmx mdrun -v -deffnm md

echo "生产模拟完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_md_mdp(self, path: Path):
        """生成生产模拟参数文件"""
        mdp_content = f"""
; 生产模拟参数
integrator  = md
nsteps      = {int(self.config['production']['time'] * 1000000 / 0.002)}  ; {self.config['production']['time']} ns
dt          = 0.002

; 输出控制
nstxout     = 0
nstvout     = 0
nstfout     = 0
nstxout-compressed  = 5000
nstenergy   = 5000
nstlog      = 5000

; 键约束
constraints         = h-bonds
constraint_algorithm = lincs

; 邻区搜索
cutoff-scheme   = Verlet
nstlist         = 20
rlist           = 1.0

; 静电相互作用
coulombtype     = PME
rcoulomb        = 1.0

; 范德华相互作用
vdwtype         = cutoff
rvdw            = 1.0

; 温度耦合
tcoupl          = V-rescale
tc-grps         = Protein Non-Protein
tau_t           = 0.1    0.1
ref_t           = {self.config['production']['temperature']} {self.config['production']['temperature']}

; 压力耦合
pcoupl          = Parrinello-Rahman
pcoupltype      = isotropic
tau_p           = 2.0
ref_p           = {self.config['production']['pressure']}
compressibility = 4.5e-5

; 周期边界条件
pbc             = xyz

; 速度生成
gen_vel         = no
"""
        with open(path, 'w') as f:
            f.write(mdp_content)
    
    def _generate_analysis_script(self, path: Path):
        """生成分析脚本"""
        script = """#!/bin/bash
# 轨迹分析

echo "========================================="
echo "轨迹分析"
echo "========================================="

# 1. RMSD分析
echo "计算RMSD..."
echo "Backbone Backbone" | gmx rms -s md.tpr -f md.xtc -o rmsd.xvg -tu ns

# 2. RMSF分析
echo "计算RMSF..."
echo "Backbone" | gmx rmsf -s md.tpr -f md.xtc -o rmsf.xvg -res

# 3. 回旋半径
echo "计算回旋半径..."
echo "Protein" | gmx gyrate -s md.tpr -f md.xtc -o gyrate.xvg

# 4. 氢键分析
echo "分析氢键..."
gmx hbond -s md.tpr -f md.xtc -num hbond.xvg

# 5. SASA分析
echo "计算SASA..."
echo "Protein" | gmx sasa -s md.tpr -f md.xtc -o sasa.xvg

# 6. 结合能计算（需要g_mmpbsa或gmx_MMPBSA）
echo "计算结合自由能..."
echo "注意: 需要安装gmx_MMPBSA工具"
echo "安装方法: pip install gmx_MMPBSA"

echo "分析完成!"
"""
        with open(path, 'w') as f:
            f.write(script)
        path.chmod(0o755)
    
    def _generate_readme(self, path: Path):
        """生成MD模拟README"""
        readme = f"""# GROMACS分子动力学模拟指南

## 环境要求

- GROMACS 2020或更高版本
- Python 3.7+
- gmx_MMPBSA（用于结合能计算）
- ACPYPE或LigParGen（用于配体参数化）

## 安装

```bash
# 安装GROMACS
conda install -c conda-forge gromacs

# 安装gmx_MMPBSA
pip install gmx_MMPBSA

# 安装ACPYPE
conda install -c conda-forge acpype
```

## 使用流程

### 1. 准备输入文件

- `protein.pdb`: VEGFR2蛋白质结构
- `ligand.mol2`: 配体结构

### 2. 执行模拟步骤

```bash
# 步骤1: 准备系统
./01_prepare.sh

# 步骤2: 能量最小化
./02_minimization.sh

# 步骤3: NVT平衡
./03_nvt_equilibration.sh

# 步骤4: NPT平衡
./04_npt_equilibration.sh

# 步骤5: 生产模拟
./05_production.sh

# 步骤6: 轨迹分析
./06_analysis.sh
```

## 模拟参数

- 力场: {self.config['force_field']}
- 水模型: {self.config['water_model']}
- 温度: {self.config['production']['temperature']} K
- 压力: {self.config['production']['pressure']} bar
- 模拟时长: {self.config['production']['time']} ns

## 结果文件

- `rmsd.xvg`: 主链RMSD
- `rmsf.xvg`: 残基RMSF
- `gyrate.xvg`: 回旋半径
- `hbond.xvg`: 氢键数量
- `sasa.xvg`: 溶剂可及表面积

## 结合能计算

使用gmx_MMPBSA计算结合自由能:

```bash
gmx_MMPBSA -O -i mmpbsa.in -cs md.tpr -ct md.xtc -cp topol.top
```

## 注意事项

1. 首次运行前，请确保已正确准备配体拓扑文件
2. 根据系统大小调整模拟时长
3. 定期检查模拟稳定性（温度、压力、RMSD）
4. 对于长时间模拟，建议使用GPU加速

## 参考资料

- GROMACS官方文档: http://manual.gromacs.org/
- gmx_MMPBSA: https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA
- GROMACS教程: http://www.mdtutorials.com/gmx/
"""
        with open(path, 'w') as f:
            f.write(readme)


def main():
    """主函数"""
    logger = setup_logger('md_guide', LOGS_DIR / 'md_simulation.log')
    
    logger.info("生成GROMACS模拟脚本...")
    
    # 创建指南生成器
    guide = MDSimulationGuide(logger)
    
    # 生成脚本
    output_dir = RESULTS_DIR / 'md_simulation_scripts'
    guide.generate_gromacs_scripts(output_dir)
    
    logger.info(f"脚本已生成到: {output_dir}")
    logger.info("请查看README_MD.md了解使用方法")


if __name__ == "__main__":
    main()

