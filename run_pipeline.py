"""
完整流程运行脚本
一键运行从数据处理到虚拟筛选的完整流程
"""

import argparse
import sys
from pathlib import Path

from config import *
from utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='VEGFR2抑制剂发现 - 完整流程',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 完整流程（训练+筛选）
  python run_pipeline.py --all
  
  # 仅训练模型
  python run_pipeline.py --train
  
  # 仅虚拟筛选（需要已训练的模型）
  python run_pipeline.py --screen
  
  # 生成MD模拟脚本
  python run_pipeline.py --md-guide
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='运行完整流程（训练+筛选）')
    parser.add_argument('--train', action='store_true',
                       help='训练模型')
    parser.add_argument('--screen', action='store_true',
                       help='虚拟筛选')
    parser.add_argument('--docking', action='store_true',
                       help='分子对接')
    parser.add_argument('--md-guide', action='store_true',
                       help='生成MD模拟脚本')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger('pipeline', LOGS_DIR / 'pipeline.log')
    
    logger.info("=" * 80)
    logger.info("VEGFR2抑制剂发现流程")
    logger.info(f"实验: {EXPERIMENT_NAME} {VERSION}")
    logger.info("=" * 80)
    
    # 检查数据文件
    if not VEGFR2_DATA_PATH.exists():
        logger.error(f"数据文件不存在: {VEGFR2_DATA_PATH}")
        logger.info("请先运行数据收集脚本:")
        logger.info("  cd data_collection")
        logger.info("  python collect_vegfr2_data.py")
        sys.exit(1)
    
    # 如果没有指定任何参数，显示帮助
    if not any([args.all, args.train, args.screen, args.docking, args.md_guide]):
        parser.print_help()
        sys.exit(0)
    
    try:
        # 运行训练
        if args.all or args.train:
            logger.info("\n" + "=" * 80)
            logger.info("步骤1: 训练模型")
            logger.info("=" * 80)
            
            import train
            train.main()
            
            logger.info("模型训练完成!")
        
        # 运行虚拟筛选
        if args.all or args.screen:
            logger.info("\n" + "=" * 80)
            logger.info("步骤2: 虚拟筛选")
            logger.info("=" * 80)
            
            # 检查模型文件
            model_path = CHECKPOINT_DIR / 'best_model.pt'
            if not model_path.exists():
                logger.error(f"模型文件不存在: {model_path}")
                logger.info("请先运行训练: python run_pipeline.py --train")
                sys.exit(1)
            
            import virtual_screening
            virtual_screening.main()
            
            logger.info("虚拟筛选完成!")
        
        # 运行分子对接
        if args.docking:
            logger.info("\n" + "=" * 80)
            logger.info("步骤3: 分子对接")
            logger.info("=" * 80)
            
            import molecular_docking
            molecular_docking.main()
            
            logger.info("分子对接完成!")
        
        # 生成MD模拟脚本
        if args.md_guide:
            logger.info("\n" + "=" * 80)
            logger.info("步骤4: 生成MD模拟脚本")
            logger.info("=" * 80)
            
            import md_simulation_guide
            md_simulation_guide.main()
            
            logger.info("MD模拟脚本生成完成!")
        
        logger.info("\n" + "=" * 80)
        logger.info("流程完成!")
        logger.info("=" * 80)
        logger.info(f"\n结果保存位置: {RESULTS_DIR}")
        logger.info(f"日志保存位置: {LOGS_DIR}")
        
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()

