#!/usr/bin/env python3
"""
医疗槽位提取 LoRA 训练脚本

基于 Qwen1.5-0.5B-Chat + PEFT + Transformers
针对 WSL2 + RTX 4060 8GB显存环境优化

使用方法:
    python train_lora.py [--epochs 10] [--batch-size 4] [--output-dir ./results]

作者：AI Agent 工程师
日期：2026-03-07
"""

import os
import sys
import argparse
import json
from datetime import datetime


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='医疗槽位提取 LoRA 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认训练（10 轮，batch size=4）
  python train_lora.py
  
  # 自定义训练轮数和批次大小
  python train_lora.py --epochs 15 --batch-size 2
  
  # 指定输出目录
  python train_lora.py --output-dir ./my_results
        """
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10, 
        help='训练轮数 (默认：10)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=4, 
        help='批次大小 (默认：4)'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./results', 
        help='模型输出目录 (默认：./results)'
    )
    parser.add_argument(
        '--overwrite-output-dir',
        action='store_true',
        help='覆盖已存在的输出目录'
    )
    
    args = parser.parse_args()
    
    # 打印训练信息
    print("\n" + "="*70)
    print("🏥 医疗槽位提取 LoRA 训练系统")
    print("="*70)
    print(f"📅 开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚙️  训练轮数：{args.epochs}")
    print(f"📦 批次大小：{args.batch_size}")
    print(f"💾 输出目录：{args.output_dir}")
    print(f"🎯 底座模型：Qwen1.5-0.5B-Chat")
    print(f"📊 数据规模：200 条标注数据")
    print(f"🖥️  硬件环境：WSL2 + RTX 4060 8GB")
    print("="*70 + "\n")
    
    # 导入训练模块
    try:
        from core.trainer.lora_trainer import train_medical_slot_model
    except ImportError as e:
        print(f"❌ 错误：无法导入训练模块")
        print(f"详情：{e}")
        print(f"\n请确保当前工作目录为：{os.path.dirname(os.path.abspath(__file__))}")
        sys.exit(1)
    
    # 开始训练
    try:
        trainer = train_medical_slot_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            overwrite_output_dir=args.overwrite_output_dir
        )
        
        # 打印最终结果
        print("\n" + "="*70)
        print("✅ 训练完成！")
        print("="*70)
        
        # 读取并显示评估结果
        eval_path = os.path.join(args.output_dir, "evaluation.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r") as f:
                eval_metrics = json.load(f)
            
            print(f"\n📊 评估结果:")
            print(f"  症状提取准确率：{eval_metrics.get('symptom_accuracy', 0)*100:.2f}%")
        
        # 读取训练指标
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            print(f"\n📈 训练指标:")
            print(f"  训练损失：{metrics.get('train_loss', 0):.4f}")
            print(f"  训练时间：{metrics.get('train_runtime', 0)/60:.2f}分钟")
            print(f"  每秒样本数：{metrics.get('train_samples_per_second', 0):.2f}")
        
        print(f"\n💾 模型保存位置：{os.path.join(args.output_dir, 'final_model')}")
        print(f"📁 检查点目录：{args.output_dir}/checkpoint-*")
        print(f"📄 日志文件：{args.output_dir}/logs/")
        print("="*70 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        return 1
        
    except Exception as e:
        print(f"\n❌ 训练失败：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
