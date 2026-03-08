"""
LoRA 训练器核心实现
针对医疗槽位提取任务优化
"""

import os
import sys
import json
from datetime import datetime
import torch
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.model_utils import load_medical_model, print_gpu_memory
from core.utils.data_utils import load_medical_data, prepare_training_samples


class MedicalSlotTrainer:
    """
    医疗槽位提取 LoRA 训练器
    
    特性:
    - 4-bit 量化训练，降低显存占用
    - 针对 WSL2 + RTX 4060 8G 优化
    - 支持医疗领域特殊token
    - 自动保存检查点和最终模型
    """
    
    def __init__(self, output_dir="./results", overwrite_output_dir=False):
        """
        初始化训练器
        
        Args:
            output_dir: 模型输出目录
            overwrite_output_dir: 是否覆盖已存在的输出目录
        """
        self.output_dir = output_dir
        self.overwrite_output_dir = overwrite_output_dir
        
        # 检查输出目录是否存在且不为空
        if os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite_output_dir:
            raise ValueError(f"Output directory ({output_dir}) already exists and is not empty. "
                           f"Use --overwrite-output-dir to overwrite it.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print("正在加载基础模型和tokenizer...")
        print("="*50)
        self.model, self.tokenizer = load_medical_model()
        
        print("\n" + "="*50)
        print("正在加载数据集...")
        print("="*50)
        self.datasets = load_medical_data()
        
        # 准备模型进行 k-bit 训练
        print("\n" + "="*50)
        print("配置 LoRA 参数...")
        print("="*50)
        self.model = prepare_model_for_kbit_training(self.model)
        
        # 设置 LoRA 配置
        self.lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, self.lora_config)
        
        # 打印模型信息
        self.model.print_trainable_parameters()
        
        # 准备训练数据
        print("\n" + "="*50)
        print("准备训练数据...")
        print("="*50)
        self.train_data = prepare_training_samples(self.datasets["train"], self.tokenizer)
        self.val_data = prepare_training_samples(self.datasets["val"], self.tokenizer)
        
        print(f"\n✓ 训练数据准备完成:")
        print(f"  训练样本：{len(self.train_data)}条")
        print(f"  验证样本：{len(self.val_data)}条")
    
    def setup_lora_config(self):
        """配置LoRA参数，特别优化医疗槽位提取"""
        return LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def train(self, num_epochs=10, batch_size=4):
        """
        执行训练
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            dict: 训练指标
        """
        # 保存训练配置
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "lora_r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": list(self.lora_config.target_modules),
                "medical_special_tokens": getattr(self.lora_config, 'medical_special_tokens', []),
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        # 设置训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            fp16=True,
            report_to="none",
            load_best_model_at_end=True
            # Removed overwrite_output_dir due to Transformers version incompatibility
        )
        
        # 创建 Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data
        )
        
        # 开始训练
        print("\n" + "="*50)
        print(f"开始训练... {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"训练配置:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Gradient Accumulation: {training_args.gradient_accumulation_steps}")
        print(f"  Learning Rate: {training_args.learning_rate}")
        print("="*50 + "\n")
        
        train_result = trainer.train()
        
        # 保存最终模型
        final_model_path = os.path.join(self.output_dir, "final_model")
        print(f"\n正在保存最终模型到：{final_model_path}")
        self.model.save_pretrained(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        # 保存训练指标
        metrics = train_result.metrics
        metrics["train_samples"] = len(self.train_data)
        metrics["val_samples"] = len(self.val_data)
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ 训练完成！模型已保存至：{final_model_path}")
        print(f"✓ 训练指标已保存至：{os.path.join(self.output_dir, 'metrics.json')}")
        
        return metrics
    
    def evaluate(self):
        """
        评估模型性能
        
        Returns:
            dict: 评估指标
        """
        print("\n" + "="*50)
        print("开始在测试集上评估模型...")
        print("="*50)
        
        correct_symptoms = 0
        total = len(self.datasets["test"])
        
        for i, sample in enumerate(self.datasets["test"]):
            # 构建输入文本
            input_text = (
                "你是一个医疗助手，请从以下患者描述中提取关键信息：\n\n"
                f"患者描述：{sample['text']}\n\n"
                "请以 JSON 格式返回以下信息：症状 (symptom)、持续时间 (duration)、"
                "严重程度 (severity)、症状部位 (location)、伴随症状 (accompanying_symptoms)、"
                "诱发因素 (trigger)、推荐科室 (department)"
            )
            
            # Tokenize
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.model.device)
            
            # 生成预测
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=100)
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 检查症状是否正确提取
            expected_symptom = sample["slots"].get("symptom", "").lower()
            if expected_symptom and expected_symptom in response.lower():
                correct_symptoms += 1
        
        accuracy = correct_symptoms / total if total > 0 else 0
        print(f"\n症状提取准确率：{accuracy:.2%} ({correct_symptoms}/{total})")
        
        return {"symptom_accuracy": accuracy}


def train_medical_slot_model(epochs=10, batch_size=4, output_dir="./results", overwrite_output_dir=False):
    """
    一键训练函数
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        output_dir: 输出目录
        overwrite_output_dir: 是否覆盖已存在的输出目录
        
    Returns:
        MedicalSlotTrainer: 训练器实例
    """
    print("\n" + "="*70)
    print("医疗槽位提取 LoRA 训练")
    print("="*70)
    print(f"目标：基于 Qwen1.5-0.5B-Chat 微调医疗槽位提取模型")
    print(f"数据量：200 条标注数据")
    print(f"硬件环境：WSL2 + RTX 4060 8GB")
    print("="*70 + "\n")
    
    # 创建训练器
    trainer = MedicalSlotTrainer(output_dir=output_dir, overwrite_output_dir=overwrite_output_dir)
    
    # 训练模型
    trainer.train(num_epochs=epochs, batch_size=batch_size)
    
    # 评估模型
    metrics = trainer.evaluate()
    
    # 保存评估结果
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ 评估结果已保存至：{eval_path}")
    print(f"\n训练完成！最终模型位于：{os.path.join(output_dir, 'final_model')}")
    
    return trainer