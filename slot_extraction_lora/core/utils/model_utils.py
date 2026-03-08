"""
模型加载与配置工具
针对 WSL2 + RTX 4060 8G 显存环境优化
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from modelscope import snapshot_download


def get_quantization_config():
    """
    配置 4-bit 量化，优化 WSL2+4060 环境
    
    Returns:
        BitsAndBytesConfig: 量化配置对象
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )


def download_qwen_model():
    """
    从魔搭下载Qwen 模型（国内镜像）
    
    Returns:
        str: 模型本地路径
    """
    model_dir = os.path.join(os.getcwd(), "data", "models", "qwen-0.5b-chat")
    os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型是否已下载
    actual_model_path = os.path.join(model_dir, "qwen", "Qwen1___5-0___5B-Chat")
    
    if os.path.exists(actual_model_path):
        print(f"✓ 模型已存在：{actual_model_path}")
        return actual_model_path
    
    print("开始下载Qwen1.5-0.5B-Chat模型（约 500MB）...")
    try:
        # 下载模型到指定目录
        snapshot_download(
            'qwen/Qwen1.5-0.5B-Chat',
            revision='master',
            cache_dir=model_dir,
            ignore_file_pattern=[".*.msgpack", ".*.h5"]
        )
        print(f"✓ 模型下载完成：{actual_model_path}")
        return actual_model_path
    except Exception as e:
        print(f"✗ 模型下载失败：{e}")
        raise


def load_medical_model():
    """
    加载医疗槽位提取专用模型
    
    Returns:
        tuple: (model, tokenizer)
    """
    model_path = download_qwen_model()
    
    # 量化配置
    bnb_config = get_quantization_config()
    
    print("正在加载模型到 GPU...")
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    # 设置 pad_token（必须）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    print_gpu_memory("模型加载完成")
    return model, tokenizer


def print_gpu_memory(stage=""):
    """
    打印 GPU 内存使用情况
    
    Args:
        stage: 当前阶段描述
    """
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\n[{stage}] GPU 内存使用:")
    print(f"  已分配：{allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
    print(f"  已保留：{reserved:.2f} GB / {total:.24f} GB ({reserved/total*100:.1f}%)")
    print(f"  可用：{total - reserved:.2f} GB\n")
