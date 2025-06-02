from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM
import os

def load_big_model(model_name: str, device: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    model = LLM(model_name, enable_prefix_caching=True, trust_remote_code=True)
    print(f"Initialized {model_name} with vLLM.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def load_small_model(model_name: str, device: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    model = LLM(model_name, gpu_memory_utilization=0.7, enable_prefix_caching=True, trust_remote_code=True)
    print(f"Initialized {model_name} with vLLM.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
