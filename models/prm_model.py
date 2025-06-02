import os
from reward_hub.reward_hub import AutoRM

def load_prm_model(prm_model_path: str, device: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]
    os.environ["VLLM_USE_V1"] = "0"
    model = AutoRM.load(prm_model_path, load_method="vllm", device=None)
    print(f"Initialized {prm_model_path} with RewardHub vLLM.")
    return model
