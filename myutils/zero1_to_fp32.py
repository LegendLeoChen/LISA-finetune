# zero1训练后获取完整权重（代替zero_to_fp32）
import torch

# 加载你的 checkpoint 文件路径
ckpt_path = "./runs/lisa_enhancemlp/ckpt_model/global_step39000/mp_rank_00_model_states.pt"

# 加载 checkpoint 文件
ckpt = torch.load(ckpt_path, map_location="cpu")

# 检查是否包含 'module' 这个关键 key（DeepSpeed Stage 1 保存方式）
if 'module' not in ckpt:
    raise KeyError("'module' key not found in checkpoint — 这不是一个合法的 DeepSpeed ZeRO Stage 1 模型参数文件。")

# 提取模型参数
state_dict = ckpt['module']

# 打印所有参数 key（检查是不是完整的，比如包含 q_proj.weight 等）
print("\n== 模型参数 key 列表（共 {} 个） ==".format(len(state_dict)))
for key in list(state_dict.keys())[:20]:  # 只打印前 20 个参数
    print(key)

# 也可以保存为临时的 pytorch_model.bin 以供 merge LoRA 使用
torch.save(state_dict, "./pytorch_model.bin")
print("\n已保存模型参数到 pytorch_model.bin（可用于 merge_lora_weights_and_save_hf_model.py）")
