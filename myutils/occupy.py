# 占用剩余GPU的程序
import torch
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

# 模拟每张卡做一点小计算
tensors = []
for i in range(torch.cuda.device_count()):
    d = torch.device(f"cuda:{i}")
    # 创建一个小矩阵放在 GPU 上
    tensors.append(torch.randn(4, 4, device=d))

print("Starting light GPU occupation loop...")
try:
    while True:
        for i, tensor in enumerate(tensors):
            # 做一次小矩阵乘法 → 保证 GPU 有一点点算力占用
            result = tensor @ tensor.T
            _ = result.sum().item()  # 防止优化掉
        time.sleep(0.01)  # 每隔 0.5 秒运行一次
except KeyboardInterrupt:
    print("Stopped.")
