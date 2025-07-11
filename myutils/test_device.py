# 测试GPU设备
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6,7'

def check_gpus():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    print(f"CUDA_VISIBLE_DEVICES: {visible_devices}")

    num_devices = torch.cuda.device_count()
    print(f"Number of visible CUDA devices: {num_devices}")

    if num_devices == 0:
        print("❌ No GPUs available.")
        return

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"→ Device {i}: {device_name} | {total_mem:.2f} GB")

        # 测试 tensor 分配
        try:
            x = torch.randn(1000, 1000, device=f"cuda:{i}")
            y = torch.matmul(x, x)
            print(f"   ✅ Tensor operation succeeded on cuda:{i}")
        except Exception as e:
            print(f"   ❌ Tensor operation failed on cuda:{i}: {e}")

if __name__ == "__main__":
    check_gpus()
