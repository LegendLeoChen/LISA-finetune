#!/usr/bin/env python
# 测试zero stage 0-2的速度
import time
import argparse
import torch
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-Stage Speed Test with DeepSpeed")
    # Registers DeepSpeed config arguments (e.g., deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)

    # Local rank for distributed launch
    parser.add_argument(
        "--local_rank", type=int, default=0,
        help="Local rank passed in by deepspeed launcher"
    )

    parser.add_argument(
        "--zero_stage", type=int, default=0, choices=[0, 1, 2],
        help="ZeRO optimization stage (0, 1, or 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--seq_length", type=int, default=1024,
        help="Sequence length (for synthetic data)"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=1024,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=50,
        help="Number of timed iterations"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )

    return parser.parse_args()


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def main():
    args = parse_args()

    # DeepSpeed configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": args.lr}
        },
        "zero_optimization": {"stage": args.zero_stage}
    }

    # Build model and move to GPU
    torch.cuda.set_device(args.local_rank)
    model = SimpleModel(args.hidden_size).cuda()

    # Initialize with DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Create synthetic input
    x = torch.randn(
        args.batch_size, args.seq_length, args.hidden_size,
        device=args.local_rank
    )

    # Warmup iterations
    for _ in range(args.warmup):
        loss = model_engine(x).mean()
        model_engine.backward(loss)
        model_engine.step()

    # Timed iterations
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(args.iters):
        loss = model_engine(x).mean()
        model_engine.backward(loss)
        model_engine.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    avg_step = elapsed / args.iters
    throughput = args.batch_size / avg_step

    # Print only on global rank 0
    if model_engine.global_rank == 0:
        print(f"ZeRO stage {args.zero_stage}: avg step time = {avg_step:.4f}s, throughput = {throughput:.2f} samples/s")


if __name__ == "__main__":
    main()
