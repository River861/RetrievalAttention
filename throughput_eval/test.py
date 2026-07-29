import os
import sys
import json
import torch
import argparse
import random
import numpy as np
from termcolor import colored

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from model_hub import load_model, load_tokenizer, add_model_args
from config import generate_config, add_config_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_cuda_peak_memory_stats():
    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)
        torch.cuda.reset_peak_memory_stats(device_idx)


def collect_cuda_memory_stats():
    if not torch.cuda.is_available():
        return {}

    per_device = []
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)
        per_device.append(
            {
                "device": device_idx,
                "allocated_mib": torch.cuda.memory_allocated(device_idx) / (1024 ** 2),
                "reserved_mib": torch.cuda.memory_reserved(device_idx) / (1024 ** 2),
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device_idx) / (1024 ** 2),
                "peak_reserved_mib": torch.cuda.max_memory_reserved(device_idx) / (1024 ** 2),
            }
        )
    if not per_device:
        return {}

    return {
        "torch_cuda_memory_allocated_mib": sum(item["allocated_mib"] for item in per_device),
        "torch_cuda_memory_reserved_mib": sum(item["reserved_mib"] for item in per_device),
        "torch_cuda_peak_allocated_mib": max(item["peak_allocated_mib"] for item in per_device),
        "torch_cuda_peak_reserved_mib": max(item["peak_reserved_mib"] for item in per_device),
        "torch_cuda_peak_allocated_all_devices_mib": sum(item["peak_allocated_mib"] for item in per_device),
        "torch_cuda_peak_reserved_all_devices_mib": sum(item["peak_reserved_mib"] for item in per_device),
        "torch_cuda_per_device_memory": per_device,
    }


def collect_block_cache_stats(llm):
    kv_cache = getattr(llm, "kv_cache", None)
    metadata = getattr(kv_cache, "block_cache_metadata", None)
    if metadata is None:
        return {}
    return metadata()


def parse_args():
    parser = argparse.ArgumentParser(description="Test example")
    parser.add_argument("--batch_size", type=int, default=1, help="Total Batch size")
    parser.add_argument("--prefill_bsz", type=int, default=1, help="Prefilling batch size")
    parser.add_argument("--prefill_method", type=str, default="full", choices=["full", "xattn", "minfer"],
                        help="Prefilling method")
    parser.add_argument("--context_len", type=int, default=120000, help="Input context length")
    parser.add_argument("--gen_len", type=int, default=100, help="Generation length")
    parser.add_argument("--task_name", type=str, default="NIAH", choices=["NIAH", "fwe", "vt", "qa1", "AIME"],
                        help="Test task name")
    parser = add_model_args(parser)
    parser = add_config_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(2025)
    print(args)

    model_name = args.model_name
    batch_size = args.batch_size
    attn_type = args.attn_type
    dtype = torch.bfloat16
    device = args.device
    task_name = args.task_name

    TEST_DIR = os.path.join(PROJECT_ROOT, "throughput_eval/test_data")
    if task_name == "NIAH":
        TEST_FILE = os.path.join(TEST_DIR, f"NIAH_{args.context_len}.json")
        data = json.load(open(TEST_FILE))[0]
        prompt = data['input']
        groundtruth = data['answer']
    else:
        TEST_FILE = os.path.join(TEST_DIR, f"{task_name}.json")
        data = json.load(open(TEST_FILE))
        prompt = data['input']
        groundtruth = data['outputs']
    prompts = [prompt for _ in range(batch_size)]

    tokenizer = load_tokenizer(model_name)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_masks = inputs.attention_mask
    input_len = input_ids.shape[1]
    gen_len = args.gen_len
    max_len = input_len + gen_len
    print(colored(f"Input length: {input_len}, Gen length: {gen_len}", 'yellow'))

    attn_config = generate_config(model_name, input_len, attn_type, 
                                  float(args.retrieval_budget), float(args.estimation_budget), float(args.cache_ratio),
                                  args.use_cuda_graph, args.gpu_only)
    reset_cuda_peak_memory_stats()
    llm = load_model(model_name, max_len, dtype, device)

    out = llm.generate(
        attention_type=attn_type,
        inputs_ids=input_ids.to(llm.layers[0].device),
        attention_masks=attention_masks,
        max_new_length=gen_len, 
        attn_config=attn_config,
        do_sample=False, 
        ignore_eos=True,
        prefill_bsz=args.prefill_bsz,
        prefill_method=args.prefill_method
    )
    torch_memory_stats = collect_cuda_memory_stats()
    metrics = getattr(llm, "last_metrics", None)
    if metrics is not None:
        result = {
            "runner": "retroinfer_native",
            "model_name": model_name,
            "attention_type": attn_type,
            "task_name": task_name,
            "context_len_arg": args.context_len,
            "batch_size": batch_size,
            "prefill_bsz": args.prefill_bsz,
            "prefill_method": args.prefill_method,
            "retrieval_budget": float(args.retrieval_budget),
            "estimation_budget": float(args.estimation_budget),
            "cache_ratio": float(args.cache_ratio),
            "use_cuda_graph": bool(args.use_cuda_graph),
            "gpu_only": bool(args.gpu_only),
        }
        result.update(metrics)
        result.update(torch_memory_stats)
        result.update(collect_block_cache_stats(llm))
        print("RETROINFER_RESULT_JSON=" + json.dumps(result, sort_keys=True))
    
    if gen_len <= 100:
        result = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(groundtruth)
        print(result)
        task_hits = [str(groundtruth) in item for item in result]
        print("RETROINFER_OUTPUT_JSON=" + json.dumps({
            "groundtruth": groundtruth,
            "outputs": result,
            "task_contains_groundtruth": task_hits,
            "all_outputs_contain_groundtruth": all(task_hits),
        }, sort_keys=True))