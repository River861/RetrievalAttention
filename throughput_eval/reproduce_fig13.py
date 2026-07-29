#!/usr/bin/env python3
"""Run and parse single-GPU RetroInfer Figure 13 throughput experiments."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import importlib.util
import json
import math
import os
import platform
import re
import shlex
import shutil
import statistics
import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THROUGHPUT_DIR = PROJECT_ROOT / "throughput_eval"
DEFAULT_MODEL = "gradientai/Llama-3-8B-Instruct-Gradient-1048k"
DEFAULT_TASK = "NIAH"
RESULT_JSON_RE = re.compile(r"(RETROINFER|VLLM)_RESULT_JSON=(\{.*\})")
MEMORY_JSON_RE = re.compile(r"FIG13_MEMORY_JSON=(\{.*\})")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

PAPER_BATCHES: dict[str, dict[int, list[int]]] = {
    "Full_Flash_Attn": {
        30000: [1, 2, 4, 8, 16],
        60000: [1, 2, 4, 8],
        120000: [1, 2, 4],
        1024000: [],
    },
    "RetroInfer": {
        30000: [1, 2, 4, 8, 16, 32, 64, 128],
        60000: [1, 2, 4, 8, 16, 32, 64],
        120000: [1, 2, 4, 8, 16, 32],
        1024000: [1, 2, 4],
    },
    "RetroInfer_GPU": {
        30000: [1, 2, 4, 8, 16],
        60000: [1, 2, 4, 8],
        120000: [1, 2, 3],
        1024000: [],
    },
}

FIGURE_SPECS = [
    {
        "name": "throughput",
        "basename": "fig13_single_gpu_throughput",
        "metric": "mean_decode_throughput_tokens_s",
        "scale": 1.0,
        "title": "Figure 13 single-GPU decoding throughput",
        "ylabel": "Decode throughput (tokens/s)",
    },
    {
        "name": "gpu_memory",
        "basename": "fig13_single_gpu_gpu_memory",
        "metric": "mean_peak_process_gpu_memory_mib",
        "scale": 1.0 / 1024.0,
        "title": "Figure 13 single-GPU GPU memory",
        "ylabel": "Peak process GPU memory (GiB)",
    },
]
FIGURE_EXTENSIONS = ("pdf", "svg", "png")
METHOD_ORDER = ["Full_Flash_Attn", "RetroInfer", "RetroInfer_GPU", "vLLM"]
METHOD_LABELS = {
    "Full_Flash_Attn": "Full FlashAttn",
    "RetroInfer": "RetroInfer",
    "RetroInfer_GPU": "RetroInfer GPU",
    "vLLM": "vLLM",
}
METHOD_STYLES = {
    "Full_Flash_Attn": {"color": "#0072B2", "shape": "circle"},
    "RetroInfer": {"color": "#D55E00", "shape": "square"},
    "RetroInfer_GPU": {"color": "#009E73", "shape": "triangle"},
    "vLLM": {"color": "#CC79A7", "shape": "diamond"},
}

BITMAP_FONT_5X7 = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "J": ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "01100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ";": ["00000", "01100", "01100", "00000", "01100", "01100", "01000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
}

RUN_FIELDS = [
    "run_id",
    "suite",
    "method",
    "model_name",
    "task_name",
    "context_len",
    "batch_size",
    "gen_len",
    "round",
    "status",
    "returncode",
    "duration_s",
    "input_length",
    "prefill_latency_s",
    "decode_latency_s",
    "decode_steps",
    "decode_ms_per_step",
    "decode_throughput_tokens_s",
    "e2e_latency_s",
    "avg_e2e_latency_s",
    "request_throughput_req_s",
    "output_throughput_tokens_s",
    "torch_cuda_memory_allocated_mib",
    "torch_cuda_memory_reserved_mib",
    "torch_cuda_peak_allocated_mib",
    "torch_cuda_peak_reserved_mib",
    "torch_cuda_peak_allocated_all_devices_mib",
    "torch_cuda_peak_reserved_all_devices_mib",
    "block_cache_source",
    "block_cache_ratio",
    "block_cache_index_clusters",
    "block_cache_retrieval_clusters",
    "block_cache_clusters_per_layer",
    "block_cache_pages_per_cluster",
    "block_cache_pages_per_layer",
    "block_cache_page_size_vectors",
    "block_cache_vectors_per_layer",
    "block_cache_layer_count",
    "block_cache_total_pages",
    "block_cache_total_vectors",
    "block_cache_dtype_bytes",
    "block_cache_bytes_per_layer",
    "block_cache_total_bytes",
    "peak_process_gpu_memory_mib",
    "gpu_memory_sample_interval_s",
    "gpu_memory_sample_count",
    "gpu_memory_observed_process_count_peak",
    "gpu_memory_sampler_error_count",
    "gpu_memory_samples_path",
    "log_path",
    "error_summary",
]

SUMMARY_FIELDS = [
    "suite",
    "method",
    "model_name",
    "task_name",
    "context_len",
    "batch_size",
    "gen_len",
    "total_runs",
    "passed_runs",
    "failed_runs",
    "mean_decode_throughput_tokens_s",
    "stdev_decode_throughput_tokens_s",
    "mean_decode_latency_s",
    "mean_e2e_latency_s",
    "mean_request_throughput_req_s",
    "mean_output_throughput_tokens_s",
    "mean_torch_cuda_peak_allocated_mib",
    "mean_torch_cuda_peak_reserved_mib",
    "mean_peak_process_gpu_memory_mib",
    "max_peak_process_gpu_memory_mib",
]


def clean_text(text: str) -> str:
    return ANSI_RE.sub("", text)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_method_list(value: str) -> list[str]:
    methods = [part.strip() for part in value.split(",") if part.strip()]
    allowed = set(PAPER_BATCHES) | {"vLLM"}
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown method(s): {', '.join(unknown)}")
    return methods


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def run_command(command: list[str], cwd: Path | None = None, timeout: int = 30) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {"returncode": proc.returncode, "output": proc.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "output": str(exc)}


def tracked_pids(root_pid: int) -> set[int]:
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        children_path = Path("/proc") / str(pid) / "task" / str(pid) / "children"
        try:
            children = [int(item) for item in children_path.read_text(encoding="utf-8").split()]
        except (FileNotFoundError, ProcessLookupError):
            children = []
        stack.extend(child for child in children if child not in seen)
    return seen


def query_compute_apps() -> tuple[list[dict[str, Any]], str | None]:
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found"
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return [], proc.stderr.strip() or proc.stdout.strip() or f"nvidia-smi exited {proc.returncode}"

    apps: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            return apps, f"unexpected nvidia-smi compute-apps row: {line}"
        gpu_uuid, pid_text, process_name, used_memory_text = parts
        try:
            pid = int(pid_text)
            used_memory_mib = float(used_memory_text.split()[0])
        except (IndexError, ValueError) as exc:
            return apps, f"cannot parse nvidia-smi compute-apps row {line!r}: {exc}"
        apps.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": pid,
                "process_name": process_name,
                "used_memory_mib": used_memory_mib,
            }
        )
    return apps, None


def query_gpu_uuid_by_index() -> tuple[dict[str, str], str | None]:
    if shutil.which("nvidia-smi") is None:
        return {}, "nvidia-smi not found"
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {}, proc.stderr.strip() or proc.stdout.strip() or f"nvidia-smi exited {proc.returncode}"

    uuid_by_index: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=1)]
        if len(parts) != 2:
            return uuid_by_index, f"unexpected nvidia-smi GPU row: {line}"
        uuid_by_index[parts[0]] = parts[1]
    return uuid_by_index, None


def visible_gpu_uuids(cuda_visible_devices: str) -> tuple[set[str] | None, str | None]:
    value = cuda_visible_devices.strip()
    if not value or value in {"all", "ALL"}:
        return None, None

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    uuid_tokens = {token for token in tokens if token.startswith("GPU-")}
    index_tokens = [token for token in tokens if not token.startswith("GPU-")]
    if not index_tokens:
        return uuid_tokens, None

    uuid_by_index, error = query_gpu_uuid_by_index()
    if error is not None:
        return None, error

    missing = [token for token in index_tokens if token not in uuid_by_index]
    if missing:
        return None, f"CUDA_VISIBLE_DEVICES references unknown GPU index/indices: {', '.join(missing)}"
    return uuid_tokens | {uuid_by_index[token] for token in index_tokens}, None


def write_memory_event(sample_path: Path, event: dict[str, Any]) -> None:
    with sample_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def sample_gpu_memory(sample_path: Path, root_pid: int, run_id: str, sample_index: int, started_at: float) -> dict[str, Any]:
    pids = tracked_pids(root_pid)
    apps, error = query_compute_apps()
    processes = [app for app in apps if app["pid"] in pids]
    event = {
        "event": "sample",
        "run_id": run_id,
        "sample_index": sample_index,
        "sampled_at": utc_now(),
        "elapsed_s": time.time() - started_at,
        "root_pid": root_pid,
        "tracked_pids": sorted(pids),
        "processes": processes,
        "total_used_memory_mib": sum(float(app["used_memory_mib"]) for app in processes),
    }
    if error is not None:
        event["error"] = error
    write_memory_event(sample_path, event)
    return event


def summarize_memory_samples(sample_path: Path, output_dir: Path, interval_s: float) -> dict[str, Any]:
    if not sample_path.is_file():
        return {}

    sample_count = 0
    error_count = 0
    peak_process_gpu_memory_mib: float | None = None
    peak_sample_at: str | None = None
    observed_process_count_peak = 0
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "sample":
            if event.get("event") == "error":
                error_count += 1
            continue
        sample_count += 1
        if event.get("error"):
            error_count += 1
        observed_process_count_peak = max(observed_process_count_peak, len(event.get("processes") or []))
        memory_mib = float(event.get("total_used_memory_mib") or 0.0)
        if peak_process_gpu_memory_mib is None or memory_mib > peak_process_gpu_memory_mib:
            peak_process_gpu_memory_mib = memory_mib
            peak_sample_at = event.get("sampled_at")

    if peak_process_gpu_memory_mib is None:
        return {
            "gpu_memory_sample_interval_s": interval_s,
            "gpu_memory_sample_count": sample_count,
            "gpu_memory_observed_process_count_peak": observed_process_count_peak,
            "gpu_memory_sampler_error_count": error_count,
            "gpu_memory_samples_path": str(sample_path.relative_to(output_dir)),
        }

    return {
        "peak_process_gpu_memory_mib": peak_process_gpu_memory_mib,
        "peak_process_gpu_memory_sample_at": peak_sample_at,
        "gpu_memory_sample_interval_s": interval_s,
        "gpu_memory_sample_count": sample_count,
        "gpu_memory_observed_process_count_peak": observed_process_count_peak,
        "gpu_memory_sampler_error_count": error_count,
        "gpu_memory_samples_path": str(sample_path.relative_to(output_dir)),
    }


def visible_gpu_compute_apps(cuda_visible_devices: str) -> list[dict[str, Any]]:
    apps, error = query_compute_apps()
    if error is not None:
        return [{"error": error}]
    uuids, uuid_error = visible_gpu_uuids(cuda_visible_devices)
    if uuid_error is not None:
        return [{"error": uuid_error, "all_compute_apps": apps}]
    if uuids is None:
        return apps
    return [app for app in apps if app.get("gpu_uuid") in uuids]


def collect_environment() -> dict[str, Any]:
    packages = [
        "torch",
        "transformers",
        "vllm",
        "flash_attn",
        "flashinfer",
        "retroinfer_kernels",
        "minference",
    ]
    package_status = {
        package: ("found" if importlib.util.find_spec(package) is not None else "missing")
        for package in packages
    }
    commands = {
        "git_head": ["git", "--no-pager", "rev-parse", "HEAD"],
        "git_status": ["git", "--no-pager", "status", "--short", "--untracked-files=all"],
        "python_version": [sys.executable, "--version"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "nvidia_smi_query": [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        "nvidia_smi": ["nvidia-smi"],
        "numactl_hardware": ["numactl", "--hardware"],
        "nvcc_version": ["nvcc", "--version"],
        "cuda_nvcc_version": ["/usr/local/cuda/bin/nvcc", "--version"],
        "lscpu": ["lscpu"],
    }
    return {
        "collected_at": utc_now(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "package_status": package_status,
        "commands": {
            name: run_command(command, cwd=PROJECT_ROOT, timeout=20)
            for name, command in commands.items()
        },
    }


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.context_lens is not None:
        context_lens = args.context_lens
    else:
        context_lens = [30000, 60000, 120000, 1024000]

    runs: list[dict[str, Any]] = []
    for method in args.methods:
        for context_len in context_lens:
            if args.batch_sizes is not None:
                batch_sizes = args.batch_sizes
            elif method == "vLLM":
                batch_sizes = PAPER_BATCHES["Full_Flash_Attn"].get(context_len, [])
            else:
                batch_sizes = PAPER_BATCHES[method].get(context_len, [])
            for batch_size in batch_sizes:
                for round_idx in range(1, args.rounds + 1):
                    run_id = (
                        f"{args.suite}_{method.lower().replace('_', '-')}"
                        f"_ctx{context_len}_bsz{batch_size}_r{round_idx}"
                    )
                    runs.append(
                        {
                            "run_id": run_id,
                            "suite": args.suite,
                            "method": method,
                            "model_name": args.model_name,
                            "task_name": args.task_name,
                            "context_len": context_len,
                            "batch_size": batch_size,
                            "gen_len": args.gen_len,
                            "round": round_idx,
                        }
                    )
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    return runs


def command_for_run(run: dict[str, Any], args: argparse.Namespace) -> list[str]:
    method = run["method"]
    if method == "vLLM":
        command = [
            sys.executable,
            "-u",
            "test_vllm.py",
            "--model_name",
            run["model_name"],
            "--context_len",
            str(run["context_len"]),
            "--task_name",
            run["task_name"],
            "--gen_len",
            str(run["gen_len"]),
            "--batch_size",
            str(run["batch_size"]),
        ]
    else:
        attn_type = "RetroInfer" if method == "RetroInfer_GPU" else method
        command = [
            sys.executable,
            "-u",
            "test.py",
            "--model_name",
            run["model_name"],
            "--attn_type",
            attn_type,
            "--context_len",
            str(run["context_len"]),
            "--task_name",
            run["task_name"],
            "--gen_len",
            str(run["gen_len"]),
            "--batch_size",
            str(run["batch_size"]),
            "--prefill_bsz",
            str(args.prefill_bsz),
            "--retrieval_budget",
            str(args.retrieval_budget),
            "--estimation_budget",
            str(args.estimation_budget),
            "--cache_ratio",
            str(args.cache_ratio),
        ]
        if args.use_cuda_graph and method.startswith("RetroInfer"):
            command.append("--use_cuda_graph")
        if method == "RetroInfer_GPU":
            command.append("--gpu_only")

    if args.use_numactl and shutil.which("numactl") is not None:
        return ["numactl", "--cpunodebind=0", "--membind=0"] + command
    return command


def parse_error_summary(text: str) -> str:
    cleaned = clean_text(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        if (
            "ModuleNotFoundError:" in line
            or "ImportError:" in line
            or "CUDA out of memory" in line
            or "OutOfMemoryError" in line
            or "RuntimeError:" in line
            or "ValueError:" in line
            or "AssertionError:" in line
        ):
            return line
    for idx, line in enumerate(lines):
        if line.startswith("Traceback "):
            return lines[min(len(lines) - 1, idx + 1)]
    return lines[-1] if lines else ""


def parse_log_text(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    parsed: dict[str, Any] = {}
    for match in RESULT_JSON_RE.finditer(cleaned):
        try:
            parsed.update(json.loads(match.group(2)))
        except json.JSONDecodeError:
            pass
    for match in MEMORY_JSON_RE.finditer(cleaned):
        try:
            parsed.update(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass

    patterns: list[tuple[str, str, Any]] = [
        ("input_length", r"Input length:\s*(\d+),\s*Gen length:\s*(\d+)", int),
        ("prefill_latency_s", r"Prefilling latency:\s*([0-9.]+)\s*s", float),
        ("decode_latency_s", r"Decoding latency:\s*([0-9.]+)\s*s", float),
        ("decode_ms_per_step", r"Decoding latency:[^\n]*\(([0-9.]+)\s*ms/step\)", float),
        ("decode_throughput_tokens_s", r"Throughput:\s*([0-9.]+)\s*tokens/s", float),
        ("e2e_latency_s", r"End2End Latency:\s*([0-9.]+)\s*s", float),
        ("e2e_wall_time_s", r"End2End time:\s*([0-9.]+)\s*seconds", float),
        ("avg_e2e_latency_s", r"Avg\.\s*E2E Latency\s*([0-9.]+)\s*s/req", float),
    ]
    for key, pattern, caster in patterns:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        if key == "input_length":
            parsed.setdefault("input_length", int(match.group(1)))
            parsed.setdefault("gen_len", int(match.group(2)))
        else:
            parsed.setdefault(key, caster(match.group(1)))

    if "avg_e2e_latency_s" in parsed and "batch_size" in parsed and parsed["avg_e2e_latency_s"]:
        parsed.setdefault("request_throughput_req_s", parsed["batch_size"] / parsed["avg_e2e_latency_s"])
    if "e2e_wall_time_s" in parsed and "batch_size" in parsed and "gen_len" in parsed and parsed["e2e_wall_time_s"]:
        parsed.setdefault(
            "output_throughput_tokens_s",
            parsed["batch_size"] * parsed["gen_len"] / parsed["e2e_wall_time_s"],
        )
    parsed["error_summary"] = parse_error_summary(cleaned)
    return parsed


def run_one(run: dict[str, Any], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    logs_dir = output_dir / "raw_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "memory_samples"
    if args.gpu_memory_sampling:
        samples_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run['run_id']}.txt"
    sample_path = samples_dir / f"{run['run_id']}.jsonl"
    command = command_for_run(run, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONUNBUFFERED"] = "1"

    started = utc_now()
    start = time.time()
    returncode: int | None = None
    timeout_s = args.timeout_seconds if args.timeout_seconds > 0 else None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# run_id: {run['run_id']}\n")
        log.write(f"# started_at: {started}\n")
        log.write(f"# cwd: {THROUGHPUT_DIR}\n")
        log.write(f"# command: {shlex.join(command)}\n")
        log.write(f"# CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}\n\n")
        log.flush()
        if args.dry_run:
            log.write("# dry_run: command was not executed\n")
            returncode = 0
        else:
            try:
                if args.gpu_memory_sampling:
                   sample_path.write_text("", encoding="utf-8")
                   before_apps = visible_gpu_compute_apps(args.cuda_visible_devices)
                   write_memory_event(
                       sample_path,
                       {
                           "event": "before_run_compute_apps",
                           "run_id": run["run_id"],
                           "sampled_at": utc_now(),
                           "apps": before_apps,
                       },
                   )
                   other_apps = [app for app in before_apps if app.get("pid") is not None]
                   if args.require_idle_gpu and other_apps:
                       for app in other_apps:
                           log.write(f"# gpu_idle_check_other_process: {json.dumps(app, sort_keys=True)}\n")
                       raise RuntimeError(
                           f"GPU is not idle before {run['run_id']}; {len(other_apps)} compute app(s) observed"
                       )

                proc = subprocess.Popen(
                    command,
                    cwd=THROUGHPUT_DIR,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                sample_index = 0
                deadline = time.time() + timeout_s if timeout_s is not None else None
                while True:
                   if args.gpu_memory_sampling:
                       sample_gpu_memory(sample_path, proc.pid, run["run_id"], sample_index, start)
                       sample_index += 1
                   if deadline is not None and time.time() >= deadline:
                       proc.terminate()
                       try:
                           proc.wait(timeout=10)
                       except subprocess.TimeoutExpired:
                           proc.kill()
                           proc.wait()
                       log.write(f"\n# timeout after {timeout_s} seconds\n")
                       returncode = 124
                       break
                   if args.gpu_memory_sampling:
                       wait_timeout = args.gpu_memory_sample_interval_s
                   elif deadline is not None:
                       wait_timeout = max(0.0, deadline - time.time())
                   else:
                       wait_timeout = None
                   if deadline is not None and wait_timeout is not None:
                       wait_timeout = max(0.0, min(wait_timeout, deadline - time.time()))
                   try:
                       returncode = proc.wait(timeout=wait_timeout)
                       if args.gpu_memory_sampling:
                           sample_gpu_memory(sample_path, proc.pid, run["run_id"], sample_index, start)
                       break
                   except subprocess.TimeoutExpired:
                       continue
                if args.gpu_memory_sampling:
                   write_memory_event(
                       sample_path,
                       {
                           "event": "after_run_compute_apps",
                           "run_id": run["run_id"],
                           "sampled_at": utc_now(),
                           "apps": visible_gpu_compute_apps(args.cuda_visible_devices),
                       },
                   )
            except subprocess.TimeoutExpired as exc:
                log.write(f"\n# timeout after {exc.timeout} seconds\n")
                returncode = 124
            except OSError as exc:
                log.write(f"\n# failed to launch command: {exc}\n")
                returncode = 127
            except RuntimeError as exc:
                log.write(f"\n# failed before launch: {exc}\n")
                returncode = 126
        finished = utc_now()
        memory_summary = summarize_memory_samples(sample_path, output_dir, args.gpu_memory_sample_interval_s)
        if memory_summary:
            log.write("FIG13_MEMORY_JSON=" + json.dumps(memory_summary, sort_keys=True) + "\n")
        log.write(f"\n# finished_at: {finished}\n")
        log.write(f"# returncode: {returncode}\n")

    duration_s = time.time() - start
    parsed = parse_log_text(log_path.read_text(encoding="utf-8", errors="replace"))
    success_metric = "decode_throughput_tokens_s" in parsed or "request_throughput_req_s" in parsed
    status = "passed" if returncode == 0 and (args.dry_run or success_metric) else "failed"
    result = {
        **run,
        **parsed,
        "status": status,
        "returncode": returncode,
        "duration_s": duration_s,
        "log_path": str(log_path.relative_to(output_dir)),
    }
    if args.dry_run:
        result["error_summary"] = "dry run; no measurements executed"
    elif status == "passed":
        result["error_summary"] = ""
    elif status == "failed" and not result.get("error_summary"):
        result["error_summary"] = "command failed or no throughput metric was parsed"
    return result


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("suite"),
            result.get("method"),
            result.get("model_name"),
            result.get("task_name"),
            result.get("context_len"),
            result.get("batch_size"),
            result.get("gen_len"),
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        passed = [item for item in items if item.get("status") == "passed"]

        def mean_of(field: str) -> float | None:
            values = [float(item[field]) for item in passed if item.get(field) is not None]
            return statistics.mean(values) if values else None

        throughputs = [
            float(item["decode_throughput_tokens_s"])
            for item in passed
            if item.get("decode_throughput_tokens_s") is not None
        ]
        rows.append(
            {
                "suite": key[0],
                "method": key[1],
                "model_name": key[2],
                "task_name": key[3],
                "context_len": key[4],
                "batch_size": key[5],
                "gen_len": key[6],
                "total_runs": len(items),
                "passed_runs": len(passed),
                "failed_runs": len(items) - len(passed),
                "mean_decode_throughput_tokens_s": statistics.mean(throughputs) if throughputs else None,
                "stdev_decode_throughput_tokens_s": statistics.stdev(throughputs) if len(throughputs) > 1 else None,
                "mean_decode_latency_s": mean_of("decode_latency_s"),
                "mean_e2e_latency_s": mean_of("e2e_latency_s") or mean_of("avg_e2e_latency_s"),
                "mean_request_throughput_req_s": mean_of("request_throughput_req_s"),
                "mean_output_throughput_tokens_s": mean_of("output_throughput_tokens_s"),
                "mean_torch_cuda_peak_allocated_mib": mean_of("torch_cuda_peak_allocated_mib"),
                "mean_torch_cuda_peak_reserved_mib": mean_of("torch_cuda_peak_reserved_mib"),
                "mean_peak_process_gpu_memory_mib": mean_of("peak_process_gpu_memory_mib"),
                "max_peak_process_gpu_memory_mib": (
                    max(
                        float(item["peak_process_gpu_memory_mib"])
                        for item in passed
                        if item.get("peak_process_gpu_memory_mib") is not None
                    )
                    if any(item.get("peak_process_gpu_memory_mib") is not None for item in passed)
                    else None
                ),
            }
        )
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]], fields: list[str], limit: int | None = None) -> str:
    visible = rows if limit is None else rows[:limit]
    if not visible:
        return "_No rows._"
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in visible:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                value = f"{value:.4f}"
            elif value is None:
                value = ""
            values.append(str(value).replace("|", "\\|"))
        body.append("| " + " | ".join(values) + " |")
    suffix = ""
    if limit is not None and len(rows) > limit:
        suffix = f"\n\n_Only first {limit} of {len(rows)} rows shown; see CSV/JSONL artifacts._"
    return "\n".join([header, sep, *body]) + suffix


def number_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def context_label(context_len: int) -> str:
    if context_len >= 1000 and context_len % 1000 == 0:
        return f"{context_len // 1000}K"
    return str(context_len)


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def nice_ticks(max_value: float, tick_count: int = 5) -> tuple[float, list[float]]:
    if max_value <= 0 or not math.isfinite(max_value):
        return 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    raw_step = max_value / max(1, tick_count - 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    axis_max = step * math.ceil(max_value / step)
    tick_total = int(round(axis_max / step))
    ticks = [idx * step for idx in range(tick_total + 1)]
    return axis_max, ticks


def format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def collect_plot_points(
    summary: list[dict[str, Any]],
    metric: str,
    scale: float,
) -> dict[int, dict[str, list[tuple[int, float]]]]:
    points: dict[int, dict[str, list[tuple[int, float]]]] = {}
    for row in summary:
        if int_or_none(row.get("passed_runs")) == 0:
            continue
        context_len = int_or_none(row.get("context_len"))
        batch_size = int_or_none(row.get("batch_size"))
        value = number_or_none(row.get(metric))
        method = str(row.get("method") or "")
        if context_len is None or batch_size is None or value is None or not method:
            continue
        points.setdefault(context_len, {}).setdefault(method, []).append((batch_size, value * scale))
    for methods in points.values():
        for method_points in methods.values():
            method_points.sort(key=lambda item: item[0])
    return points


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def marker_polygon(shape: str, x: float, y: float, size: float) -> list[tuple[float, float]]:
    if shape == "triangle":
        return [(x, y - size), (x + size, y + size), (x - size, y + size)]
    if shape == "diamond":
        return [(x, y - size), (x + size, y), (x, y + size), (x - size, y)]
    return [
        (x - size, y - size),
        (x + size, y - size),
        (x + size, y + size),
        (x - size, y + size),
    ]


def estimate_text_width(text: str, size: float) -> float:
    return len(text) * size * 0.56


def text_x_for_anchor(x: float, text: str, size: float, anchor: str) -> float:
    if anchor == "middle":
        return x - estimate_text_width(text, size) / 2
    if anchor == "end":
        return x - estimate_text_width(text, size)
    return x


def make_plot_primitives(
    points: dict[int, dict[str, list[tuple[int, float]]]],
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    width = 1800
    height = 760
    margin_left = 105
    margin_right = 45
    margin_top = 145
    margin_bottom = 130
    panel_gap = 72
    contexts = sorted(points)
    panel_count = max(1, len(contexts))
    panel_width = (width - margin_left - margin_right - panel_gap * (panel_count - 1)) / panel_count
    panel_height = height - margin_top - margin_bottom
    all_values = [
        value
        for methods in points.values()
        for method_points in methods.values()
        for _, value in method_points
    ]
    axis_max, y_ticks = nice_ticks(max(all_values) * 1.08 if all_values else 1.0)
    primitives: list[dict[str, Any]] = [
        {"type": "rect", "x": 0, "y": 0, "w": width, "h": height, "fill": "#FFFFFF", "stroke": None},
        {"type": "text", "x": width / 2, "y": 38, "text": title, "size": 25, "anchor": "middle", "color": "#111111"},
        {
            "type": "text",
            "x": width / 2,
            "y": 68,
            "text": "Figure 13 single-GPU matrix; markers are measured batch sizes",
            "size": 15,
            "anchor": "middle",
            "color": "#444444",
        },
        {"type": "text", "x": margin_left, "y": 105, "text": ylabel, "size": 16, "anchor": "start", "color": "#111111"},
    ]

    methods = sorted(
        {method for context_methods in points.values() for method in context_methods},
        key=method_sort_key,
    )
    legend_x = width / 2 - len(methods) * 120 / 2
    for idx, method in enumerate(methods):
        style = METHOD_STYLES.get(method, {"color": "#333333", "shape": "circle"})
        x = legend_x + idx * 150
        y = 96
        primitives.append({"type": "line", "x1": x, "y1": y, "x2": x + 35, "y2": y, "stroke": style["color"], "width": 4})
        primitives.append({"type": "marker", "x": x + 17.5, "y": y, "shape": style["shape"], "size": 7, "fill": style["color"], "stroke": "#FFFFFF", "width": 1.2})
        primitives.append({"type": "text", "x": x + 45, "y": y + 5, "text": METHOD_LABELS.get(method, method), "size": 14, "anchor": "start", "color": "#222222"})

    for panel_idx, context_len in enumerate(contexts):
        panel_x0 = margin_left + panel_idx * (panel_width + panel_gap)
        panel_y0 = margin_top
        panel_x1 = panel_x0 + panel_width
        panel_y1 = panel_y0 + panel_height
        methods_for_context = points[context_len]
        batches = sorted({batch for method_points in methods_for_context.values() for batch, _ in method_points})
        min_log = math.log2(min(batches)) if batches else 0.0
        max_log = math.log2(max(batches)) if batches else 1.0
        if min_log == max_log:
            min_log -= 0.5
            max_log += 0.5

        def x_for_batch(batch_size: int) -> float:
            return panel_x0 + (math.log2(batch_size) - min_log) / (max_log - min_log) * panel_width

        def y_for_value(value: float) -> float:
            return panel_y1 - value / axis_max * panel_height

        primitives.append({"type": "text", "x": (panel_x0 + panel_x1) / 2, "y": panel_y0 - 20, "text": f"{context_label(context_len)} context", "size": 17, "anchor": "middle", "color": "#111111"})
        primitives.append({"type": "line", "x1": panel_x0, "y1": panel_y1, "x2": panel_x1, "y2": panel_y1, "stroke": "#222222", "width": 1.4})
        primitives.append({"type": "line", "x1": panel_x0, "y1": panel_y0, "x2": panel_x0, "y2": panel_y1, "stroke": "#222222", "width": 1.4})
        for tick in y_ticks:
            y = y_for_value(tick)
            primitives.append({"type": "line", "x1": panel_x0, "y1": y, "x2": panel_x1, "y2": y, "stroke": "#E2E2E2", "width": 1})
            if panel_idx == 0:
                primitives.append({"type": "text", "x": panel_x0 - 10, "y": y + 5, "text": format_tick(tick), "size": 12, "anchor": "end", "color": "#333333"})
        for batch_size in batches:
            x = x_for_batch(batch_size)
            primitives.append({"type": "line", "x1": x, "y1": panel_y1, "x2": x, "y2": panel_y1 + 6, "stroke": "#222222", "width": 1.2})
            primitives.append({"type": "text", "x": x, "y": panel_y1 + 24, "text": str(batch_size), "size": 12, "anchor": "middle", "color": "#333333"})
        primitives.append({"type": "text", "x": (panel_x0 + panel_x1) / 2, "y": panel_y1 + 54, "text": "Batch size (log2 scale)", "size": 13, "anchor": "middle", "color": "#222222"})

        for method in methods:
            method_points = methods_for_context.get(method, [])
            if not method_points:
                continue
            style = METHOD_STYLES.get(method, {"color": "#333333", "shape": "circle"})
            xy_points = [(x_for_batch(batch), y_for_value(value)) for batch, value in method_points]
            if len(xy_points) > 1:
                primitives.append({"type": "polyline", "points": xy_points, "stroke": style["color"], "width": 4})
            for x, y in xy_points:
                primitives.append({"type": "marker", "x": x, "y": y, "shape": style["shape"], "size": 7, "fill": style["color"], "stroke": "#FFFFFF", "width": 1.2})

    primitives.append(
        {
            "type": "text",
            "x": width / 2,
            "y": height - 24,
            "text": "Lines connect only existing measured points; unequal batch matrices are not silently aligned.",
            "size": 13,
            "anchor": "middle",
            "color": "#444444",
        }
    )
    return {"width": width, "height": height, "primitives": primitives}


def render_svg(drawing: dict[str, Any]) -> str:
    width = drawing["width"]
    height = drawing["height"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>RetroInfer Figure 13 single-GPU reproduction</title>",
        '<g font-family="DejaVu Sans, Arial, sans-serif">',
    ]
    for primitive in drawing["primitives"]:
        typ = primitive["type"]
        if typ == "rect":
            stroke = primitive.get("stroke") or "none"
            fill = primitive.get("fill") or "none"
            lines.append(
                f'<rect x="{primitive["x"]:.3f}" y="{primitive["y"]:.3f}" width="{primitive["w"]:.3f}" height="{primitive["h"]:.3f}" fill="{fill}" stroke="{stroke}" />'
            )
        elif typ == "line":
            lines.append(
                f'<line x1="{primitive["x1"]:.3f}" y1="{primitive["y1"]:.3f}" x2="{primitive["x2"]:.3f}" y2="{primitive["y2"]:.3f}" stroke="{primitive["stroke"]}" stroke-width="{primitive["width"]:.3f}" stroke-linecap="round" />'
            )
        elif typ == "polyline":
            pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in primitive["points"])
            lines.append(
                f'<polyline points="{pts}" fill="none" stroke="{primitive["stroke"]}" stroke-width="{primitive["width"]:.3f}" stroke-linecap="round" stroke-linejoin="round" />'
            )
        elif typ == "marker":
            shape = primitive.get("shape")
            if shape == "circle":
                lines.append(
                    f'<circle cx="{primitive["x"]:.3f}" cy="{primitive["y"]:.3f}" r="{primitive["size"]:.3f}" fill="{primitive["fill"]}" stroke="{primitive["stroke"]}" stroke-width="{primitive["width"]:.3f}" />'
                )
            else:
                pts = " ".join(f"{x:.3f},{y:.3f}" for x, y in marker_polygon(shape, primitive["x"], primitive["y"], primitive["size"]))
                lines.append(
                    f'<polygon points="{pts}" fill="{primitive["fill"]}" stroke="{primitive["stroke"]}" stroke-width="{primitive["width"]:.3f}" />'
                )
        elif typ == "text":
            text = html.escape(str(primitive["text"]))
            lines.append(
                f'<text x="{primitive["x"]:.3f}" y="{primitive["y"]:.3f}" font-size="{primitive["size"]:.3f}" text-anchor="{primitive.get("anchor", "start")}" fill="{primitive["color"]}">{text}</text>'
            )
    lines.extend(["</g>", "</svg>", ""])
    return "\n".join(lines)


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def pdf_color_command(color: str, stroke: bool) -> str:
    r, g, b = (component / 255.0 for component in hex_to_rgb(color))
    operator = "RG" if stroke else "rg"
    return f"{r:.4f} {g:.4f} {b:.4f} {operator}"


def render_pdf(drawing: dict[str, Any]) -> bytes:
    width = float(drawing["width"])
    height = float(drawing["height"])
    commands: list[str] = ["1 1 1 rg 0 0 %.3f %.3f re f" % (width, height)]

    def y_pdf(y: float) -> float:
        return height - y

    for primitive in drawing["primitives"]:
        typ = primitive["type"]
        if typ == "rect":
            fill = primitive.get("fill")
            stroke = primitive.get("stroke")
            x = float(primitive["x"])
            y = y_pdf(float(primitive["y"]) + float(primitive["h"]))
            w = float(primitive["w"])
            h = float(primitive["h"])
            if fill and fill != "none":
                commands.append(f"{pdf_color_command(fill, False)} {x:.3f} {y:.3f} {w:.3f} {h:.3f} re f")
            if stroke and stroke != "none":
                commands.append(f"{pdf_color_command(stroke, True)} {x:.3f} {y:.3f} {w:.3f} {h:.3f} re S")
        elif typ == "line":
            commands.append(
                f"{pdf_color_command(primitive['stroke'], True)} {float(primitive['width']):.3f} w {float(primitive['x1']):.3f} {y_pdf(float(primitive['y1'])):.3f} m {float(primitive['x2']):.3f} {y_pdf(float(primitive['y2'])):.3f} l S"
            )
        elif typ == "polyline":
            pts = primitive["points"]
            if not pts:
                continue
            path = [f"{pts[0][0]:.3f} {y_pdf(pts[0][1]):.3f} m"]
            path.extend(f"{x:.3f} {y_pdf(y):.3f} l" for x, y in pts[1:])
            commands.append(f"{pdf_color_command(primitive['stroke'], True)} {float(primitive['width']):.3f} w " + " ".join(path) + " S")
        elif typ == "marker":
            x = float(primitive["x"])
            y = y_pdf(float(primitive["y"]))
            size = float(primitive["size"])
            fill = primitive["fill"]
            stroke = primitive["stroke"]
            if primitive.get("shape") == "circle":
                k = 0.5522847498 * size
                path = (
                    f"{x + size:.3f} {y:.3f} m "
                    f"{x + size:.3f} {y + k:.3f} {x + k:.3f} {y + size:.3f} {x:.3f} {y + size:.3f} c "
                    f"{x - k:.3f} {y + size:.3f} {x - size:.3f} {y + k:.3f} {x - size:.3f} {y:.3f} c "
                    f"{x - size:.3f} {y - k:.3f} {x - k:.3f} {y - size:.3f} {x:.3f} {y - size:.3f} c "
                    f"{x + k:.3f} {y - size:.3f} {x + size:.3f} {y - k:.3f} {x + size:.3f} {y:.3f} c h"
                )
            else:
                pts = [(px, y_pdf(py)) for px, py in marker_polygon(primitive.get("shape"), float(primitive["x"]), float(primitive["y"]), size)]
                path = f"{pts[0][0]:.3f} {pts[0][1]:.3f} m " + " ".join(f"{px:.3f} {py:.3f} l" for px, py in pts[1:]) + " h"
            commands.append(f"{pdf_color_command(fill, False)} {path} f")
            commands.append(f"{pdf_color_command(stroke, True)} {float(primitive['width']):.3f} w {path} S")
        elif typ == "text":
            text = pdf_escape(str(primitive["text"]))
            size = float(primitive["size"])
            x = text_x_for_anchor(float(primitive["x"]), str(primitive["text"]), size, primitive.get("anchor", "start"))
            y = y_pdf(float(primitive["y"]))
            commands.append(f"{pdf_color_command(primitive['color'], False)} BT /F1 {size:.3f} Tf {x:.3f} {y:.3f} Td ({text}) Tj ET")

    content = ("\n".join(commands) + "\n").encode("utf-8")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.3f} {height:.3f}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode("utf-8"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"endstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    offset = (y * width + x) * 3
    pixels[offset : offset + 3] = bytes(color)


def draw_thick_point(pixels: bytearray, width: int, height: int, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    r2 = radius * radius
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            if (xx - x) ** 2 + (yy - y) ** 2 <= r2:
                set_pixel(pixels, width, height, xx, yy, color)


def draw_png_line(
    pixels: bytearray,
    width: int,
    height: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    steps = max(1, int(max(abs(x2 - x1), abs(y2 - y1))))
    radius = max(1, thickness // 2)
    for idx in range(steps + 1):
        t = idx / steps
        x = int(round(x1 + (x2 - x1) * t))
        y = int(round(y1 + (y2 - y1) * t))
        draw_thick_point(pixels, width, height, x, y, radius, color)


def draw_png_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
) -> None:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            set_pixel(pixels, width, height, xx, yy, color)


def draw_png_text(
    pixels: bytearray,
    width: int,
    height: int,
    x: float,
    y: float,
    text: str,
    color: tuple[int, int, int],
    scale: int,
    anchor: str = "start",
) -> None:
    text = text.upper()
    char_width = 6 * scale
    total_width = len(text) * char_width
    start_x = int(round(x))
    if anchor == "middle":
        start_x -= total_width // 2
    elif anchor == "end":
        start_x -= total_width
    start_y = int(round(y - 7 * scale))
    for char_idx, char in enumerate(text):
        pattern = BITMAP_FONT_5X7.get(char, BITMAP_FONT_5X7[" "])
        base_x = start_x + char_idx * char_width
        for row_idx, row in enumerate(pattern):
            for col_idx, bit in enumerate(row):
                if bit != "1":
                    continue
                draw_png_rect(
                    pixels,
                    width,
                    height,
                    base_x + col_idx * scale,
                    start_y + row_idx * scale,
                    scale,
                    scale,
                    color,
                )


def write_png(path: Path, width: int, height: int, pixels: bytearray) -> None:
    raw_rows = bytearray()
    stride = width * 3
    for y in range(height):
        raw_rows.append(0)
        raw_rows.extend(pixels[y * stride : (y + 1) * stride])

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
    png.extend(chunk(b"IDAT", zlib.compress(bytes(raw_rows), level=9)))
    png.extend(chunk(b"IEND", b""))
    path.write_bytes(bytes(png))


def render_png(drawing: dict[str, Any], scale: int = 2) -> tuple[int, int, bytearray]:
    width = int(drawing["width"]) * scale
    height = int(drawing["height"]) * scale
    pixels = bytearray([255] * width * height * 3)
    for primitive in drawing["primitives"]:
        typ = primitive["type"]
        if typ == "rect":
            fill = primitive.get("fill")
            if fill and fill != "none":
                draw_png_rect(
                    pixels,
                    width,
                    height,
                    int(float(primitive["x"]) * scale),
                    int(float(primitive["y"]) * scale),
                    int(float(primitive["w"]) * scale),
                    int(float(primitive["h"]) * scale),
                    hex_to_rgb(fill),
                )
        elif typ == "line":
            draw_png_line(
                pixels,
                width,
                height,
                float(primitive["x1"]) * scale,
                float(primitive["y1"]) * scale,
                float(primitive["x2"]) * scale,
                float(primitive["y2"]) * scale,
                hex_to_rgb(primitive["stroke"]),
                max(1, int(float(primitive["width"]) * scale)),
            )
        elif typ == "polyline":
            pts = primitive["points"]
            for first, second in zip(pts, pts[1:]):
                draw_png_line(
                    pixels,
                    width,
                    height,
                    first[0] * scale,
                    first[1] * scale,
                    second[0] * scale,
                    second[1] * scale,
                    hex_to_rgb(primitive["stroke"]),
                    max(1, int(float(primitive["width"]) * scale)),
                )
        elif typ == "marker":
            color = hex_to_rgb(primitive["fill"])
            size = int(float(primitive["size"]) * scale)
            x = int(float(primitive["x"]) * scale)
            y = int(float(primitive["y"]) * scale)
            if primitive.get("shape") == "circle":
                draw_thick_point(pixels, width, height, x, y, size, color)
            else:
                draw_png_rect(pixels, width, height, x - size, y - size, size * 2, size * 2, color)
        elif typ == "text":
            scale_factor = max(1, int(round(float(primitive["size"]) / 7 * scale)))
            draw_png_text(
                pixels,
                width,
                height,
                float(primitive["x"]) * scale,
                float(primitive["y"]) * scale,
                str(primitive["text"]),
                hex_to_rgb(primitive["color"]),
                scale_factor,
                primitive.get("anchor", "start"),
            )
    return width, height, pixels


def write_figure_artifacts(output_dir: Path, summary: list[dict[str, Any]]) -> list[str]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    manifest: dict[str, Any] = {
        "generated_at": utc_now(),
        "source": "throughput_eval/reproduce_fig13.py",
        "figures": [],
    }
    for spec in FIGURE_SPECS:
        points = collect_plot_points(summary, str(spec["metric"]), float(spec["scale"]))
        if not any(methods for methods in points.values()):
            continue
        drawing = make_plot_primitives(points, str(spec["title"]), str(spec["ylabel"]))
        basename = str(spec["basename"])
        svg_path = figures_dir / f"{basename}.svg"
        pdf_path = figures_dir / f"{basename}.pdf"
        png_path = figures_dir / f"{basename}.png"
        svg_path.write_text(render_svg(drawing), encoding="utf-8")
        pdf_path.write_bytes(render_pdf(drawing))
        png_width, png_height, png_pixels = render_png(drawing, scale=2)
        write_png(png_path, png_width, png_height, png_pixels)
        figure_files = [
            str(svg_path.relative_to(output_dir)),
            str(pdf_path.relative_to(output_dir)),
            str(png_path.relative_to(output_dir)),
        ]
        manifest["figures"].append(
            {
                "name": spec["name"],
                "metric": spec["metric"],
                "unit": spec["ylabel"],
                "files": figure_files,
            }
        )
        written.extend(figure_files)
    if written:
        write_json(figures_dir / "figure_manifest.json", manifest)
        written.append(str((figures_dir / "figure_manifest.json").relative_to(output_dir)))
    return written


def peak_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    peaks: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary:
        throughput = row.get("mean_decode_throughput_tokens_s")
        if throughput is None:
            continue
        key = (row.get("method"), row.get("context_len"))
        if key not in peaks or throughput > peaks[key].get("peak_decode_throughput_tokens_s", 0):
            peaks[key] = {
                "method": row.get("method"),
                "context_len": row.get("context_len"),
                "best_batch_size": row.get("batch_size"),
                "peak_decode_throughput_tokens_s": throughput,
                "mean_decode_latency_s": row.get("mean_decode_latency_s"),
                "mean_e2e_latency_s": row.get("mean_e2e_latency_s"),
            }
    return sorted(peaks.values(), key=lambda row: (int(row["context_len"]), str(row["method"])))


def speedup_rows(peaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_context: dict[Any, dict[str, dict[str, Any]]] = {}
    for row in peaks:
        by_context.setdefault(row["context_len"], {})[row["method"]] = row

    rows: list[dict[str, Any]] = []
    for context_len, methods in sorted(by_context.items(), key=lambda item: int(item[0])):
        full = methods.get("Full_Flash_Attn")
        if not full or not full.get("peak_decode_throughput_tokens_s"):
            continue
        for method, row in sorted(methods.items()):
            if method == "Full_Flash_Attn":
                continue
            rows.append(
                {
                    "context_len": context_len,
                    "method": method,
                    "method_peak_tokens_s": row.get("peak_decode_throughput_tokens_s"),
                    "full_attention_peak_tokens_s": full.get("peak_decode_throughput_tokens_s"),
                    "speedup_vs_full_attention": (
                        row.get("peak_decode_throughput_tokens_s") / full.get("peak_decode_throughput_tokens_s")
                    ),
                }
            )
    return rows


def write_report(
    path: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    results: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    figure_paths: list[str] | None = None,
) -> None:
    _ = environment, results
    peaks = peak_rows(summary)
    speedups = speedup_rows(peaks)
    figure_paths = figure_paths or []

    lines = [
        "# RetroInfer Figure 13 single-GPU reproduction",
        "",
        f"- Generated at: `{config['generated_at']}`",
        f"- Suite: `{config['suite']}`",
        f"- Model: `{config['model_name']}`",
        f"- Task: `{config['task_name']}`",
        f"- Decode tokens per request: `{config['gen_len']}`",
        f"- Retrieval budget / estimation budget: `{config['retrieval_budget']}` / `{config['estimation_budget']}`",
        f"- GPU visibility: `CUDA_VISIBLE_DEVICES={config['cuda_visible_devices']}`",
        "",
    ]
    if figure_paths:
        lines.extend(
            [
                "## Figures",
                "",
                *[f"- `{path}`" for path in figure_paths],
                "",
            ]
        )

    if summary:
        lines.extend(
            [
                "## Measured batch summary",
                "",
                markdown_table(
                    summary,
                    [
                        "method",
                        "context_len",
                        "batch_size",
                        "passed_runs",
                        "mean_decode_throughput_tokens_s",
                        "mean_peak_process_gpu_memory_mib",
                    ],
                ),
                "",
                "## Peak throughput by method and context",
                "",
                markdown_table(
                    peaks,
                    [
                        "context_len",
                        "method",
                        "best_batch_size",
                        "peak_decode_throughput_tokens_s",
                    ],
                ),
                "",
                "## Peak throughput ratios",
                "",
                markdown_table(
                    speedups,
                    [
                        "context_len",
                        "method",
                        "method_peak_tokens_s",
                        "full_attention_peak_tokens_s",
                        "speedup_vs_full_attention",
                    ],
                ),
                "",
            ]
        )
    else:
        lines.extend(["## Measured batch summary", "", "_No successful measured rows._", ""])

    lines.extend(
        [
            "## Artifact manifest",
            "",
            "- `config.json`: exact suite configuration and generated command plan.",
            "- `commands.jsonl`: command for each planned run.",
            "- `environment.json`: hardware, git, Python, package, CUDA, and NUMA observations.",
            "- `raw_logs/*.txt`: raw unedited stdout/stderr for every run.",
            "- `memory_samples/*.jsonl`: process-level GPU memory samples captured during each measured run when memory sampling is enabled.",
            "- `results.jsonl` and `results.csv`: per-run parsed records.",
            "- `summary.csv`: grouped averages computed from successful measured runs.",
            "- `figures/fig13_single_gpu_throughput.{pdf,svg,png}` and `figures/fig13_single_gpu_gpu_memory.{pdf,svg,png}`: script-generated paper-ready throughput and GPU-memory plots.",
            "- `figures/figure_manifest.json`: plot provenance and file manifest.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_config(output_dir: Path, args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    commands = [{**run, "command": command_for_run(run, args)} for run in runs]
    config = {
        "generated_at": utc_now(),
        "suite": args.suite,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "gen_len": args.gen_len,
        "rounds": args.rounds,
        "methods": args.methods,
        "cuda_visible_devices": args.cuda_visible_devices,
        "use_numactl": args.use_numactl,
        "use_cuda_graph": args.use_cuda_graph,
        "retrieval_budget": args.retrieval_budget,
        "estimation_budget": args.estimation_budget,
        "cache_ratio": args.cache_ratio,
        "dry_run": args.dry_run,
        "gpu_memory_measurement": {
            "enabled": args.gpu_memory_sampling,
            "primary_metric": "peak_process_gpu_memory_mib",
            "unit": "MiB",
            "sample_interval_s": args.gpu_memory_sample_interval_s,
            "sampling_source": "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "scope": "sum of process-level used_memory for the launched runner PID and its descendants during the child-process lifetime",
            "torch_allocator_metrics": [
                "torch_cuda_peak_allocated_mib",
                "torch_cuda_peak_reserved_mib",
            ],
            "synchronization": "native runner calls torch.cuda.synchronize before resetting and before reading allocator peaks",
            "require_idle_gpu": args.require_idle_gpu,
        },
        "paper_source": {
            "arxiv": "2505.02922",
            "figure_file": "figures/throughput_different_length.pdf",
            "caption": "Decoding throughput at different context lengths (Llama3-8B-1048K).",
            "context_lengths": [30000, 60000, 120000, 1024000],
        },
        "runs": commands,
    }
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "commands.jsonl", commands)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["figure13"], default="figure13")
    parser.add_argument("--methods", type=parse_method_list, default=parse_method_list("Full_Flash_Attn,RetroInfer"))
    parser.add_argument("--context-lens", dest="context_lens", type=parse_int_list)
    parser.add_argument("--batch-sizes", dest="batch_sizes", type=parse_int_list)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--gen-len", type=int, default=100)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--prefill-bsz", type=int, default=1)
    parser.add_argument("--retrieval-budget", type=float, default=0.018)
    parser.add_argument("--estimation-budget", type=float, default=0.232)
    parser.add_argument("--cache-ratio", type=float, default=0.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--gpu-memory-sample-interval-s", type=float, default=0.5)
    parser.add_argument("--no-gpu-memory-sampling", dest="gpu_memory_sampling", action="store_false")
    parser.add_argument("--require-idle-gpu", action="store_true")
    parser.add_argument("--use-numactl", action="store_true")
    parser.add_argument("--no-cuda-graph", dest="use_cuda_graph", action="store_false")
    parser.set_defaults(use_cuda_graph=True, gpu_memory_sampling=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.max_runs is not None and args.max_runs < 1:
        raise SystemExit("--max-runs must be >= 1")
    if args.gpu_memory_sample_interval_s <= 0:
        raise SystemExit("--gpu-memory-sample-interval-s must be > 0")

    output_dir = args.output_dir
    if output_dir is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "research" / f"fig13_throughput_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = build_runs(args)
    config = write_config(output_dir, args, runs)
    environment = collect_environment()
    write_json(output_dir / "environment.json", environment)

    results: list[dict[str, Any]] = []
    for run in runs:
        result = run_one(run, args, output_dir)
        results.append(result)
        write_jsonl(output_dir / "results.jsonl", results)
        write_csv(output_dir / "results.csv", results, RUN_FIELDS)
        if args.stop_on_error and result["status"] != "passed":
            break

    summary = summarize_results(results)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    figure_paths = write_figure_artifacts(output_dir, summary)
    write_report(output_dir / "REPORT.md", config, environment, results, summary, figure_paths)
    print(f"Wrote Figure 13 reproduction artifacts to {output_dir}")
    failed = [result for result in results if result.get("status") != "passed"]
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
