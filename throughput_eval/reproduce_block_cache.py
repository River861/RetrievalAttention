#!/usr/bin/env python3
"""Run a single-A100 RetroInfer block-cache capacity sensitivity matrix."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from throughput_eval.reproduce_fig13 import (
        DEFAULT_MODEL,
        DEFAULT_TASK,
        PROJECT_ROOT,
        THROUGHPUT_DIR,
        clean_text,
        collect_environment,
        markdown_table,
        parse_error_summary,
        parse_int_list,
        parse_log_text,
        render_pdf,
        render_png,
        render_svg,
        sample_gpu_memory,
        summarize_memory_samples,
        utc_now,
        visible_gpu_compute_apps,
        write_json,
        write_jsonl,
        write_memory_event,
        write_png,
    )
except ModuleNotFoundError:
    from reproduce_fig13 import (
        DEFAULT_MODEL,
        DEFAULT_TASK,
        PROJECT_ROOT,
        THROUGHPUT_DIR,
        clean_text,
        collect_environment,
        markdown_table,
        parse_error_summary,
        parse_int_list,
        parse_log_text,
        render_pdf,
        render_png,
        render_svg,
        sample_gpu_memory,
        summarize_memory_samples,
        utc_now,
        visible_gpu_compute_apps,
        write_json,
        write_jsonl,
        write_memory_event,
        write_png,
    )


MEMORY_JSON_RE = re.compile(r"BLOCK_CACHE_MEMORY_JSON=(\{.*\})")
RETURNCODE_RE = re.compile(r"^# returncode:\s*(.+)$", re.MULTILINE)
CACHE_PAGES_RE = re.compile(r"^Cache pages:\s*(?P<cache_pages>\d+),\s*Buffer pages:\s*(?P<buffer_pages>\d+)", re.MULTILINE)
INITIAL_CENTROIDS_RE = re.compile(
    r"^Initial n_centroids:\s*(?P<n_centroids>\d+),\s*nprobe:\s*(?P<nprobe>\d+),",
    re.MULTILINE,
)
DEFAULT_CACHE_RATIOS = "0.005,0.025,0.05,0.10,0.0"
DEFAULT_CONTEXT_LENS = "30000,120000"
DEFAULT_BATCH_SIZES = "4"
DEFAULT_BASELINE_CACHE_RATIO = 0.05
DEFAULT_SEED = 2025
FLOAT_TOLERANCE = 1e-6
A100_80GB_TOTAL_MIB = 80.0 * 1024.0
ARTIFACT_SCHEMA_VERSION = 2

BLOCK_CACHE_FIELDS = [
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
    "block_cache_total_gib",
]

DERIVED_MEMORY_FIELDS = [
    "block_cache_total_mib",
    "block_cache_percent_of_a100_80gb",
    "block_cache_percent_of_peak_process_memory",
    "non_block_cache_peak_process_memory_mib",
]

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
    "seed",
    "cache_ratio",
    "cache_role",
    "status",
    "failure_class",
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
    *BLOCK_CACHE_FIELDS,
    "block_cache_total_mib",
    "block_cache_percent_of_a100_80gb",
    "peak_process_gpu_memory_mib",
    "block_cache_percent_of_peak_process_memory",
    "non_block_cache_peak_process_memory_mib",
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
    "seed",
    "cache_ratio",
    "cache_role",
    "total_runs",
    "passed_runs",
    "failed_runs",
    *BLOCK_CACHE_FIELDS,
    "block_cache_total_mib",
    "block_cache_percent_of_a100_80gb",
    "round_decode_throughput_tokens_s",
    "round_peak_process_gpu_memory_mib",
    "mean_decode_throughput_tokens_s",
    "stdev_decode_throughput_tokens_s",
    "variance_decode_throughput_tokens_s",
    "mean_decode_latency_s",
    "mean_e2e_latency_s",
    "mean_torch_cuda_peak_allocated_mib",
    "mean_torch_cuda_peak_reserved_mib",
    "mean_peak_process_gpu_memory_mib",
    "max_peak_process_gpu_memory_mib",
    "mean_block_cache_percent_of_peak_process_memory",
    "mean_non_block_cache_peak_process_memory_mib",
]

DELTA_FIELDS = [
    "suite",
    "method",
    "model_name",
    "task_name",
    "context_len",
    "batch_size",
    "gen_len",
    "seed",
    "baseline_cache_ratio",
    "cache_ratio",
    "cache_role",
    "baseline_block_cache_total_gib",
    "block_cache_total_gib",
    "block_cache_total_gib_delta",
    "baseline_mean_peak_process_gpu_memory_mib",
    "mean_peak_process_gpu_memory_mib",
    "peak_process_gpu_memory_delta_mib",
    "peak_process_gpu_memory_delta_pct",
    "baseline_block_cache_percent_of_peak_process_memory",
    "block_cache_percent_of_peak_process_memory",
    "block_cache_percent_of_peak_process_memory_delta",
    "baseline_non_block_cache_peak_process_memory_mib",
    "non_block_cache_peak_process_memory_mib",
    "non_block_cache_peak_process_memory_delta_mib",
    "baseline_mean_decode_throughput_tokens_s",
    "mean_decode_throughput_tokens_s",
    "decode_throughput_delta_tokens_s",
    "decode_throughput_delta_pct",
    "decode_throughput_ratio_vs_baseline",
]

PER_ROUND_FIELDS = [
    "context_len",
    "batch_size",
    "cache_ratio",
    "cache_role",
    "round",
    "block_cache_total_gib",
    "block_cache_total_mib",
    "block_cache_percent_of_a100_80gb",
    "block_cache_percent_of_peak_process_memory",
    "non_block_cache_peak_process_memory_mib",
    "decode_throughput_tokens_s",
    "decode_latency_s",
    "peak_process_gpu_memory_mib",
    "torch_cuda_peak_allocated_mib",
    "torch_cuda_peak_reserved_mib",
    "status",
    "failure_class",
    "log_path",
]

CURVE_POINT_FIELDS = [
    "context_len",
    "batch_size",
    "cache_ratio",
    "cache_role",
    "passed_runs",
    "failed_runs",
    "block_cache_total_gib",
    "block_cache_percent_of_a100_80gb",
    "mean_block_cache_percent_of_peak_process_memory",
    "mean_peak_process_gpu_memory_mib",
    "mean_decode_throughput_tokens_s",
]

CURVE_SPECS = [
    {
        "name": "block_cache_peak_share",
        "metric": "mean_block_cache_percent_of_peak_process_memory",
        "scale": 1.0,
        "title": "Block-cache share of peak process memory",
        "ylabel": "Block cache / process peak (%)",
    },
    {
        "name": "peak_process_gpu_memory",
        "metric": "mean_peak_process_gpu_memory_mib",
        "scale": 1.0 / 1024.0,
        "title": "Peak process GPU memory",
        "ylabel": "Peak process GPU memory (GiB)",
    },
    {
        "name": "decode_throughput",
        "metric": "mean_decode_throughput_tokens_s",
        "scale": 1.0,
        "title": "Decode throughput",
        "ylabel": "Decode throughput (tokens/s)",
    },
]

CURVE_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#999999",
]


def parse_float_list(value: str | None) -> list[float] | None:
    if value is None or value == "":
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_context_batch_groups(value: str | None) -> list[tuple[int, int]] | None:
    if value is None or value == "":
        return None
    groups: list[tuple[int, int]] = []
    for part in value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[x:]\s*(\d+)", token)
        if match is None:
            raise argparse.ArgumentTypeError(
                "--context-batch-groups entries must look like 120000x1 or 120000:1"
            )
        groups.append((int(match.group(1)), int(match.group(2))))
    return groups


def ratio_token(cache_ratio: float) -> str:
    text = f"{cache_ratio:g}"
    return text.replace("-", "m").replace(".", "p")


def ratios_equal(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) <= FLOAT_TOLERANCE
    except (TypeError, ValueError):
        return False


def cache_role(cache_ratio: float, baseline_cache_ratio: float) -> str:
    if ratios_equal(cache_ratio, baseline_cache_ratio):
        return "paper_default_5pct"
    if cache_ratio == 0.0:
        return "code_fallback_default"
    if cache_ratio < baseline_cache_ratio:
        return "smaller_than_default"
    return "larger_than_default"


def number_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_retroinfer_config_log_text(text: str) -> dict[str, Any]:
    for line in clean_text(text).splitlines():
        stripped = line.strip()
        if not stripped.startswith("{") or "pages_per_cluster" not in stripped or "n_centroids" not in stripped:
            continue
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_block_cache_log_hints(text: str) -> dict[str, Any]:
    cleaned = clean_text(text)
    hints: dict[str, Any] = {}

    config = parse_retroinfer_config_log_text(cleaned)
    config_n_centroids = int_or_none(config.get("n_centroids"))
    config_pages_per_cluster = int_or_none(config.get("pages_per_cluster"))
    config_cache_ratio = number_or_none(config.get("cache_ratio"))
    if config_n_centroids is not None:
        hints["block_cache_index_clusters"] = config_n_centroids
    if config_pages_per_cluster is not None:
        hints["block_cache_pages_per_cluster"] = config_pages_per_cluster
    if config_cache_ratio is not None:
        hints["block_cache_ratio"] = config_cache_ratio
        hints["block_cache_source"] = "cache_ratio" if config_cache_ratio > 0.0 else "retrieval_budget_default"

    initial_match = INITIAL_CENTROIDS_RE.search(cleaned)
    if initial_match:
        hints["block_cache_index_clusters"] = int(initial_match.group("n_centroids"))
        hints["block_cache_retrieval_clusters"] = int(initial_match.group("nprobe"))

    cache_pages_match = CACHE_PAGES_RE.search(cleaned)
    if cache_pages_match:
        cache_pages = int(cache_pages_match.group("cache_pages"))
        hints["block_cache_pages_per_layer"] = cache_pages
        pages_per_cluster = int_or_none(hints.get("block_cache_pages_per_cluster"))
        if pages_per_cluster:
            hints["block_cache_clusters_per_layer"] = cache_pages // pages_per_cluster

    return hints


def parse_block_cache_log_text(text: str) -> dict[str, Any]:
    parsed = parse_log_text(text)
    cleaned = clean_text(text)
    for match in MEMORY_JSON_RE.finditer(cleaned):
        try:
            parsed.update(json.loads(match.group(1)))
        except json.JSONDecodeError:
            pass
    for key, value in parse_block_cache_log_hints(cleaned).items():
        parsed.setdefault(key, value)
    if "error_summary" not in parsed:
        parsed["error_summary"] = parse_error_summary(cleaned)
    return parsed


def parse_returncode_from_log(text: str) -> int | None:
    match = RETURNCODE_RE.search(clean_text(text))
    if not match:
        return None
    value = match.group(1).strip()
    if value in {"None", ""}:
        return None
    return int(value)


def enrich_block_cache_fields(row: dict[str, Any]) -> None:
    total_bytes = number_or_none(row.get("block_cache_total_bytes"))
    if total_bytes is not None:
        total_mib = total_bytes / (1024**2)
        row["block_cache_total_mib"] = total_mib
        row["block_cache_total_gib"] = total_bytes / (1024**3)
        row["block_cache_percent_of_a100_80gb"] = 100.0 * total_mib / A100_80GB_TOTAL_MIB

    total_mib = number_or_none(row.get("block_cache_total_mib"))
    peak_mib = number_or_none(row.get("peak_process_gpu_memory_mib"))
    if total_mib is not None and peak_mib is not None:
        row["block_cache_percent_of_peak_process_memory"] = 100.0 * total_mib / peak_mib if peak_mib > 0 else None
        row["non_block_cache_peak_process_memory_mib"] = peak_mib - total_mib


def infer_block_cache_allocation_templates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for row in results:
        total_bytes = int_or_none(row.get("block_cache_total_bytes"))
        pages_per_layer = int_or_none(row.get("block_cache_pages_per_layer"))
        batch_size = int_or_none(row.get("batch_size"))
        layer_count = int_or_none(row.get("block_cache_layer_count"))
        page_size_vectors = int_or_none(row.get("block_cache_page_size_vectors"))
        dtype_bytes = int_or_none(row.get("block_cache_dtype_bytes"))
        if (
            total_bytes is None
            or pages_per_layer is None
            or batch_size is None
            or layer_count is None
            or page_size_vectors is None
            or dtype_bytes is None
            or pages_per_layer <= 0
            or batch_size <= 0
            or layer_count <= 0
        ):
            continue
        templates.append(
            {
                "cache_ratio": number_or_none(row.get("cache_ratio")),
                "bytes_per_page_per_batch": total_bytes / (pages_per_layer * batch_size),
                "block_cache_layer_count": layer_count,
                "block_cache_page_size_vectors": page_size_vectors,
                "block_cache_dtype_bytes": dtype_bytes,
                "block_cache_pages_per_cluster": int_or_none(row.get("block_cache_pages_per_cluster")),
            }
        )
    return templates


def select_block_cache_allocation_template(templates: list[dict[str, Any]], cache_ratio_value: float | None) -> dict[str, Any]:
    if cache_ratio_value is not None:
        for template in templates:
            if ratios_equal(template.get("cache_ratio"), cache_ratio_value):
                return template
    return templates[0] if templates else {}


def fill_missing_block_cache_capacity_fields(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    templates = infer_block_cache_allocation_templates(results)
    for row in results:
        cache_ratio_value = number_or_none(row.get("cache_ratio"))
        template = select_block_cache_allocation_template(templates, cache_ratio_value)
        bytes_per_page_per_batch = number_or_none(template.get("bytes_per_page_per_batch"))
        if not row.get("block_cache_source") and cache_ratio_value is not None:
            row["block_cache_source"] = "cache_ratio" if cache_ratio_value > 0.0 else "retrieval_budget_default"
        if row.get("block_cache_ratio") in (None, "") and cache_ratio_value is not None:
            row["block_cache_ratio"] = cache_ratio_value

        pages_per_layer = int_or_none(row.get("block_cache_pages_per_layer"))
        pages_per_cluster = int_or_none(row.get("block_cache_pages_per_cluster")) or int_or_none(
            template.get("block_cache_pages_per_cluster")
        )
        if pages_per_layer is not None and pages_per_cluster:
            row["block_cache_pages_per_cluster"] = pages_per_cluster
            if row.get("block_cache_clusters_per_layer") in (None, ""):
                row["block_cache_clusters_per_layer"] = pages_per_layer // pages_per_cluster

        batch_size = int_or_none(row.get("batch_size"))
        if (
            row.get("block_cache_total_bytes") in (None, "")
            and bytes_per_page_per_batch is not None
            and pages_per_layer is not None
            and batch_size is not None
            and batch_size > 0
        ):
            layer_count = int_or_none(template.get("block_cache_layer_count"))
            page_size_vectors = int_or_none(template.get("block_cache_page_size_vectors"))
            dtype_bytes = int_or_none(template.get("block_cache_dtype_bytes"))
            if layer_count is not None and page_size_vectors is not None and dtype_bytes is not None:
                total_bytes = int(round(bytes_per_page_per_batch * pages_per_layer * batch_size))
                vectors_per_layer = pages_per_layer * page_size_vectors
                row["block_cache_page_size_vectors"] = page_size_vectors
                row["block_cache_vectors_per_layer"] = vectors_per_layer
                row["block_cache_layer_count"] = layer_count
                row["block_cache_total_pages"] = pages_per_layer * layer_count
                row["block_cache_total_vectors"] = vectors_per_layer * layer_count
                row["block_cache_dtype_bytes"] = dtype_bytes
                row["block_cache_bytes_per_layer"] = int(round(total_bytes / layer_count))
                row["block_cache_total_bytes"] = total_bytes
        enrich_block_cache_fields(row)
    return results


def classify_run_failure(status: str, returncode: Any, error_summary: Any) -> str:
    if status == "passed":
        return "success"
    code = int_or_none(returncode)
    text = str(error_summary or "").lower()
    if code == 124 or "timeout" in text:
        return "timeout"
    if (
        "out of memory" in text
        or "cuda oom" in text
        or "cannot allocate memory" in text
        or "cublas_status_alloc_failed" in text
    ):
        return "oom"
    if "unsupported" in text or "not supported" in text:
        return "unsupported"
    if code == 126:
        return "prelaunch_failed"
    if code == 127:
        return "launch_failed"
    if code == 0:
        return "missing_metrics"
    return "failed"


def command_for_run(run: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "test.py",
        "--model_name",
        run["model_name"],
        "--attn_type",
        "RetroInfer",
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
        str(run["cache_ratio"]),
    ]
    if args.use_cuda_graph:
        command.append("--use_cuda_graph")
    if args.use_numactl and shutil.which("numactl") is not None:
        return ["numactl", "--cpunodebind=0", "--membind=0"] + command
    return command


def make_run(
    args: argparse.Namespace,
    context_len: int,
    batch_size: int,
    cache_ratio_value: float,
    round_idx: int,
) -> dict[str, Any]:
    run_id = (
        f"{args.suite}_retroinfer_ctx{context_len}_bsz{batch_size}"
        f"_cr{ratio_token(cache_ratio_value)}_r{round_idx}"
    )
    return {
        "run_id": run_id,
        "suite": args.suite,
        "method": "RetroInfer",
        "model_name": args.model_name,
        "task_name": args.task_name,
        "context_len": context_len,
        "batch_size": batch_size,
        "gen_len": args.gen_len,
        "round": round_idx,
        "seed": args.seed,
        "cache_ratio": cache_ratio_value,
        "cache_role": cache_role(cache_ratio_value, args.baseline_cache_ratio),
    }


def build_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    cache_ratios = args.cache_ratios or parse_float_list(DEFAULT_CACHE_RATIOS) or []
    runs: list[dict[str, Any]] = []
    for context_len, batch_size in selected_context_batch_groups(args):
        for cache_ratio_value in cache_ratios:
            for round_idx in range(1, args.rounds + 1):
                runs.append(make_run(args, context_len, batch_size, cache_ratio_value, round_idx))
    return runs


def selected_context_batch_groups(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.context_batch_groups:
        return list(args.context_batch_groups)
    context_lens = args.context_lens or parse_int_list(DEFAULT_CONTEXT_LENS) or []
    batch_sizes = args.batch_sizes or parse_int_list(DEFAULT_BATCH_SIZES) or []
    return [(context_len, batch_size) for context_len in context_lens for batch_size in batch_sizes]


def context_batch_group_dicts(groups: list[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"context_len": context_len, "batch_size": batch_size} for context_len, batch_size in groups]


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
            except OSError as exc:
                log.write(f"\n# failed to launch command: {exc}\n")
                returncode = 127
            except RuntimeError as exc:
                log.write(f"\n# failed before launch: {exc}\n")
                returncode = 126
        finished = utc_now()
        memory_summary = summarize_memory_samples(sample_path, output_dir, args.gpu_memory_sample_interval_s)
        if memory_summary:
            log.write("BLOCK_CACHE_MEMORY_JSON=" + json.dumps(memory_summary, sort_keys=True) + "\n")
        log.write(f"\n# finished_at: {finished}\n")
        log.write(f"# returncode: {returncode}\n")

    duration_s = time.time() - start
    parsed = parse_block_cache_log_text(log_path.read_text(encoding="utf-8", errors="replace"))
    enrich_block_cache_fields(parsed)
    success_metric = "decode_throughput_tokens_s" in parsed
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
    result["failure_class"] = classify_run_failure(status, returncode, result.get("error_summary"))
    return result


def _mean(items: list[dict[str, Any]], field: str) -> float | None:
    values = [number_or_none(item.get(field)) for item in items]
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _first_present(items: list[dict[str, Any]], field: str) -> Any:
    for item in items:
        value = item.get(field)
        if value is not None and value != "":
            return value
    return None


def summarize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("suite"),
            result.get("method"),
            result.get("model_name"),
            result.get("task_name"),
            int_or_none(result.get("context_len")),
            int_or_none(result.get("batch_size")),
            int_or_none(result.get("gen_len")),
            int_or_none(result.get("seed")),
            float(result.get("cache_ratio")),
            result.get("cache_role"),
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0])):
        passed = [item for item in items if item.get("status") == "passed"]
        memory_items = passed if passed else [item for item in items if number_or_none(item.get("peak_process_gpu_memory_mib")) is not None]
        throughputs = [
            float(item["decode_throughput_tokens_s"])
            for item in passed
            if number_or_none(item.get("decode_throughput_tokens_s")) is not None
        ]
        peak_memories = [
            float(item["peak_process_gpu_memory_mib"])
            for item in memory_items
            if number_or_none(item.get("peak_process_gpu_memory_mib")) is not None
        ]
        row: dict[str, Any] = {
            "suite": key[0],
            "method": key[1],
            "model_name": key[2],
            "task_name": key[3],
            "context_len": key[4],
            "batch_size": key[5],
            "gen_len": key[6],
            "seed": key[7],
            "cache_ratio": key[8],
            "cache_role": key[9],
            "total_runs": len(items),
            "passed_runs": len(passed),
            "failed_runs": len(items) - len(passed),
            "round_decode_throughput_tokens_s": ";".join(
                f"r{item.get('round')}={float(item['decode_throughput_tokens_s']):.6f}"
                for item in sorted(passed, key=lambda row: int(row.get("round", 0)))
                if number_or_none(item.get("decode_throughput_tokens_s")) is not None
            ),
            "round_peak_process_gpu_memory_mib": ";".join(
                f"r{item.get('round')}={float(item['peak_process_gpu_memory_mib']):.1f}"
                for item in sorted(memory_items, key=lambda row: int(row.get("round", 0)))
                if number_or_none(item.get("peak_process_gpu_memory_mib")) is not None
            ),
            "mean_decode_throughput_tokens_s": statistics.mean(throughputs) if throughputs else None,
            "stdev_decode_throughput_tokens_s": statistics.stdev(throughputs) if len(throughputs) > 1 else None,
            "variance_decode_throughput_tokens_s": statistics.variance(throughputs) if len(throughputs) > 1 else None,
            "mean_decode_latency_s": _mean(passed, "decode_latency_s"),
            "mean_e2e_latency_s": _mean(passed, "e2e_latency_s") or _mean(passed, "avg_e2e_latency_s"),
            "mean_torch_cuda_peak_allocated_mib": _mean(passed, "torch_cuda_peak_allocated_mib"),
            "mean_torch_cuda_peak_reserved_mib": _mean(passed, "torch_cuda_peak_reserved_mib"),
            "mean_peak_process_gpu_memory_mib": statistics.mean(peak_memories) if peak_memories else None,
            "max_peak_process_gpu_memory_mib": max(peak_memories) if peak_memories else None,
            "mean_block_cache_percent_of_peak_process_memory": _mean(memory_items, "block_cache_percent_of_peak_process_memory"),
            "mean_non_block_cache_peak_process_memory_mib": _mean(memory_items, "non_block_cache_peak_process_memory_mib"),
        }
        for field in BLOCK_CACHE_FIELDS:
            row[field] = _first_present(items, field)
        for field in ("block_cache_total_mib", "block_cache_percent_of_a100_80gb"):
            row[field] = _first_present(items, field)
        rows.append(row)
    return rows


def compute_deltas(summary: list[dict[str, Any]], baseline_cache_ratio: float) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in summary:
        key = (
            row.get("suite"),
            row.get("method"),
            row.get("model_name"),
            row.get("task_name"),
            row.get("context_len"),
            row.get("batch_size"),
            row.get("gen_len"),
            row.get("seed"),
        )
        if ratios_equal(row.get("cache_ratio"), baseline_cache_ratio):
            by_key[key] = row

    deltas: list[dict[str, Any]] = []
    for row in summary:
        key = (
            row.get("suite"),
            row.get("method"),
            row.get("model_name"),
            row.get("task_name"),
            row.get("context_len"),
            row.get("batch_size"),
            row.get("gen_len"),
            row.get("seed"),
        )
        baseline = by_key.get(key)
        if baseline is None:
            continue
        row_memory = number_or_none(row.get("mean_peak_process_gpu_memory_mib"))
        base_memory = number_or_none(baseline.get("mean_peak_process_gpu_memory_mib"))
        row_throughput = number_or_none(row.get("mean_decode_throughput_tokens_s"))
        base_throughput = number_or_none(baseline.get("mean_decode_throughput_tokens_s"))
        row_cache_gib = number_or_none(row.get("block_cache_total_gib"))
        base_cache_gib = number_or_none(baseline.get("block_cache_total_gib"))
        row_cache_peak_pct = number_or_none(row.get("mean_block_cache_percent_of_peak_process_memory"))
        base_cache_peak_pct = number_or_none(baseline.get("mean_block_cache_percent_of_peak_process_memory"))
        row_non_cache_mib = number_or_none(row.get("mean_non_block_cache_peak_process_memory_mib"))
        base_non_cache_mib = number_or_none(baseline.get("mean_non_block_cache_peak_process_memory_mib"))
        memory_delta = row_memory - base_memory if row_memory is not None and base_memory is not None else None
        throughput_delta = (
            row_throughput - base_throughput
            if row_throughput is not None and base_throughput is not None
            else None
        )
        deltas.append(
            {
                "suite": row.get("suite"),
                "method": row.get("method"),
                "model_name": row.get("model_name"),
                "task_name": row.get("task_name"),
                "context_len": row.get("context_len"),
                "batch_size": row.get("batch_size"),
                "gen_len": row.get("gen_len"),
                "seed": row.get("seed"),
                "baseline_cache_ratio": baseline_cache_ratio,
                "cache_ratio": row.get("cache_ratio"),
                "cache_role": row.get("cache_role"),
                "baseline_block_cache_total_gib": base_cache_gib,
                "block_cache_total_gib": row_cache_gib,
                "block_cache_total_gib_delta": (
                    row_cache_gib - base_cache_gib
                    if row_cache_gib is not None and base_cache_gib is not None
                    else None
                ),
                "baseline_mean_peak_process_gpu_memory_mib": base_memory,
                "mean_peak_process_gpu_memory_mib": row_memory,
                "peak_process_gpu_memory_delta_mib": memory_delta,
                "peak_process_gpu_memory_delta_pct": (
                    100.0 * memory_delta / base_memory
                    if memory_delta is not None and base_memory
                    else None
                ),
                "baseline_block_cache_percent_of_peak_process_memory": base_cache_peak_pct,
                "block_cache_percent_of_peak_process_memory": row_cache_peak_pct,
                "block_cache_percent_of_peak_process_memory_delta": (
                    row_cache_peak_pct - base_cache_peak_pct
                    if row_cache_peak_pct is not None and base_cache_peak_pct is not None
                    else None
                ),
                "baseline_non_block_cache_peak_process_memory_mib": base_non_cache_mib,
                "non_block_cache_peak_process_memory_mib": row_non_cache_mib,
                "non_block_cache_peak_process_memory_delta_mib": (
                    row_non_cache_mib - base_non_cache_mib
                    if row_non_cache_mib is not None and base_non_cache_mib is not None
                    else None
                ),
                "baseline_mean_decode_throughput_tokens_s": base_throughput,
                "mean_decode_throughput_tokens_s": row_throughput,
                "decode_throughput_delta_tokens_s": throughput_delta,
                "decode_throughput_delta_pct": (
                    100.0 * throughput_delta / base_throughput
                    if throughput_delta is not None and base_throughput
                    else None
                ),
                "decode_throughput_ratio_vs_baseline": (
                    row_throughput / base_throughput
                    if row_throughput is not None and base_throughput
                    else None
                ),
            }
        )
    return sorted(
        deltas,
        key=lambda row: (
            int(row.get("context_len") or 0),
            int(row.get("batch_size") or 0),
            float(row.get("cache_ratio") or 0),
        ),
    )


def per_round_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sorted(
        results,
        key=lambda row: (
            int(row.get("context_len") or 0),
            int(row.get("batch_size") or 0),
            float(row.get("cache_ratio") or 0),
            int(row.get("round") or 0),
        ),
    ):
        rows.append({field: result.get(field) for field in PER_ROUND_FIELDS})
    return rows


def paper_code_mapping() -> list[dict[str, str]]:
    return [
        {
            "paper_concept": "Tripartite attention zones",
            "paper_evidence": (
                "Paper Section 1/4.2 partitions tokens into steady, retrieval, and estimation zones; "
                "steady and retrieval compute precise attention, estimation approximates by centroids."
            ),
            "code_mapping": (
                "`retroinfer_cache.static_pattern_start/end` and `steady_zone_keys/values` implement the steady zone; "
                "`nprobe`/`nprobe_new` are retrieval-zone cluster counts; `es_cluster_num` is the estimation-zone count."
            ),
        },
        {
            "paper_concept": "Wave buffer",
            "paper_evidence": (
                "Paper Section 4.1/4.3 says the wave buffer contains a GPU block cache, a steady-zone buffer, "
                "and an execution buffer, with a CPU-resident buffer manager."
            ),
            "code_mapping": (
                "`cache_hub/retroinfer_cache.py` creates per-layer `WaveBufferCPU`, `cache_keys/cache_values`, "
                "`steady_zone_*`, and `execution_buffer_*`; `wave_buffer_cpu.cpp` owns the CPU `BufferManager`."
            ),
        },
        {
            "paper_concept": "GPU KV block cache capacity",
            "paper_evidence": (
                "Paper Section 5.1 sets GPU cache size to 5% of all KV vectors and physical block size to 2KB."
            ),
            "code_mapping": (
                "`config.compute_retroinfer_block_cache_capacity()` turns `cache_ratio` into cached clusters/pages; "
                "`cache_ratio=0.05` is the paper/default 5% setting, while `cache_ratio=0.0` preserves the code fallback "
                "`3 * retrieval_clusters`. Python `page_size=8` bf16 vectors gives 2KB per K or V page."
            ),
        },
        {
            "paper_concept": "Logical clusters vs physical blocks",
            "paper_evidence": (
                "Paper Section 4.3 describes cluster-level access, fixed-size KV blocks, and a cluster mapping table."
            ),
            "code_mapping": (
                "`WaveBufferCPU` stores `ClusterDescriptor` entries with `inBlockCache`, GPU block IDs, CPU start index, "
                "block count, and LRU pointer; `BufferManager.capacity` is the per-group block-cache page budget."
            ),
        },
        {
            "paper_concept": "Synchronous access and asynchronous update",
            "paper_evidence": (
                "Paper Section 4.3 states block-cache access is on the critical path, but replacement/update is asynchronous."
            ),
            "code_mapping": (
                "`WaveBufferCPU::para_batch_access()` synchronously determines hit/miss blocks, then queues "
                "`para_batch_updata()`; Python waits with `wave_buffer.sync()` before `gather_copy_and_scatter()` updates cache tensors."
            ),
        },
    ]


def trend_sentence(deltas: list[dict[str, Any]]) -> str:
    usable = [
        row
        for row in deltas
        if number_or_none(row.get("decode_throughput_delta_pct")) is not None
        and number_or_none(row.get("peak_process_gpu_memory_delta_mib")) is not None
    ]
    if not usable:
        return "No successful baseline-aligned rows were available, so no cache-size trend can be inferred."
    best_throughput = max(
        usable,
        key=lambda row: number_or_none(row.get("mean_decode_throughput_tokens_s")) or float("-inf"),
    )
    lowest_memory = min(
        usable,
        key=lambda row: number_or_none(row.get("mean_peak_process_gpu_memory_mib")) or float("inf"),
    )
    return (
        "Best mean decode throughput in the measured matrix occurred at "
        f"context {best_throughput['context_len']}, batch {best_throughput['batch_size']}, "
        f"cache_ratio {float(best_throughput['cache_ratio']):g}; the lowest measured process peak memory occurred at "
        f"context {lowest_memory['context_len']}, batch {lowest_memory['batch_size']}, "
        f"cache_ratio {float(lowest_memory['cache_ratio']):g}. Deltas in the table are computed only within identical "
        "context/batch/gen/seed groups against cache_ratio 0.05."
    )


def context_tradeoff_rows(deltas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in deltas:
        groups.setdefault((row.get("context_len"), row.get("batch_size")), []).append(row)

    rows: list[dict[str, Any]] = []
    for (context_len, batch_size), items in sorted(groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        usable = [
            row
            for row in items
            if number_or_none(row.get("mean_decode_throughput_tokens_s")) is not None
            and number_or_none(row.get("mean_peak_process_gpu_memory_mib")) is not None
        ]
        if not usable:
            continue
        best_throughput = max(usable, key=lambda row: number_or_none(row.get("mean_decode_throughput_tokens_s")) or float("-inf"))
        lowest_memory = min(usable, key=lambda row: number_or_none(row.get("mean_peak_process_gpu_memory_mib")) or float("inf"))
        memory_saving_candidates = [
            row
            for row in usable
            if (number_or_none(row.get("peak_process_gpu_memory_delta_mib")) or 0.0) < 0.0
            and (number_or_none(row.get("decode_throughput_delta_pct")) or float("-inf")) >= -5.0
        ]
        best_saving = (
            min(memory_saving_candidates, key=lambda row: number_or_none(row.get("peak_process_gpu_memory_delta_mib")) or 0.0)
            if memory_saving_candidates
            else None
        )
        score_candidates = [
            row
            for row in usable
            if number_or_none(row.get("decode_throughput_delta_pct")) is not None
            and number_or_none(row.get("peak_process_gpu_memory_delta_pct")) is not None
        ]
        best_score = max(
            score_candidates,
            key=lambda row: (
                (number_or_none(row.get("decode_throughput_delta_pct")) or 0.0)
                - abs(number_or_none(row.get("peak_process_gpu_memory_delta_pct")) or 0.0)
            ),
        )
        rows.append(
            {
                "context_len": context_len,
                "batch_size": batch_size,
                "best_throughput_cache_ratio": best_throughput.get("cache_ratio"),
                "best_throughput_delta_pct": best_throughput.get("decode_throughput_delta_pct"),
                "best_throughput_memory_delta_pct": best_throughput.get("peak_process_gpu_memory_delta_pct"),
                "lowest_memory_cache_ratio": lowest_memory.get("cache_ratio"),
                "lowest_memory_delta_pct": lowest_memory.get("peak_process_gpu_memory_delta_pct"),
                "lowest_memory_throughput_delta_pct": lowest_memory.get("decode_throughput_delta_pct"),
                "best_memory_saving_within_5pct_throughput": (
                    best_saving.get("cache_ratio") if best_saving is not None else ""
                ),
                "best_tradeoff_score_cache_ratio": best_score.get("cache_ratio"),
            }
        )
    return rows


def curve_point_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sorted(
        summary,
        key=lambda item: (
            int(item.get("context_len") or 0),
            int(item.get("batch_size") or 0),
            float(item.get("cache_ratio") or 0),
        ),
    ):
        if number_or_none(row.get("mean_peak_process_gpu_memory_mib")) is None and number_or_none(
            row.get("mean_decode_throughput_tokens_s")
        ) is None:
            continue
        rows.append({field: row.get(field) for field in CURVE_POINT_FIELDS})
    return rows


def _curve_group_label(context_len: int, batch_size: int) -> str:
    if context_len >= 1000 and context_len % 1000 == 0:
        context_text = f"{context_len // 1000}K"
    else:
        context_text = str(context_len)
    return f"ctx {context_text}, bsz {batch_size}"


def _curve_axis_max(values: list[float]) -> tuple[float, list[float]]:
    if not values:
        return 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    max_value = max(max(values), 0.0)
    if max_value == 0.0:
        return 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]
    magnitude = 10 ** math.floor(math.log10(max_value))
    for multiplier in (1, 2, 5, 10):
        axis_max = multiplier * magnitude
        if max_value <= axis_max:
            ticks = [axis_max * idx / 4.0 for idx in range(5)]
            return axis_max, ticks
    axis_max = 10 * magnitude
    return axis_max, [axis_max * idx / 4.0 for idx in range(5)]


def _format_curve_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _collect_curve_points(
    curve_rows: list[dict[str, Any]],
    metric: str,
    scale: float,
) -> dict[tuple[int, int], list[tuple[float, float]]]:
    points: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for row in curve_rows:
        context_len = int_or_none(row.get("context_len"))
        batch_size = int_or_none(row.get("batch_size"))
        cache_ratio = number_or_none(row.get("cache_ratio"))
        value = number_or_none(row.get(metric))
        if context_len is None or batch_size is None or cache_ratio is None or value is None:
            continue
        points.setdefault((context_len, batch_size), []).append((cache_ratio, value * scale))
    for key in list(points):
        points[key] = sorted(points[key], key=lambda item: item[0])
    return points


def _make_cache_curve_primitives(
    points: dict[tuple[int, int], list[tuple[float, float]]],
    title: str,
    ylabel: str,
) -> dict[str, Any]:
    width = 1280
    height = 760
    margin_left = 105
    margin_right = 55
    margin_top = 130
    margin_bottom = 110
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_x1 = width - margin_right
    plot_y1 = height - margin_bottom
    plot_width = plot_x1 - plot_x0
    plot_height = plot_y1 - plot_y0
    ratios = sorted({ratio for group_points in points.values() for ratio, _ in group_points})
    values = [value for group_points in points.values() for _, value in group_points]
    min_ratio = min(ratios) if ratios else 0.0
    max_ratio = max(ratios) if ratios else 1.0
    if min_ratio == max_ratio:
        min_ratio = max(0.0, min_ratio - 0.01)
        max_ratio += 0.01
    axis_max, y_ticks = _curve_axis_max(values)

    def x_for_ratio(cache_ratio: float) -> float:
        return plot_x0 + (cache_ratio - min_ratio) / (max_ratio - min_ratio) * plot_width

    def y_for_value(value: float) -> float:
        return plot_y1 - value / axis_max * plot_height

    primitives: list[dict[str, Any]] = [
        {"type": "rect", "x": 0, "y": 0, "w": width, "h": height, "fill": "#FFFFFF", "stroke": None},
        {"type": "text", "x": width / 2, "y": 38, "text": title, "size": 25, "anchor": "middle", "color": "#111111"},
        {
            "type": "text",
            "x": width / 2,
            "y": 68,
            "text": "Measured context/batch groups at cache ratios 0.5%, 5%, and 10%",
            "size": 15,
            "anchor": "middle",
            "color": "#444444",
        },
        {"type": "text", "x": margin_left, "y": 105, "text": ylabel, "size": 16, "anchor": "start", "color": "#111111"},
        {"type": "line", "x1": plot_x0, "y1": plot_y1, "x2": plot_x1, "y2": plot_y1, "stroke": "#222222", "width": 1.4},
        {"type": "line", "x1": plot_x0, "y1": plot_y0, "x2": plot_x0, "y2": plot_y1, "stroke": "#222222", "width": 1.4},
    ]
    for tick in y_ticks:
        y = y_for_value(tick)
        primitives.append({"type": "line", "x1": plot_x0, "y1": y, "x2": plot_x1, "y2": y, "stroke": "#E2E2E2", "width": 1})
        primitives.append(
            {
                "type": "text",
                "x": plot_x0 - 10,
                "y": y + 5,
                "text": _format_curve_tick(tick),
                "size": 12,
                "anchor": "end",
                "color": "#333333",
            }
        )
    for ratio in ratios:
        x = x_for_ratio(ratio)
        primitives.append({"type": "line", "x1": x, "y1": plot_y1, "x2": x, "y2": plot_y1 + 6, "stroke": "#222222", "width": 1.2})
        primitives.append(
            {
                "type": "text",
                "x": x,
                "y": plot_y1 + 25,
                "text": f"{ratio:g}",
                "size": 12,
                "anchor": "middle",
                "color": "#333333",
            }
        )
    primitives.append(
        {
            "type": "text",
            "x": (plot_x0 + plot_x1) / 2,
            "y": plot_y1 + 58,
            "text": "cache_ratio",
            "size": 14,
            "anchor": "middle",
            "color": "#222222",
        }
    )

    for group_idx, (group, group_points) in enumerate(sorted(points.items())):
        color = CURVE_COLORS[group_idx % len(CURVE_COLORS)]
        xy_points = [(x_for_ratio(ratio), y_for_value(value)) for ratio, value in group_points]
        if len(xy_points) > 1:
            primitives.append({"type": "polyline", "points": xy_points, "stroke": color, "width": 4})
        for x, y in xy_points:
            primitives.append(
                {
                    "type": "marker",
                    "x": x,
                    "y": y,
                    "shape": "circle",
                    "size": 7,
                    "fill": color,
                    "stroke": "#FFFFFF",
                    "width": 1.2,
                }
            )
        legend_x = plot_x0 + 10 + (group_idx % 4) * 260
        legend_y = 92 + (group_idx // 4) * 24
        primitives.append({"type": "line", "x1": legend_x, "y1": legend_y, "x2": legend_x + 34, "y2": legend_y, "stroke": color, "width": 4})
        primitives.append({"type": "marker", "x": legend_x + 17, "y": legend_y, "shape": "circle", "size": 6, "fill": color, "stroke": "#FFFFFF", "width": 1.0})
        primitives.append(
            {
                "type": "text",
                "x": legend_x + 44,
                "y": legend_y + 5,
                "text": _curve_group_label(group[0], group[1]),
                "size": 13,
                "anchor": "start",
                "color": "#222222",
            }
        )
    return {"width": width, "height": height, "primitives": primitives}


def write_curve_artifacts(output_dir: Path, summary: list[dict[str, Any]]) -> list[str]:
    curves_dir = output_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = curve_point_rows(summary)
    curve_points_path = curves_dir / "curve_points.csv"
    write_csv(curve_points_path, curve_rows, CURVE_POINT_FIELDS)
    written = [str(curve_points_path.relative_to(output_dir))]
    manifest: dict[str, Any] = {
        "generated_at": utc_now(),
        "source": "throughput_eval/reproduce_block_cache.py",
        "curve_points": str(curve_points_path.relative_to(output_dir)),
        "curves": [],
    }

    for spec in CURVE_SPECS:
        points = _collect_curve_points(curve_rows, str(spec["metric"]), float(spec["scale"]))
        if not any(points.values()):
            continue
        drawing = _make_cache_curve_primitives(points, str(spec["title"]), str(spec["ylabel"]))
        basename = f"block_cache_{spec['name']}"
        svg_path = curves_dir / f"{basename}.svg"
        pdf_path = curves_dir / f"{basename}.pdf"
        png_path = curves_dir / f"{basename}.png"
        svg_path.write_text(render_svg(drawing), encoding="utf-8")
        pdf_path.write_bytes(render_pdf(drawing))
        png_width, png_height, png_pixels = render_png(drawing, scale=2)
        write_png(png_path, png_width, png_height, png_pixels)
        files = [
            str(svg_path.relative_to(output_dir)),
            str(pdf_path.relative_to(output_dir)),
            str(png_path.relative_to(output_dir)),
        ]
        manifest["curves"].append(
            {
                "name": spec["name"],
                "metric": spec["metric"],
                "unit": spec["ylabel"],
                "files": files,
            }
        )
        written.extend(files)

    manifest_path = curves_dir / "curve_manifest.json"
    write_json(manifest_path, manifest)
    written.append(str(manifest_path.relative_to(output_dir)))
    return written


def write_report(
    path: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    results: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    curve_files: list[str] | None = None,
) -> None:
    passed = [result for result in results if result.get("status") == "passed"]
    gpu_query = environment.get("commands", {}).get("nvidia_smi_query", {}).get("output", "")
    package_status = environment.get("package_status", {})
    package_lines = ", ".join(f"{name}={status}" for name, status in sorted(package_status.items()))

    lines = [
        "# RetroInfer block-cache capacity single-A100 report",
        "",
        f"- Generated at: `{config['generated_at']}`",
        f"- Suite: `{config['suite']}`",
        f"- Model/task: `{config['model_name']}` / `{config['task_name']}`",
        f"- Context lengths: `{config['context_lens']}`",
        f"- Batch sizes: `{config['batch_sizes']}`",
        f"- Decode tokens per request: `{config['gen_len']}`",
        f"- Rounds per point: `{config['rounds']}`",
        f"- Seed: `{config['seed']}`",
        f"- Retrieval/estimation budget: `{config['retrieval_budget']}` / `{config['estimation_budget']}`",
        f"- Cache ratios: `{config['cache_ratios']}`",
        f"- Baseline/default ratio for deltas: `{config['baseline_cache_ratio']}`",
        f"- GPU visibility: `CUDA_VISIBLE_DEVICES={config['cuda_visible_devices']}`",
        f"- GPU observed: `{gpu_query}`",
        f"- Package availability: `{package_lines}`",
        "",
        "## 1. Paper/code mapping for block cache",
        "",
        markdown_table(
            config["paper_code_mapping"],
            ["paper_concept", "paper_evidence", "code_mapping"],
        ),
        "",
        (
            "Design choice: this experiment varies `--cache_ratio`, because that value now feeds the same "
            "`compute_retroinfer_block_cache_capacity()` helper used by the runtime constructor before it allocates "
            "`cache_keys/cache_values` and instantiates `WaveBufferCPU`. The 0.05 point matches the paper's 5% GPU "
            "cache setting; `cache_ratio=0.0` remains the repository's backward-compatible fallback when included."
        ),
        "",
        "## 2. Execution and artifact integrity",
        "",
        f"- Planned runs: `{len(results)}`",
        f"- Successful measured runs: `{len(passed)}`",
        "- Raw stdout/stderr: `raw_logs/*.txt`.",
        "- Process-level GPU memory samples: `memory_samples/*.jsonl`.",
        "- Structured run records: `results.jsonl` and `results.csv`.",
        "- Aggregates and default-relative deltas: `summary.csv`, `per_round_table.csv`, and `deltas.csv`.",
        "",
        "## 3. Per-round throughput and memory",
        "",
        markdown_table(
            per_round_rows(results),
            [
                "context_len",
                "batch_size",
                "cache_ratio",
                "cache_role",
                "round",
                "block_cache_total_gib",
                "block_cache_percent_of_a100_80gb",
                "block_cache_percent_of_peak_process_memory",
                "non_block_cache_peak_process_memory_mib",
                "decode_throughput_tokens_s",
                "peak_process_gpu_memory_mib",
                "failure_class",
                "status",
            ],
        ),
        "",
        "## 4. Block-cache memory accounting and aggregate performance",
        "",
        markdown_table(
            summary,
            [
                "context_len",
                "batch_size",
                "cache_ratio",
                "cache_role",
                "block_cache_total_gib",
                "block_cache_percent_of_a100_80gb",
                "mean_block_cache_percent_of_peak_process_memory",
                "mean_non_block_cache_peak_process_memory_mib",
                "round_decode_throughput_tokens_s",
                "mean_decode_throughput_tokens_s",
                "variance_decode_throughput_tokens_s",
                "mean_peak_process_gpu_memory_mib",
                "mean_torch_cuda_peak_allocated_mib",
                "mean_torch_cuda_peak_reserved_mib",
            ],
        ),
        "",
        "## 5. Deltas relative to cache_ratio 0.05",
        "",
        markdown_table(
            deltas,
            [
                "context_len",
                "batch_size",
                "cache_ratio",
                "block_cache_total_gib_delta",
                "peak_process_gpu_memory_delta_mib",
                "peak_process_gpu_memory_delta_pct",
                "block_cache_percent_of_peak_process_memory_delta",
                "non_block_cache_peak_process_memory_delta_mib",
                "decode_throughput_delta_tokens_s",
                "decode_throughput_delta_pct",
                "decode_throughput_ratio_vs_baseline",
            ],
        ),
        "",
        "## 6. Context-specific tradeoff summary",
        "",
        markdown_table(
            context_tradeoff_rows(deltas),
            [
                "context_len",
                "batch_size",
                "best_throughput_cache_ratio",
                "best_throughput_delta_pct",
                "best_throughput_memory_delta_pct",
                "lowest_memory_cache_ratio",
                "lowest_memory_delta_pct",
                "lowest_memory_throughput_delta_pct",
                "best_memory_saving_within_5pct_throughput",
                "best_tradeoff_score_cache_ratio",
            ],
        ),
        "",
        "## 7. Curve artifacts",
        "",
        "Curves are generated directly from the measured rows in `summary.csv`.",
        "",
        markdown_table(
            [{"artifact": file_path} for file_path in (curve_files or [])],
            ["artifact"],
        ),
        "",
        "## 8. Trend, tradeoff, and comparability note",
        "",
        trend_sentence(deltas),
        "",
        (
            "The block cache changes GPU memory directly through the per-layer `cache_keys/cache_values` tensors and "
            "also changes data-transfer behavior through the LRU-managed hit/miss path. Throughput therefore need not "
            "vary monotonically with estimated cache bytes: very small caches can reduce memory but increase CPU-to-GPU "
            "miss traffic, while larger caches consume more GPU memory and can help only if their extra capacity raises "
            "the hit ratio enough to offset allocation and copy overhead."
        ),
        "",
        (
            "Comparability note: the paper reports the default 5% GPU cache on an A100 80GB server with a specific "
            "CPU/NUMA/PCIe setup. This artifact uses the visible single A100 80GB GPU, the local CPU/NUMA topology, "
            "the repository NIAH prompts, and a bounded pressure matrix. Absolute numbers are host-specific; the "
            "valid conclusion is the within-host trend across cache capacities under fixed non-cache variables."
        ),
        "",
    ]
    lines.extend(
        [
            "## 9. Artifact manifest",
            "",
            "- `config.json`: exact matrix configuration, command plan, and paper/code mapping.",
            "- `commands.jsonl`: exact command for every planned run.",
            "- `environment.json`: hardware, git, package, CUDA, and NUMA observations.",
            "- `raw_logs/*.txt`: unedited stdout/stderr, including structured `RETROINFER_RESULT_JSON` output.",
            "- `memory_samples/*.jsonl`: sampled process-level GPU memory events.",
            "- `results.jsonl` / `results.csv`: per-run parsed records.",
            "- `per_round_table.csv`: report-ready per-round table.",
            "- `summary.csv`: grouped means, stdev, variance, and block-cache capacity metadata.",
            "- `deltas.csv`: memory and throughput deltas against cache_ratio 0.05 within identical non-cache settings.",
            "- `curves/curve_points.csv` and `curves/curve_manifest.json`: reconstructable cache-ratio curve inputs and generated curve manifest.",
            "- `curves/block_cache_*.{pdf,svg,png}`: generated memory-share, peak-memory, and throughput curves.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_config(output_dir: Path, args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    commands = [{**run, "command": command_for_run(run, args)} for run in runs]
    context_batch_groups = selected_context_batch_groups(args)
    config = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "suite": args.suite,
        "model_name": args.model_name,
        "task_name": args.task_name,
        "context_lens": sorted({context_len for context_len, _ in context_batch_groups}),
        "batch_sizes": sorted({batch_size for _, batch_size in context_batch_groups}),
        "context_batch_groups": context_batch_group_dicts(context_batch_groups),
        "gen_len": args.gen_len,
        "rounds": args.rounds,
        "seed": args.seed,
        "cuda_visible_devices": args.cuda_visible_devices,
        "use_numactl": args.use_numactl,
        "use_cuda_graph": args.use_cuda_graph,
        "retrieval_budget": args.retrieval_budget,
        "estimation_budget": args.estimation_budget,
        "cache_ratios": args.cache_ratios or parse_float_list(DEFAULT_CACHE_RATIOS),
        "baseline_cache_ratio": args.baseline_cache_ratio,
        "dry_run": args.dry_run,
        "fixed_non_cache_variables": {
            "model_name": args.model_name,
            "task_name": args.task_name,
            "context_batch_groups": context_batch_group_dicts(context_batch_groups),
            "gen_len": args.gen_len,
            "prefill_bsz": args.prefill_bsz,
            "retrieval_budget": args.retrieval_budget,
            "estimation_budget": args.estimation_budget,
            "seed": args.seed,
            "cuda_visible_devices": args.cuda_visible_devices,
            "use_cuda_graph": args.use_cuda_graph,
        },
        "gpu_memory_measurement": {
            "enabled": args.gpu_memory_sampling,
            "primary_metric": "peak_process_gpu_memory_mib",
            "unit": "MiB",
            "sample_interval_s": args.gpu_memory_sample_interval_s,
            "sampling_source": "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "scope": "sum of process-level used_memory for the launched runner PID and its descendants",
            "require_idle_gpu": args.require_idle_gpu,
        },
        "paper_source": {
            "arxiv": "2505.02922",
            "title": "RetroInfer: A Vector Storage Engine for Scalable Long-Context LLM Inference",
            "block_cache_sections": ["1", "4.1", "4.3", "4.6", "5.1", "5.4"],
            "default_gpu_cache_ratio": 0.05,
            "physical_block_size": "2KB per K or V page in the implementation mapping",
        },
        "paper_code_mapping": paper_code_mapping(),
        "runs": commands,
    }
    write_json(output_dir / "config.json", config)
    write_jsonl(output_dir / "commands.jsonl", commands)
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="block_cache_single_a100")
    parser.add_argument("--context-lens", dest="context_lens", type=parse_int_list, default=parse_int_list(DEFAULT_CONTEXT_LENS))
    parser.add_argument("--batch-sizes", dest="batch_sizes", type=parse_int_list, default=parse_int_list(DEFAULT_BATCH_SIZES))
    parser.add_argument(
        "--context-batch-groups",
        dest="context_batch_groups",
        type=parse_context_batch_groups,
        help="Comma-separated context/batch pairs such as 120000x1,120000x8. Overrides the context/batch cross product.",
    )
    parser.add_argument("--cache-ratios", dest="cache_ratios", type=parse_float_list, default=parse_float_list(DEFAULT_CACHE_RATIOS))
    parser.add_argument("--baseline-cache-ratio", type=float, default=DEFAULT_BASELINE_CACHE_RATIO)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--gen-len", type=int, default=100)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--task-name", default=DEFAULT_TASK)
    parser.add_argument("--prefill-bsz", type=int, default=1)
    parser.add_argument("--retrieval-budget", type=float, default=0.018)
    parser.add_argument("--estimation-budget", type=float, default=0.232)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
    parser.add_argument(
        "--repeat-successes-only",
        action="store_true",
        help="Run only the first round for failed points while still repeating successful points.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if args.gpu_memory_sample_interval_s <= 0:
        raise SystemExit("--gpu-memory-sample-interval-s must be > 0")
    if not args.cache_ratios:
        raise SystemExit("--cache-ratios must contain at least one ratio")
    if not any(ratios_equal(ratio, args.baseline_cache_ratio) for ratio in args.cache_ratios):
        print(
            "warning: --cache-ratios does not include --baseline-cache-ratio; "
            "baseline-relative deltas will be empty",
            file=sys.stderr,
        )
    context_batch_groups = selected_context_batch_groups(args)
    if not context_batch_groups:
        raise SystemExit("at least one context/batch group is required")
    if len(set(context_batch_groups)) != len(context_batch_groups):
        raise SystemExit("--context-batch-groups must not contain duplicates")


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        output_dir = PROJECT_ROOT / "research" / f"block_cache_single_a100_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = collect_environment()
    write_json(output_dir / "environment.json", environment)

    results: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    if args.repeat_successes_only:
        stop_requested = False
        for context_len, batch_size in selected_context_batch_groups(args):
            for cache_ratio_value in args.cache_ratios or []:
                first_run = make_run(args, context_len, batch_size, cache_ratio_value, 1)
                runs.append(first_run)
                first_result = run_one(first_run, args, output_dir)
                results.append(first_result)
                write_jsonl(output_dir / "results.jsonl", results)
                write_csv(output_dir / "results.csv", results, RUN_FIELDS)
                if first_result["status"] == "passed":
                    for round_idx in range(2, args.rounds + 1):
                        run = make_run(args, context_len, batch_size, cache_ratio_value, round_idx)
                        runs.append(run)
                        result = run_one(run, args, output_dir)
                        results.append(result)
                        write_jsonl(output_dir / "results.jsonl", results)
                        write_csv(output_dir / "results.csv", results, RUN_FIELDS)
                        if args.stop_on_error and result["status"] != "passed":
                            stop_requested = True
                            break
                elif args.stop_on_error:
                    stop_requested = True
                if stop_requested:
                    break
            if stop_requested:
                break
    else:
        runs = build_runs(args)
        for run in runs:
            result = run_one(run, args, output_dir)
            results.append(result)
            write_jsonl(output_dir / "results.jsonl", results)
            write_csv(output_dir / "results.csv", results, RUN_FIELDS)
            if args.stop_on_error and result["status"] != "passed":
                break

    config = write_config(output_dir, args, runs)

    fill_missing_block_cache_capacity_fields(results)
    write_jsonl(output_dir / "results.jsonl", results)
    write_csv(output_dir / "results.csv", results, RUN_FIELDS)

    summary = summarize_results(results)
    deltas = compute_deltas(summary, args.baseline_cache_ratio)
    write_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    write_csv(output_dir / "deltas.csv", deltas, DELTA_FIELDS)
    write_csv(output_dir / "per_round_table.csv", per_round_rows(results), PER_ROUND_FIELDS)
    curve_files = write_curve_artifacts(output_dir, summary)
    write_report(output_dir / "REPORT.md", config, environment, results, summary, deltas, curve_files)
    print(f"Wrote block-cache artifacts to {output_dir}")
    failed = [result for result in results if result.get("status") != "passed"]
    return 1 if failed and not args.dry_run else 0


if __name__ == "__main__":
    raise SystemExit(main())
