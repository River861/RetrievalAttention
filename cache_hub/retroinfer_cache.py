import math
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from retroinfer_kernels import ThreadPool, WaveBufferCPU
from retroinfer_kernels import (
    gather_copy_and_concat,
    gather_copy_and_scatter,
    gather_copy_cluster_and_concat_fuse,
    gather_copy_vectors,
    batch_gemm_softmax,
)

from config.config import compute_retroinfer_block_cache_capacity
from .cache import KV_Cache
from .kmeans import segment_k_means
from weighted_flash_decoding import weighted_flash_decoding


RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR_ENV = "RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}")


def _env_float(name: str, default: float) -> tuple[float, str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return default, "unset"
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite positive float; got {value!r}") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive float; got {value!r}")
    return parsed, value.strip()


def _env_optional_positive_int(name: str) -> tuple[int | None, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None, "unset"
    raw_value = value.strip()
    try:
        parsed = int(raw_value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return parsed, raw_value


def _resolve_buffer_pages_per_cluster(pages_per_cluster: int) -> dict:
    if pages_per_cluster <= 0:
        raise ValueError(f"pages_per_cluster must be positive; got {pages_per_cluster}")
    floor, floor_env = _env_optional_positive_int(
        RETROINFER_BUFFER_PAGES_PER_CLUSTER_FLOOR_ENV
    )
    floor_active = floor is not None
    effective = max(pages_per_cluster, floor) if floor_active else pages_per_cluster
    return {
        "block_cache_pages_per_cluster": pages_per_cluster,
        "floor": floor,
        "floor_env": floor_env,
        "floor_active": floor_active,
        "effective": effective,
        "source": (
            "env_floor"
            if floor_active and effective != pages_per_cluster
            else ("env_floor_noop" if floor_active else "block_cache_pages_per_cluster")
        ),
    }


def _env_positive_int(name: str) -> tuple[int | None, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None, "unset"
    raw_value = value.strip()
    try:
        parsed = int(raw_value, 10)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer; got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    available_cores = os.cpu_count()
    if available_cores is not None and parsed > available_cores:
        raise ValueError(
            f"{name}={parsed} exceeds available CPU cores ({available_cores})"
        )
    return parsed, raw_value


def _resolve_wave_buffer_cpu_core(config_core: int) -> dict:
    if config_core <= 0:
        raise ValueError(f"RetroInfer core must be positive; got {config_core}")
    override, override_env = _env_positive_int("RETROINFER_WAVE_BUFFER_CORE_OVERRIDE")
    override_active = override is not None
    effective_core = override if override_active else config_core
    return {
        "config_value": config_core,
        "effective_value": effective_core,
        "override_env": override_env,
        "override_active": override_active,
        "source": "env_override" if override_active else "config_core",
    }


def _env_allocation_policy(name: str) -> tuple[str, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return "auto", "unset"
    raw_value = value.strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"auto", "default"}:
        return "auto", raw_value
    if normalized in {"late", "defer", "deferred", "after_prefill", "post_prefill"}:
        return "late", raw_value
    if normalized in {"preallocate", "prealloc", "before_prefill", "prefill"}:
        return "preallocate", raw_value
    raise ValueError(
        f"{name} must be one of auto/default, late/after_prefill/deferred, "
        f"or preallocate/before_prefill; got {value!r}"
    )


def _env_index_metadata_prefill_residency(name: str) -> tuple[str, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return "cpu", "unset"
    raw_value = value.strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"cpu", "host", "host_cpu", "default"}:
        return "cpu", raw_value
    if normalized in {"gpu", "cuda", "device"}:
        return "gpu", raw_value
    raise ValueError(f"{name} must be one of cpu/host/default or gpu/cuda/device; got {value!r}")


def _env_late_index_metadata_migration_policy(name: str) -> tuple[str, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return "pageable_blocking", "unset"
    raw_value = value.strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"default", "pageable", "pageable_blocking", "blocking"}:
        return "pageable_blocking", raw_value
    if normalized in {"pin", "pinned", "pin_only", "pinned_blocking"}:
        return "pinned_blocking", raw_value
    if normalized in {"non_blocking", "pinned_non_blocking"}:
        return "pinned_non_blocking", raw_value
    if normalized in {"side_stream", "pinned_side_stream", "stream"}:
        return "pinned_side_stream", raw_value
    raise ValueError(
        f"{name} must be one of default/pageable_blocking, pin_only/pinned_blocking, "
        f"pinned_non_blocking, or pinned_side_stream; got {value!r}"
    )


def _env_late_block_cache_init_policy(name: str) -> tuple[str, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return "zero", "unset"
    raw_value = value.strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"default", "zero", "zeros", "zero_fill", "zero_filled"}:
        return "zero", raw_value
    if normalized in {"empty", "uninitialized", "uninitialised", "no_zero", "no_zero_fill"}:
        return "uninitialized", raw_value
    raise ValueError(
        f"{name} must be one of default/zero/zero_fill or empty/uninitialized/no_zero_fill; got {value!r}"
    )


def _env_scratch_buffer_init_policy(name: str) -> tuple[str, str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return "zero", "unset"
    raw_value = value.strip()
    normalized = raw_value.lower().replace("-", "_")
    if normalized in {"default", "zero", "zeros", "zero_fill", "zero_filled"}:
        return "zero", raw_value
    if normalized in {"empty", "uninitialized", "uninitialised", "no_zero", "no_zero_fill"}:
        return "uninitialized", raw_value
    raise ValueError(
        f"{name} must be one of default/zero/zero_fill or empty/uninitialized/no_zero_fill; got {value!r}"
    )


def _parse_layer_range_list(value: str, layer_num: int) -> set[int]:
    layers: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start = int(raw_start)
            end = int(raw_end)
            if end < start:
                raise ValueError(f"Invalid descending layer range {part!r}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    invalid = [layer_idx for layer_idx in sorted(layers) if layer_idx < 0 or layer_idx >= layer_num]
    if invalid:
        raise ValueError(f"Layer indices out of range for {layer_num} layers: {invalid}")
    return layers


def _parse_optional_layer_selector(value: str, layer_num: int, env_name: str) -> set[int]:
    spec = value.strip()
    if not spec or spec.lower() in {"off", "none", "default"}:
        return set()
    if spec.lower() == "all":
        return set(range(layer_num))
    if spec.lower().startswith("layers:"):
        spec = spec.split(":", 1)[1].strip()
    if not spec:
        return set()
    try:
        return _parse_layer_range_list(spec, layer_num)
    except ValueError as exc:
        raise ValueError(f"Invalid {env_name} layer selector {value!r}: {exc}") from exc


def _parse_layer_capacity_scale_spec(value: str, layer_num: int) -> dict[int, float]:
    spec = value.strip()
    if not spec or spec.lower() in {"all", "default", "off", "none"}:
        return {}
    if spec.lower().startswith("layers:"):
        spec = spec.split(":", 1)[1].strip()
    if not spec:
        return {}

    scales: dict[int, float] = {}
    env_name = "RETROINFER_LAYER_CACHE_CAPACITY_SCALE"
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"{env_name} entries must look like 1=0.25 or 2-4=50%; got {part!r}")
        raw_layers, raw_scale = part.split("=", 1)
        raw_layers = raw_layers.strip()
        raw_scale = raw_scale.strip()
        if not raw_layers or not raw_scale:
            raise ValueError(f"{env_name} entries must include both layers and scale; got {part!r}")

        try:
            target_layers = _parse_layer_range_list(raw_layers, layer_num)
        except ValueError as exc:
            raise ValueError(f"Invalid {env_name} layer selector {raw_layers!r}: {exc}") from exc
        if not target_layers:
            raise ValueError(f"{env_name} layer selector {raw_layers!r} did not select any layers")

        try:
            scale = float(raw_scale[:-1]) / 100.0 if raw_scale.endswith("%") else float(raw_scale)
        except ValueError as exc:
            raise ValueError(f"{env_name} scale for {raw_layers!r} must be numeric; got {raw_scale!r}") from exc
        if not math.isfinite(scale) or scale <= 0.0 or scale > 1.0:
            raise ValueError(f"{env_name} scale for {raw_layers!r} must be > 0 and <= 1; got {raw_scale!r}")

        duplicate_layers = sorted(layer_idx for layer_idx in target_layers if layer_idx in scales)
        if duplicate_layers:
            raise ValueError(f"{env_name} assigns layers more than once: {duplicate_layers}")
        for layer_idx in target_layers:
            scales[layer_idx] = scale
    return scales


def _cpu_zeros(shape, dtype: torch.dtype, pin_memory: bool = False):
    return torch.zeros(shape, dtype=dtype, pin_memory=pin_memory).contiguous()


def _gpu_zeros(shape, dtype: torch.dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device).contiguous()


def _gpu_empty(shape, dtype: torch.dtype, device):
    return torch.empty(shape, dtype=dtype, device=device).contiguous()


class retroinfer_cache(KV_Cache):
    """
    A class representing the KV Cache of RetroInfer.
    """

    def __init__(
        self,
        valid_start,    # numpy array of valid start positions for each sample in the batch
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        max_new_length: int,
        static_pattern_start: int,
        static_pattern_end: int,
        core: int,
        n_centroids: int,
        n_segment: int,
        pages_per_cluster: int,     # 1 cluster = 2 pages = 2 * 8 vectors
        retrieval_budget: float,
        estimation_budget: float,
        cache_ratio: float,         # ratio of cache size to sequence length
        buffer_cluster_num: int,    # number of clusters in the buffer
        use_cuda_graph: bool,
        prefill_bsz: int,
        num_gpus: int,
        model_size: int
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, prefill_bsz, num_gpus, model_size)
        self.device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        wave_buffer_core_resolution = _resolve_wave_buffer_cpu_core(core)
        self.wave_buffer_cpu_core_config_value = wave_buffer_core_resolution["config_value"]
        self.wave_buffer_cpu_core_effective = wave_buffer_core_resolution["effective_value"]
        self.wave_buffer_cpu_core_override_env = wave_buffer_core_resolution["override_env"]
        self.wave_buffer_cpu_core_override_active = wave_buffer_core_resolution["override_active"]
        self.wave_buffer_cpu_core_source = wave_buffer_core_resolution["source"]

        # constant values
        self.RSQRT_DIM = 1.0 / math.sqrt(self.head_dim)
        self.DTYPE_MIN = torch.finfo(self.dtype).min

        self.valid_start_list = valid_start

        self.static_pattern_start = static_pattern_start
        self.static_pattern_end = static_pattern_end
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        self.group_size = self.num_heads // self.kv_head
        self.batch_groups = self.batch_size * self.kv_head

        self.page_size = 8
        avg_cluster_size = pages_per_cluster * self.page_size
        self.UPDATE_SEGMENT = 1024   # update segment size
        self.UPDATE_CENTROIDS = max(round(self.UPDATE_SEGMENT/avg_cluster_size) // 8 * 8, 8)  # must be divisible by 8
        self.UPDATE_NPROBE = max(round(self.UPDATE_CENTROIDS * retrieval_budget), 1)  # update retrieve zone size per segment
        self.UPDATE_ES = max(round(self.UPDATE_CENTROIDS * estimation_budget), 1)  # update estimation zone size per segment

        # whether to build index when prefilling, update index when decoding
        self.input_length = self.max_length - max_new_length
        actual_gen_len = max_new_length - 1  # exclude the first token generated during prefilling
        assert actual_gen_len >= 0, f"Decoding generation length({actual_gen_len}) should be larger than or equal to 0"
        if self.input_length <= 0:
            raise ValueError(f"input length({self.input_length}) should be larger than 0")
        elif self.input_length < self.static_pattern_total + self.UPDATE_SEGMENT:
            # input length is too short, no need to build index during prefilling
            self.build_index_when_prefilling = False
            # update index when decoding, depends on whether input + output length exceed UPDATE_SEGMENT 
            self.will_update_index = (self.input_length-self.static_pattern_total+actual_gen_len) > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.static_pattern_total + self.UPDATE_SEGMENT
                self.list_stride = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_SEGMENT
                self.n_centroids_new = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_CENTROIDS
                self.nprobe_new = ((self.input_length-self.static_pattern_total+actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_NPROBE
            else:
                # fall back to full attention, all KV stores in steady zone
                self.static_stride = self.input_length + actual_gen_len
                self.list_stride = 0
                self.n_centroids_new = 0
                self.nprobe_new = 0
        else:
            self.build_index_when_prefilling = True
            # update index when decoding, depends on whether output length exceed UPDATE_SEGMENT
            self.will_update_index = actual_gen_len > self.UPDATE_SEGMENT
            # set steady zone size, cpu kv cache size and index update parameters
            if self.will_update_index:
                self.static_stride = self.UPDATE_SEGMENT + self.static_pattern_total
                self.list_stride = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_SEGMENT + self.input_length - self.static_pattern_total
                self.n_centroids_new = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_CENTROIDS
                self.nprobe_new = ((actual_gen_len-1) // self.UPDATE_SEGMENT) * self.UPDATE_NPROBE
            else: 
                self.static_stride = actual_gen_len + self.static_pattern_total
                self.list_stride = self.input_length - self.static_pattern_total
                self.n_centroids_new = 0
                self.nprobe_new = 0

        # steady zone keys & values
        self.steady_zone_keys = [
            torch.zeros((self.batch_size, self.kv_head, self.static_stride, self.head_dim), 
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)]) 
            for ldx in range(self.layer_num)
        ]
        self.steady_zone_values = [
            torch.zeros((self.batch_size, self.kv_head, self.static_stride, self.head_dim), 
                        dtype=self.dtype, device=self.layer_mapping[str(ldx)]) 
            for ldx in range(self.layer_num)
        ]

        # index parameters
        self.n_segment = n_segment
        self.n_centroids = n_centroids if self.build_index_when_prefilling else 0
        assert self.n_centroids % math.lcm(8, self.n_segment) == 0, \
            f"n_centroids({self.n_centroids}) should be divisible by LCM of 8 and n_segment({self.n_segment})"
        # retrieve zone size (count by clusters)
        self.nprobe = max(round(self.n_centroids*retrieval_budget), 1)
        self.nprobe = min(self.nprobe, self.n_centroids)
        # estimation zone size (count by clusters)
        self.es_cluster_num = min(round(self.n_centroids*estimation_budget), self.n_centroids-self.nprobe)
        # retrieve zone + estimation zone size
        self.max_compute_cluster_num = self.es_cluster_num + self.nprobe
        assert self.max_compute_cluster_num <= self.n_centroids, \
            f"max_compute_cluster_num({self.max_compute_cluster_num}) should <= n_centroids({self.n_centroids})"
        print(f"Initial n_centroids: {self.n_centroids}, nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}")

        # CUDA graphs
        self.use_cuda_graph = use_cuda_graph
        if self.will_update_index:   # need update when decoding
            if self.use_cuda_graph:
                print("Index will be updated during decoding, so CUDA Graph will be disabled.")
            self.use_cuda_graph = False
        elif not self.build_index_when_prefilling:  # not build index during prefilling and not update during decoding
            if self.use_cuda_graph:
                print("Input + output length too small, fall back to full attention, so CUDA Graph will be disabled.")
            self.use_cuda_graph = False
        if self.use_cuda_graph:
            self.topk_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            if self.es_cluster_num > 0:
                self.es_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            self.attn_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            self.update_cudagraphs = [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]

        # calculate the GPU block cache size and compute buffer size (count by pages)
        self.block_cache_capacity = compute_retroinfer_block_cache_capacity(
            n_centroids=self.n_centroids,
            n_centroids_new=self.n_centroids_new,
            nprobe=self.nprobe,
            nprobe_new=self.nprobe_new,
            pages_per_cluster=pages_per_cluster,
            cache_ratio=cache_ratio,
        )
        self.cache_ratio = cache_ratio
        self.pages_per_cluster = pages_per_cluster
        self.cache_cluster_num = self.block_cache_capacity["cache_cluster_num"]
        self.cache_size = self.block_cache_capacity["cache_pages"]
        self.layer_cache_residency_mode = os.environ.get("RETROINFER_LAYER_CACHE_RESIDENCY", "all").strip() or "all"
        self.layer_cache_capacity_scale_spec = os.environ.get("RETROINFER_LAYER_CACHE_CAPACITY_SCALE", "").strip()
        self.stream_only_layers_spec = os.environ.get("RETROINFER_STREAM_ONLY_LAYERS", "").strip()
        self.stage_index_metadata_layers_spec = os.environ.get("RETROINFER_STAGE_INDEX_METADATA_LAYERS", "").strip()
        self.stream_only_layers = _parse_optional_layer_selector(
            self.stream_only_layers_spec,
            self.layer_num,
            "RETROINFER_STREAM_ONLY_LAYERS",
        )
        self.stage_index_metadata_layers = _parse_optional_layer_selector(
            self.stage_index_metadata_layers_spec,
            self.layer_num,
            "RETROINFER_STAGE_INDEX_METADATA_LAYERS",
        )
        (
            self.index_metadata_prefill_residency,
            self.index_metadata_prefill_residency_env,
        ) = _env_index_metadata_prefill_residency("RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY")
        (
            self.index_metadata_late_migration_policy,
            self.index_metadata_late_migration_policy_env,
        ) = _env_late_index_metadata_migration_policy("RETROINFER_LATE_INDEX_METADATA_MIGRATION_POLICY")
        (
            self.late_block_cache_init_policy,
            self.late_block_cache_init_policy_env,
        ) = _env_late_block_cache_init_policy("RETROINFER_LATE_BLOCK_CACHE_INIT")
        (
            self.scratch_buffer_init_policy,
            self.scratch_buffer_init_policy_env,
        ) = _env_scratch_buffer_init_policy("RETROINFER_SCRATCH_BUFFER_INIT")
        self.index_metadata_late_migration_host_pinned = (
            self.index_metadata_late_migration_policy
            in {"pinned_blocking", "pinned_non_blocking", "pinned_side_stream"}
        )
        self.index_metadata_late_migration_non_blocking = (
            self.index_metadata_late_migration_policy
            in {"pinned_non_blocking", "pinned_side_stream"}
        )
        if self.index_metadata_prefill_residency == "gpu" and self.stage_index_metadata_layers:
            raise ValueError(
                "RETROINFER_INDEX_METADATA_PREFILL_RESIDENCY=gpu cannot be combined with "
                "RETROINFER_STAGE_INDEX_METADATA_LAYERS, which intentionally stores selected "
                "metadata on pinned host memory with per-device GPU staging buffers"
            )
        self.stream_only_layer_mask = [
            layer_idx in self.stream_only_layers for layer_idx in range(self.layer_num)
        ]
        self.stage_index_metadata_layer_mask = [
            layer_idx in self.stage_index_metadata_layers for layer_idx in range(self.layer_num)
        ]
        cache_sizes_before_scales = self._resolve_layer_cache_sizes(self.cache_size)
        for layer_idx in self.stream_only_layers:
            cache_sizes_before_scales[layer_idx] = 0
        self.cache_sizes, self.cache_capacity_scales = self._apply_layer_cache_capacity_scales(
            cache_sizes_before_scales,
            self.cache_size,
        )
        self.cache_strides = self.cache_sizes
        self.cache_cluster_nums = [cache_pages // pages_per_cluster for cache_pages in self.cache_sizes]
        self.block_cache_slot_rotation_env = os.environ.get("RETROINFER_BLOCK_CACHE_SLOT_ROTATION", "")
        self.block_cache_slot_rotation_enabled = _env_flag("RETROINFER_BLOCK_CACHE_SLOT_ROTATION", False)
        self.block_cache_slot_rotation_delta_env = os.environ.get(
            "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA", ""
        )
        self.block_cache_slot_rotation_delta_requested = _env_flag(
            "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA",
            False,
        )
        self.block_cache_slot_rotation_dirty_page_d2h_env = os.environ.get(
            "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DIRTY_PAGE_D2H", ""
        )
        self.block_cache_slot_rotation_dirty_page_d2h_requested = _env_flag(
            "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DIRTY_PAGE_D2H",
            False,
        )
        if self.block_cache_slot_rotation_delta_requested and not self.block_cache_slot_rotation_enabled:
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA requires "
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1"
            )
        if self.block_cache_slot_rotation_dirty_page_d2h_requested and not self.block_cache_slot_rotation_enabled:
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DIRTY_PAGE_D2H requires "
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION=1"
            )
        if (
            self.block_cache_slot_rotation_delta_requested
            and self.block_cache_slot_rotation_dirty_page_d2h_requested
        ):
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DELTA and "
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION_DIRTY_PAGE_D2H are mutually exclusive"
            )
        self.block_cache_slot_rotation_delta_enabled = (
            self.block_cache_slot_rotation_enabled
            and self.block_cache_slot_rotation_delta_requested
        )
        self.block_cache_slot_rotation_dirty_page_d2h_enabled = (
            self.block_cache_slot_rotation_enabled
            and self.block_cache_slot_rotation_dirty_page_d2h_requested
        )
        self.block_cache_slot_rotation_copy_mode = (
            "full_h2d_dirty_page_d2h"
            if self.block_cache_slot_rotation_dirty_page_d2h_enabled
            else (
                "page_delta"
                if self.block_cache_slot_rotation_delta_enabled
                else ("full_layer" if self.block_cache_slot_rotation_enabled else "disabled")
            )
        )
        (
            self.block_cache_gpu_slots_requested,
            self.block_cache_gpu_slots_env,
        ) = _env_optional_positive_int("RETROINFER_BLOCK_CACHE_GPU_SLOTS")
        self.block_cache_slot_rotation_requested_slots = (
            self.block_cache_gpu_slots_requested if self.block_cache_gpu_slots_requested is not None else 8
        )
        self.block_cache_slot_rotation_actual_slots = 0
        self.block_cache_slot_rotation_path_executed = False
        self.block_cache_slot_rotation_allocation_status = "disabled"
        self.block_cache_slot_rotation_allocation_error = None
        self.block_cache_slot_rotation_cuda_graph_mode = "disabled"
        self.block_cache_slot_rotation_cuda_graph_blocker_reason = None
        self.block_cache_slot_rotation_host_meminfo = self._host_memory_facts()
        self.block_cache_slot_rotation_cpu_owner_pinned = False
        self.block_cache_slot_rotation_cpu_owner_bytes = 0
        self.block_cache_slot_rotation_cpu_owner_pinned_bytes = 0
        self.block_cache_slot_rotation_cpu_owner_pageable_bytes = 0
        self.block_cache_slot_rotation_gpu_slot_bytes = 0
        self.block_cache_slot_rotation_gpu_slot_total_bytes = 0
        self.block_cache_slot_rotation_h2d_counts = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_h2d_bytes = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_d2h_counts = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_d2h_bytes = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_h2d_total_count = 0
        self.block_cache_slot_rotation_h2d_total_bytes = 0
        self.block_cache_slot_rotation_d2h_total_count = 0
        self.block_cache_slot_rotation_d2h_total_bytes = 0
        self.block_cache_slot_rotation_page_h2d_counts = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_h2d_bytes = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_h2d_listed_pages = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_h2d_unique_pages = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_d2h_counts = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_d2h_bytes = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_d2h_listed_pages = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_d2h_unique_pages = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_page_h2d_total_count = 0
        self.block_cache_slot_rotation_page_h2d_total_bytes = 0
        self.block_cache_slot_rotation_page_h2d_total_listed_pages = 0
        self.block_cache_slot_rotation_page_h2d_total_unique_pages = 0
        self.block_cache_slot_rotation_page_d2h_total_count = 0
        self.block_cache_slot_rotation_page_d2h_total_bytes = 0
        self.block_cache_slot_rotation_page_d2h_total_listed_pages = 0
        self.block_cache_slot_rotation_page_d2h_total_unique_pages = 0
        self.block_cache_slot_rotation_hit_page_materialization_count = 0
        self.block_cache_slot_rotation_dirty_page_flush_count = 0
        self.block_cache_slot_rotation_page_index_violation_count = 0
        self.block_cache_slot_rotation_event_wait_count = 0
        self.block_cache_slot_rotation_wait_enqueue_elapsed_ms = 0.0
        self.block_cache_slot_rotation_explicit_sync_count = 0
        self.block_cache_slot_rotation_explicit_sync_elapsed_ms = 0.0
        self.block_cache_slot_rotation_overwrite_prevention_count = 0
        self.block_cache_slot_rotation_dirty_read_prevention_count = 0
        self.block_cache_slot_rotation_generation_mismatch_count = 0
        self.block_cache_slot_rotation_violation_count = 0
        self.block_cache_slot_rotation_prefetch_count = 0
        self.block_cache_slot_rotation_wraparound_prefetch_count = 0
        self.block_cache_slot_rotation_initial_prefetch_count = 0
        self.block_cache_slot_rotation_transition_count = 0
        self.block_cache_slot_rotation_transition_tail = []
        self.block_cache_slot_rotation_layer_generation = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_layer_pending_d2h = {}
        self.block_cache_slot_rotation_layer_to_slot = [None for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_slots = []
        self.block_cache_slot_rotation_h2d_streams = {}
        self.block_cache_slot_rotation_d2h_streams = {}
        self.block_cache_slot_rotation_h2d_events = []
        self.block_cache_slot_rotation_d2h_events = []
        self.block_cache_cpu_keys = []
        self.block_cache_cpu_values = []
        self.block_cache_gpu_slot_keys = []
        self.block_cache_gpu_slot_values = []
        self.block_cache_slot_rotation_initial_m_rationale = (
            "m=8 is the default because the canonical 120K/batch8/cache_ratio=0.05 geometry "
            "uses about 186 MiB per layer; eight slots cost about 1.45 GiB and give a seven-layer "
            "prefetch lead, the smallest planned lead likely to hide a full-layer PCIe H2D copy."
        )
        if self.block_cache_slot_rotation_enabled:
            self._validate_block_cache_slot_rotation_configuration()
        self.buffer_nprobe_multiplier, self.buffer_nprobe_multiplier_env = _env_float(
            "RETROINFER_BUFFER_NPROBE_MULTIPLIER",
            4.0,
        )
        buffer_pages_per_cluster_resolution = _resolve_buffer_pages_per_cluster(
            pages_per_cluster
        )
        self.buffer_pages_per_cluster = buffer_pages_per_cluster_resolution["effective"]
        self.buffer_pages_per_cluster_floor = buffer_pages_per_cluster_resolution["floor"]
        self.buffer_pages_per_cluster_floor_env = buffer_pages_per_cluster_resolution["floor_env"]
        self.buffer_pages_per_cluster_floor_active = buffer_pages_per_cluster_resolution["floor_active"]
        self.buffer_pages_per_cluster_source = buffer_pages_per_cluster_resolution["source"]
        self.buffer_min_pages = buffer_cluster_num * self.buffer_pages_per_cluster
        self.buffer_probe_pages = math.ceil(
            (self.nprobe + self.nprobe_new) * self.buffer_nprobe_multiplier
        ) * self.buffer_pages_per_cluster
        self.buffer_size = max(self.buffer_min_pages, self.buffer_probe_pages)
        self.block_cache_telemetry_enabled = _env_flag("RETROINFER_CACHE_TELEMETRY", False)
        self.block_cache_telemetry = self._init_block_cache_telemetry()
        (
            self.block_cache_allocation_policy,
            self.block_cache_allocation_policy_env,
        ) = _env_allocation_policy("RETROINFER_BLOCK_CACHE_ALLOCATION_POLICY")
        self.block_cache_allocation_decision = None
        self.block_cache_allocation_decision_reason = None
        self.block_cache_auto_preallocate_before_prefill = None
        self.block_cache_preallocated_before_prefill = None
        self.block_cache_allocated_after_prefill = False
        self.block_cache_allocation_prepare_cache_calls = 0
        self.block_cache_late_init_effective = False
        self.block_cache_late_init_mode = "zero_fill"
        self.block_cache_late_init_reason = (
            "policy_unset_zero_fill"
            if self.late_block_cache_init_policy_env == "unset"
            else "explicit_zero_fill_policy"
        )
        self.block_cache_late_init_safety = "not_applicable_zero_fill"
        self.block_cache_late_uninitialized_tensor_count = 0
        self.block_cache_late_uninitialized_bytes = 0
        self.scratch_buffer_init_effective = self.scratch_buffer_init_policy == "uninitialized"
        self.scratch_buffer_init_mode = (
            "uninitialized_empty" if self.scratch_buffer_init_effective else "zero_fill"
        )
        self.scratch_buffer_init_reason = (
            "decode_scratch_overwrite_before_read"
            if self.scratch_buffer_init_effective
            else (
                "policy_unset_zero_fill"
                if self.scratch_buffer_init_policy_env == "unset"
                else "explicit_zero_fill_policy"
            )
        )
        self.scratch_buffer_init_safety = (
            "selected decode scratch tensors are kernel/PyTorch outputs before weighted attention reads them"
            if self.scratch_buffer_init_effective
            else "not_applicable_zero_fill"
        )
        self.scratch_buffer_uninitialized_tensor_count = 0
        self.scratch_buffer_uninitialized_bytes = 0
        self.scratch_buffer_uninitialized_by_name = {}
        self.index_metadata_prefill_residency_effective = None
        self.index_metadata_prefill_residency_reason = None
        self.index_metadata_prefill_gpu_bytes = 0
        self.index_metadata_prefill_cpu_bytes = 0
        self.index_metadata_prefill_host_pinned_bytes = 0
        self.index_metadata_late_migration_streams = {}
        self.index_metadata_late_migration_events = {}
        self.index_metadata_late_migration_pending_devices = set()
        self.index_metadata_late_migration_pending_sources = []
        self.index_metadata_late_migration_copy_count = 0
        self.index_metadata_late_migration_copy_bytes = 0
        self.index_metadata_late_migration_sync_count = 0
        self.index_metadata_late_migration_source_cpu_bytes = 0
        self.index_metadata_late_migration_source_host_pinned_bytes = 0
        self.index_metadata_late_migration_elapsed_ms = 0.0
        self.index_metadata_late_migration_copy_launch_elapsed_ms = 0.0
        self.index_metadata_late_migration_sync_wait_ms = 0.0
        self.index_metadata_late_migration_prepare_window_elapsed_ms = 0.0
        self.async_cluster_id_copy_env = os.environ.get("RETROINFER_ASYNC_CLUSTER_ID_COPY", "")
        self.async_cluster_id_copy_enabled = _env_flag("RETROINFER_ASYNC_CLUSTER_ID_COPY", False)
        self.async_cluster_id_copy_streams = {}
        self.async_cluster_id_copy_events = {}
        self.async_wave_batch_access_env = os.environ.get("RETROINFER_ASYNC_WAVE_BATCH_ACCESS", "")
        self.async_wave_batch_access_enabled = _env_flag("RETROINFER_ASYNC_WAVE_BATCH_ACCESS", False)
        self.async_wave_batch_access_executor = None
        self.async_wave_batch_access_future = None
        self.async_wave_batch_access_pending_layer = None
        self.async_wave_batch_access_pending_device = None
        self.async_wave_batch_access_launch_count = 0
        self.async_wave_batch_access_sync_count = 0
        self.async_wave_batch_access_exception_count = 0
        self.async_wave_batch_access_launch_elapsed_ms = 0.0
        self.async_wave_batch_access_join_wait_ms = 0.0
        self.async_wave_batch_access_worker_elapsed_ms = 0.0
        self.async_wave_batch_access_cluster_id_wait_ms = 0.0
        self.async_wave_batch_access_batch_access_elapsed_ms = 0.0
        self.async_wave_batch_access_last_sync_reason = "none"
        self.async_cache_admission_env = os.environ.get("RETROINFER_ASYNC_CACHE_ADMISSION", "")
        self.async_cache_admission_enabled = _env_flag("RETROINFER_ASYNC_CACHE_ADMISSION", False)
        self.async_cache_admission_streams = {}
        self.async_cache_admission_events = {}
        self.async_cache_admission_pending_devices = set()
        self.async_cache_admission_pending_layers = {}
        self.async_cache_admission_launch_count = 0
        self.async_cache_admission_sync_count = 0
        self.async_cache_admission_launch_elapsed_ms = 0.0
        self.async_cache_admission_wait_enqueue_elapsed_ms = 0.0
        self.async_cache_admission_last_sync_reason = "none"
        self.async_stream_only_gather_env = os.environ.get("RETROINFER_ASYNC_STREAM_ONLY_GATHER", "")
        self.async_stream_only_gather_requested = _env_flag("RETROINFER_ASYNC_STREAM_ONLY_GATHER", False)
        self.async_stream_only_gather_enabled = (
            self.async_stream_only_gather_requested and bool(self.stream_only_layers)
        )
        self.async_stream_only_gather_disabled_reason = (
            "enabled"
            if self.async_stream_only_gather_enabled
            else (
                "no_stream_only_layers"
                if self.async_stream_only_gather_requested
                else "env_unset_or_false"
            )
        )
        self.async_stream_only_gather_streams = {}
        self.async_stream_only_gather_events = {}
        self.async_stream_only_gather_pending_devices = set()
        self.async_stream_only_gather_pending_layers = {}
        self.async_stream_only_gather_launch_count = 0
        self.async_stream_only_gather_eager_launch_count = 0
        self.async_stream_only_gather_cudagraph_launch_count = 0
        self.async_stream_only_gather_sync_count = 0
        self.async_stream_only_gather_launch_elapsed_ms = 0.0
        self.async_stream_only_gather_wait_enqueue_elapsed_ms = 0.0
        self.async_stream_only_gather_last_sync_reason = "none"
        self.stream_only_gather_cudagraphs = (
            [torch.cuda.CUDAGraph() for _ in range(self.layer_num)]
            if self.use_cuda_graph and self.async_stream_only_gather_enabled
            else None
        )
        self.index_metadata_stage_buffers = {}
        self.index_metadata_stage_streams = {}
        self.index_metadata_stage_events = {}
        self.index_metadata_stage_loaded_layers = {}
        self.index_metadata_stage_prefetch_count = 0
        self.index_metadata_stage_sync_count = 0
        self.index_metadata_stage_copy_bytes = 0
        if self.async_cluster_id_copy_enabled:
            self.async_cluster_id_copy_streams = {
                device_idx: torch.cuda.Stream(device=device_idx)
                for device_idx in self.device_list
            }
            self.async_cluster_id_copy_events = {
                device_idx: torch.cuda.Event(blocking=False)
                for device_idx in self.device_list
            }
        if self.async_wave_batch_access_enabled:
            self.async_wave_batch_access_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="retroinfer-wave-batch-access",
            )
        if self.async_cache_admission_enabled:
            self.async_cache_admission_streams = {
                device_idx: torch.cuda.Stream(device=device_idx)
                for device_idx in self.device_list
            }
            self.async_cache_admission_events = {
                device_idx: torch.cuda.Event(blocking=False)
                for device_idx in self.device_list
            }
        if self.async_stream_only_gather_enabled:
            self.async_stream_only_gather_streams = {
                device_idx: torch.cuda.Stream(device=device_idx)
                for device_idx in self.device_list
            }
            self.async_stream_only_gather_events = {
                device_idx: torch.cuda.Event(blocking=False)
                for device_idx in self.device_list
            }
        if self.index_metadata_late_migration_policy == "pinned_side_stream":
            self.index_metadata_late_migration_streams = {
                device_idx: torch.cuda.Stream(device=device_idx)
                for device_idx in self.device_list
            }
            self.index_metadata_late_migration_events = {
                device_idx: torch.cuda.Event(blocking=False)
                for device_idx in self.device_list
            }
        print(
            f"Cache pages: {self.cache_size}, Buffer pages: {self.buffer_size}, "
            f"Buffer nprobe multiplier: {self.buffer_nprobe_multiplier:g}"
        )
        if any(cache_pages != self.cache_size for cache_pages in self.cache_sizes):
            print(
                "Layer cache residency/capacity: "
                f"residency={self.layer_cache_residency_mode}, "
                f"capacity_scale={self.layer_cache_capacity_scale_spec or 'default'}, "
                f"stream_only={self.stream_only_layers_spec or 'off'}, "
                f"active layers: {self.block_cache_active_layer_count()}/{self.layer_num}"
            )
        if self.stage_index_metadata_layers:
            print(
                "Index metadata staging: "
                f"layers={sorted(self.stage_index_metadata_layers)}, "
                "mode=host_pinned_per_device_stage_prefetch"
            )
        if self.index_metadata_late_migration_policy_env != "unset":
            print(
                "Late index metadata migration: "
                f"policy={self.index_metadata_late_migration_policy}, "
                f"env={self.index_metadata_late_migration_policy_env}, "
                f"host_pinned={self.index_metadata_late_migration_host_pinned}, "
                f"non_blocking={self.index_metadata_late_migration_non_blocking}"
            )
        if self.late_block_cache_init_policy_env != "unset":
            print(
                "Late block cache init: "
                f"policy={self.late_block_cache_init_policy}, "
                f"env={self.late_block_cache_init_policy_env}"
            )
        if self.async_wave_batch_access_env:
            print(
                "Async wave batch access: "
                f"enabled={self.async_wave_batch_access_enabled}, "
                f"env={self.async_wave_batch_access_env}, "
                f"async_cluster_id_copy={self.async_cluster_id_copy_enabled}"
            )
        if self.wave_buffer_cpu_core_override_active:
            print(
                "WaveBufferCPU core override: "
                f"config_core={self.wave_buffer_cpu_core_config_value}, "
                f"effective_core={self.wave_buffer_cpu_core_effective}, "
                f"env={self.wave_buffer_cpu_core_override_env}"
            )
        if self.buffer_pages_per_cluster_floor_active:
            print(
                "Buffer pages-per-cluster floor: "
                f"block_cache_pages_per_cluster={self.pages_per_cluster}, "
                f"floor={self.buffer_pages_per_cluster_floor}, "
                f"effective={self.buffer_pages_per_cluster}, "
                f"env={self.buffer_pages_per_cluster_floor_env}"
            )
        if self.scratch_buffer_init_policy_env != "unset":
            print(
                "Scratch buffer init: "
                f"policy={self.scratch_buffer_init_policy}, "
                f"env={self.scratch_buffer_init_policy_env}, "
                f"mode={self.scratch_buffer_init_mode}"
            )
        if self.async_cache_admission_env:
            print(
                "Async cache admission: "
                f"enabled={self.async_cache_admission_enabled}, "
                f"env={self.async_cache_admission_env}, "
                f"streams={len(self.async_cache_admission_streams)}, "
                f"use_cuda_graph={self.use_cuda_graph}"
            )
        if self.async_stream_only_gather_env:
            print(
                "Async stream-only gather: "
                f"enabled={self.async_stream_only_gather_enabled}, "
                f"env={self.async_stream_only_gather_env}, "
                f"stream_only_layers={sorted(self.stream_only_layers)}, "
                f"streams={len(self.async_stream_only_gather_streams)}, "
                f"use_cuda_graph={self.use_cuda_graph}, "
                f"reason={self.async_stream_only_gather_disabled_reason}"
            )

        # whether to pre-allocate GPU cache and buffer before prefilling
        self.allocated = self.pre_allocate_decision()
        self.block_cache_preallocated_before_prefill = self.allocated
        if self.allocated and self.late_block_cache_init_policy == "uninitialized":
            self.block_cache_late_init_reason = "not_applied_to_preallocated_block_cache"
        if self.block_cache_slot_rotation_enabled and self.allocated:
            raise RuntimeError("Slot rotation must not preallocate per-layer GPU block-cache tensors")
        if self.allocated:
            self.index_metadata_prefill_residency_effective = "gpu"
            self.index_metadata_prefill_residency_reason = "block_cache_preallocated_before_prefill"
        elif self.index_metadata_prefill_residency == "gpu":
            self.index_metadata_prefill_residency_effective = "gpu"
            self.index_metadata_prefill_residency_reason = "split_late_index_metadata_gpu_prefill"
        else:
            self.index_metadata_prefill_residency_effective = "cpu"
            self.index_metadata_prefill_residency_reason = (
                "late_pinned_cpu_metadata_until_prepare_cache"
                if self.index_metadata_late_migration_host_pinned
                else "late_cpu_metadata_until_prepare_cache"
            )
        if not self.allocated or self.index_metadata_prefill_residency_env != "unset":
            print(
                "Index metadata prefill residency: "
                f"env={self.index_metadata_prefill_residency_env}, "
                f"effective={self.index_metadata_prefill_residency_effective}, "
                f"reason={self.index_metadata_prefill_residency_reason}"
            )

        # initialize thread pool
        self.thread_pool = ThreadPool(self.wave_buffer_cpu_core_effective)
        thread_pool_pointer = self.thread_pool.get()
        # initialize the Wave Buffer
        self.wave_buffer = [WaveBufferCPU(
            self.batch_size, self.kv_head, self.head_dim, self.nprobe, self.nprobe_new, self.page_size, 
            self.n_centroids+self.n_centroids_new, self.buffer_size, self.cache_sizes[ldx], self.wave_buffer_cpu_core_effective, thread_pool_pointer)
            for ldx in range(self.layer_num)
        ]

        # pin memory for hit cluster indices (unit == page)
        self.hit_unit_idices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_unit_sizes_cumsum = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.hit_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for missing cluster indices (unit == page)
        self.miss_unit_idices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_unit_sizes_cumsum = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.miss_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # pin memory for cache update cluster indices (unit == page)
        self.update_buffer_indices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_unit_sizes = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_cache_indices = [
            torch.zeros((self.batch_groups, self.buffer_size), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        self.update_num_units = [
            torch.zeros((self.batch_groups), dtype=torch.int32, pin_memory=True).contiguous()
            for _ in range(self.layer_num)
        ]
        # store searched TopK cluster IDs
        self.cluster_ids = torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            self.wave_buffer[ldx].set_indices(
                self.hit_unit_idices[ldx], self.hit_unit_sizes[ldx], self.hit_unit_sizes_cumsum[ldx], self.hit_num_units[ldx],
                self.miss_unit_idices[ldx], self.miss_unit_sizes[ldx], self.miss_unit_sizes_cumsum[ldx], self.miss_num_units[ldx],
                self.update_buffer_indices[ldx], self.update_unit_sizes[ldx], self.update_cache_indices[ldx], self.update_num_units[ldx], 
                self.cluster_ids
            )

        if self.allocated:  # allocate GPU block cache and meta index
            self.cache_keys, self.cache_values = [], []
            self.centroids, self.value_sum, self.centroids_mask, self.cluster_size = [], [], [], []
            self.cluster_size_cumsum = []
            for ldx in range(self.layer_num):
                self.cache_keys.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_sizes[ldx], self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.cache_values.append(
                    torch.zeros((self.batch_size, self.kv_head, self.cache_sizes[ldx], self.page_size, self.head_dim),
                                dtype=self.dtype, device=self.layer_mapping[str(ldx)]).contiguous()
                )
                self.centroids.append(
                    self._new_index_metadata_tensor(
                        ldx,
                        (self.batch_size*self.kv_head, self.n_centroids, self.head_dim),
                        self.dtype,
                    )
                )
                self.value_sum.append(
                    self._new_index_metadata_tensor(
                        ldx,
                        (self.batch_size*self.kv_head, self.n_centroids, self.head_dim),
                        self.dtype,
                    )
                )
                self.centroids_mask.append(
                    self._new_index_metadata_tensor(
                        ldx,
                        (self.batch_size*self.kv_head, self.n_centroids),
                        torch.bool,
                    )
                )
                self.cluster_size.append(
                    self._new_index_metadata_tensor(
                        ldx,
                        (self.batch_size*self.kv_head, self.n_centroids),
                        self.dtype,
                    )
                )
                if self._is_stream_only_layer(ldx):
                    self.cluster_size_cumsum.append(
                        self._new_index_metadata_tensor(
                            ldx,
                            (self.batch_size*self.kv_head, self.n_centroids),
                            dtype=torch.int32,
                        )
                    )
                else:
                    self.cluster_size_cumsum.append(None)
            self.cache_stride = self.cache_size
            self.allocate_computation_buffer()
        else:  # allocate meta index for prefilling; default remains CPU, optional split-late mode uses GPU
            force_cpu_metadata = self.index_metadata_prefill_residency_effective != "gpu"
            force_pin_cpu_metadata = (
                force_cpu_metadata and self.index_metadata_late_migration_host_pinned
            )
            self.centroids = [
                self._new_index_metadata_tensor(
                    ldx,
                    (self.batch_size*self.kv_head, self.n_centroids, self.head_dim),
                    self.dtype,
                    force_cpu=force_cpu_metadata,
                    force_pin_cpu=force_pin_cpu_metadata,
                )
                for ldx in range(self.layer_num)
            ]
            self.value_sum = [
                self._new_index_metadata_tensor(
                    ldx,
                    (self.batch_size*self.kv_head, self.n_centroids, self.head_dim),
                    self.dtype,
                    force_cpu=force_cpu_metadata,
                    force_pin_cpu=force_pin_cpu_metadata,
                )
                for ldx in range(self.layer_num)
            ]
            self.centroids_mask = [
                self._new_index_metadata_tensor(
                    ldx,
                    (self.batch_size*self.kv_head, self.n_centroids),
                    torch.bool,
                    force_cpu=force_cpu_metadata,
                    force_pin_cpu=force_pin_cpu_metadata,
                )
                for ldx in range(self.layer_num)
            ]
            self.cluster_size = [
                self._new_index_metadata_tensor(
                    ldx,
                    (self.batch_size*self.kv_head, self.n_centroids),
                    self.dtype,
                    force_cpu=force_cpu_metadata,
                    force_pin_cpu=force_pin_cpu_metadata,
                )
                for ldx in range(self.layer_num)
            ]
            self.cluster_size_cumsum = [
                (
                    self._new_index_metadata_tensor(
                        ldx,
                        (self.batch_size*self.kv_head, self.n_centroids),
                        dtype=torch.int32,
                        force_cpu=force_cpu_metadata,
                        force_pin_cpu=force_pin_cpu_metadata,
                    )
                    if self._is_stream_only_layer(ldx)
                    else None
                )
                for ldx in range(self.layer_num)
            ]
        (
            self.index_metadata_prefill_gpu_bytes,
            self.index_metadata_prefill_cpu_bytes,
            self.index_metadata_prefill_host_pinned_bytes,
        ) = self._current_index_metadata_residency_bytes()

        # layer-share cpu pin memory, transfer gpu keys & values to cpu for segmented clustering
        if self.build_index_when_prefilling:
            self.offload_keys = torch.empty(
                (self.prefill_bsz*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
                dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_values = torch.empty(
                (self.prefill_bsz*self.kv_head, self.input_length-self.static_pattern_total, self.head_dim), 
                dtype=self.dtype, pin_memory=True
            ).contiguous()

        # layer-share cpu pin memory, offload update keys & values to cpu for segmented clustering
        if self.will_update_index:
            self.offload_update_keys = torch.empty(
                (self.batch_size*self.kv_head, self.UPDATE_SEGMENT, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()
            self.offload_update_values = torch.empty(
                (self.batch_size*self.kv_head, self.UPDATE_SEGMENT, self.head_dim), dtype=self.dtype, pin_memory=True
            ).contiguous()

        # allocate cpu pin memory to store organized keys & values
        self.list_keys, self.list_values = [], []
        for _ in range(self.layer_num):
            self.list_keys.append(
                torch.empty((self.batch_size, self.kv_head, self.list_stride, self.head_dim), 
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
            self.list_values.append(
                torch.empty((self.batch_size, self.kv_head, self.list_stride, self.head_dim),
                            dtype=self.dtype, pin_memory=True).contiguous()
            )
        
        # set keys & values pointers in the wave buffer
        for ldx in range(self.layer_num):
            if self.build_index_when_prefilling:
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_keys, self.offload_values)
            elif self.will_update_index:
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.offload_update_keys, self.offload_update_values)
            else:
                self.placeholder = torch.empty((self.kv_head, 0, self.head_dim), dtype=self.dtype, pin_memory=True)
                self.wave_buffer[ldx].set_kv(self.list_keys[ldx], self.list_values[ldx], self.placeholder, self.placeholder)

        # create multi-streams and events for async offloading
        self.copystream = torch.cuda.Stream()
        self.mainevents = {}
        self.copyevents = {}
        for device_idx in self.device_list:
            with torch.cuda.device(device_idx):
                self.mainevents[device_idx] = torch.cuda.Event()
                self.copyevents[device_idx] = torch.cuda.Event()
        
        # set decoding attention function
        self.attn_func = self.dense_attention
    

    def _resolve_layer_cache_sizes(self, cache_pages: int) -> list[int]:
        """Resolve optional per-layer block-cache residency from RETROINFER_LAYER_CACHE_RESIDENCY."""
        mode = self.layer_cache_residency_mode.strip().lower()
        all_layers = set(range(self.layer_num))
        if mode in {"all", "default"}:
            active_layers = all_layers
        elif mode in {"none", "off", "cpu-only", "cpu_only"}:
            active_layers = set()
        elif mode == "even":
            active_layers = {layer_idx for layer_idx in range(self.layer_num) if layer_idx % 2 == 0}
        elif mode == "odd":
            active_layers = {layer_idx for layer_idx in range(self.layer_num) if layer_idx % 2 == 1}
        elif mode in {"first-half", "first_half"}:
            active_layers = set(range((self.layer_num + 1) // 2))
        elif mode in {"last-half", "last_half"}:
            active_layers = set(range(self.layer_num // 2, self.layer_num))
        elif mode.startswith("first:"):
            count = int(mode.split(":", 1)[1])
            if count < 0 or count > self.layer_num:
                raise ValueError(f"RETROINFER_LAYER_CACHE_RESIDENCY first count must be in [0, {self.layer_num}]")
            active_layers = set(range(count))
        elif mode.startswith("last:"):
            count = int(mode.split(":", 1)[1])
            if count < 0 or count > self.layer_num:
                raise ValueError(f"RETROINFER_LAYER_CACHE_RESIDENCY last count must be in [0, {self.layer_num}]")
            active_layers = set(range(self.layer_num - count, self.layer_num))
        elif mode.startswith("layers:"):
            active_layers = _parse_layer_range_list(mode.split(":", 1)[1], self.layer_num)
        else:
            raise ValueError(
                "RETROINFER_LAYER_CACHE_RESIDENCY must be one of all/default, none/off/cpu-only, "
                "even, odd, first-half, last-half, first:N, last:N, or layers:0,2,4-7"
            )
        return [cache_pages if layer_idx in active_layers else 0 for layer_idx in range(self.layer_num)]


    def _apply_layer_cache_capacity_scales(
        self,
        cache_sizes: list[int],
        nominal_cache_pages: int,
    ) -> tuple[list[int], list[float]]:
        """Apply optional cluster-aligned per-layer capacity scales."""
        scale_overrides = _parse_layer_capacity_scale_spec(self.layer_cache_capacity_scale_spec, self.layer_num)
        resolved_cache_sizes: list[int] = []
        effective_scales: list[float] = []
        for layer_idx, cache_pages in enumerate(cache_sizes):
            if cache_pages % self.pages_per_cluster != 0:
                raise ValueError(
                    f"Layer {layer_idx} cache pages({cache_pages}) must be divisible by "
                    f"pages_per_cluster({self.pages_per_cluster})"
                )
            scale = scale_overrides.get(layer_idx, 1.0)
            if layer_idx in scale_overrides and cache_pages == 0:
                raise ValueError(
                    "RETROINFER_LAYER_CACHE_CAPACITY_SCALE cannot target layer "
                    f"{layer_idx} because RETROINFER_LAYER_CACHE_RESIDENCY or "
                    "RETROINFER_STREAM_ONLY_LAYERS disables its block cache"
                )
            if cache_pages > 0 and scale < 1.0:
                cache_clusters = cache_pages // self.pages_per_cluster
                scaled_clusters = max(1, round(cache_clusters * scale))
                cache_pages = scaled_clusters * self.pages_per_cluster
            resolved_cache_sizes.append(cache_pages)
            effective_scales.append(cache_pages / nominal_cache_pages if nominal_cache_pages > 0 else 0.0)
        return resolved_cache_sizes, effective_scales


    def block_cache_active_layer_count(self):
        return sum(cache_pages > 0 for cache_pages in self.cache_sizes)


    def _host_memory_facts(self):
        facts = {
            "mem_total_kb": None,
            "mem_available_kb": None,
            "memlock_soft_bytes": None,
            "memlock_hard_bytes": None,
        }
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as meminfo:
                for line in meminfo:
                    if line.startswith("MemTotal:"):
                        facts["mem_total_kb"] = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        facts["mem_available_kb"] = int(line.split()[1])
        except OSError:
            pass
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            facts["memlock_soft_bytes"] = None if soft == resource.RLIM_INFINITY else int(soft)
            facts["memlock_hard_bytes"] = None if hard == resource.RLIM_INFINITY else int(hard)
        except (OSError, ValueError):
            pass
        return facts


    def _validate_block_cache_slot_rotation_configuration(self):
        if self.num_gpus != 1 or len(self.device_list) != 1:
            raise RuntimeError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION currently supports the audited single-GPU path only"
            )
        if self.cache_size <= 0:
            raise RuntimeError("RETROINFER_BLOCK_CACHE_SLOT_ROTATION requires a positive block-cache capacity")
        if self.layer_cache_residency_mode.strip().lower() not in {"all", "default"}:
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION cannot be combined with "
                "RETROINFER_LAYER_CACHE_RESIDENCY; CPU full-KV rotation requires all logical layers"
            )
        if self.layer_cache_capacity_scale_spec.strip():
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION cannot be combined with "
                "RETROINFER_LAYER_CACHE_CAPACITY_SCALE; static per-layer capacity scaling is out of scope"
            )
        if self.stream_only_layers:
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION cannot be combined with "
                "RETROINFER_STREAM_ONLY_LAYERS; direct-gather stream-only layers are out of scope"
            )
        if self.stage_index_metadata_layers:
            raise ValueError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION cannot be combined with "
                "RETROINFER_STAGE_INDEX_METADATA_LAYERS; this path must prove K/V body rotation directly"
            )
        if any(cache_pages != self.cache_size for cache_pages in self.cache_sizes):
            raise RuntimeError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION requires identical full logical block-cache "
                "capacity for every layer"
            )
        self.block_cache_slot_rotation_actual_slots = min(
            self.block_cache_slot_rotation_requested_slots,
            self.layer_num,
        )
        if self.block_cache_slot_rotation_actual_slots <= 0:
            raise RuntimeError("RETROINFER_BLOCK_CACHE_GPU_SLOTS resolved to zero physical slots")
        self.block_cache_slot_rotation_cuda_graph_mode = (
            "fixed_slot_address_graph" if self.use_cuda_graph else "eager_fixed_slot_addresses"
        )


    def _is_stream_only_layer(self, layer_idx):
        return self.stream_only_layer_mask[layer_idx]


    def _is_index_metadata_staged_layer(self, layer_idx):
        return self.stage_index_metadata_layer_mask[layer_idx]


    def _new_index_metadata_tensor(
        self,
        layer_idx,
        shape,
        dtype: torch.dtype,
        force_cpu: bool = False,
        force_pin_cpu: bool = False,
    ):
        pin_cpu = self._is_index_metadata_staged_layer(layer_idx) or force_pin_cpu
        if force_cpu or pin_cpu:
            return _cpu_zeros(shape, dtype=dtype, pin_memory=pin_cpu)
        return _gpu_zeros(shape, dtype=dtype, device=self.layer_mapping[str(layer_idx)])


    def _pin_if_staged(self, layer_idx, tensor):
        tensor = tensor.contiguous()
        if self._is_index_metadata_staged_layer(layer_idx) and tensor.device.type == "cpu" and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        return tensor


    def _init_block_cache_telemetry(self):
        if not self.block_cache_telemetry_enabled:
            return None
        return {
            "layer_stats": [
                {
                    "layer_idx": layer_idx,
                    "calls": 0,
                    "hit_num_units": 0,
                    "miss_num_units": 0,
                    "update_num_units": 0,
                }
                for layer_idx in range(self.layer_num)
            ]
        }


    def _record_block_cache_telemetry(self, layer_idx):
        if not self.block_cache_telemetry_enabled:
            return
        stats = self.block_cache_telemetry["layer_stats"][layer_idx]
        stats["calls"] += 1
        stats["hit_num_units"] += int(self.hit_num_units[layer_idx].sum().item())
        stats["miss_num_units"] += int(self.miss_num_units[layer_idx].sum().item())
        stats["update_num_units"] += int(self.update_num_units[layer_idx].sum().item())


    def _block_cache_telemetry_metadata(self):
        if not self.block_cache_telemetry_enabled:
            return {"block_cache_telemetry_enabled": False}
        layer_stats = []
        totals = {
            "calls": 0,
            "hit_num_units": 0,
            "miss_num_units": 0,
            "update_num_units": 0,
        }
        for stats in self.block_cache_telemetry["layer_stats"]:
            total_access_units = stats["hit_num_units"] + stats["miss_num_units"]
            layer_summary = {
                **stats,
                "access_num_units": total_access_units,
                "hit_rate_units": (
                    stats["hit_num_units"] / total_access_units if total_access_units > 0 else None
                ),
                "miss_rate_units": (
                    stats["miss_num_units"] / total_access_units if total_access_units > 0 else None
                ),
                "update_per_miss_units": (
                    stats["update_num_units"] / stats["miss_num_units"] if stats["miss_num_units"] > 0 else None
                ),
            }
            layer_stats.append(layer_summary)
            for key in totals:
                totals[key] += stats[key]

        total_access_units = totals["hit_num_units"] + totals["miss_num_units"]
        totals["access_num_units"] = total_access_units
        totals["hit_rate_units"] = totals["hit_num_units"] / total_access_units if total_access_units > 0 else None
        totals["miss_rate_units"] = totals["miss_num_units"] / total_access_units if total_access_units > 0 else None
        totals["update_per_miss_units"] = (
            totals["update_num_units"] / totals["miss_num_units"] if totals["miss_num_units"] > 0 else None
        )
        return {
            "block_cache_telemetry_enabled": True,
            "block_cache_telemetry_totals": totals,
            "block_cache_telemetry_layers": layer_stats,
        }


    def _launch_cluster_ids_copy(self, source, device_idx):
        if not self.async_cluster_id_copy_enabled:
            self.cluster_ids.copy_(source)
            return

        copy_stream = self.async_cluster_id_copy_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        copy_stream.wait_stream(current_stream)
        with torch.cuda.stream(copy_stream):
            self.cluster_ids.copy_(source, non_blocking=True)
            self.async_cluster_id_copy_events[device_idx].record(copy_stream)


    def _sync_cluster_ids_copy(self, device_idx):
        if self.async_cluster_id_copy_enabled:
            self.async_cluster_id_copy_events[device_idx].synchronize()


    def _async_cluster_id_copy_metadata(self):
        return {
            "async_cluster_id_copy_enabled": self.async_cluster_id_copy_enabled,
            "async_cluster_id_copy_env": self.async_cluster_id_copy_env or "unset",
            "async_cluster_id_copy_mode": (
                "separate_stream_d2h_non_blocking"
                if self.async_cluster_id_copy_enabled
                else "blocking_default_stream_copy"
            ),
            "async_cluster_id_copy_stream_count": len(self.async_cluster_id_copy_streams),
            "async_cluster_id_copy_destination": "pinned_cpu_cluster_ids",
            "async_cluster_id_copy_sync_point": "before_wave_buffer_batch_access",
            "async_cluster_id_copy_overlap_window": "estimation_zone_gather_and_weighted_decoding",
        }


    def _run_async_wave_batch_access_worker(self, layer_idx, device_idx):
        worker_start = time.perf_counter()
        wait_start = worker_start
        self._sync_cluster_ids_copy(device_idx)
        cluster_id_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        batch_start = time.perf_counter()
        self.wave_buffer[layer_idx].batch_access()
        batch_access_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
        worker_elapsed_ms = (time.perf_counter() - worker_start) * 1000.0
        return {
            "cluster_id_wait_ms": cluster_id_wait_ms,
            "batch_access_elapsed_ms": batch_access_elapsed_ms,
            "worker_elapsed_ms": worker_elapsed_ms,
        }


    def _launch_async_wave_batch_access(self, layer_idx, device_idx):
        if not self.async_wave_batch_access_enabled:
            return
        self._sync_async_wave_batch_access(
            "before_new_async_wave_batch_access_same_executor"
        )
        if self.async_wave_batch_access_executor is None:
            raise RuntimeError(
                "RETROINFER_ASYNC_WAVE_BATCH_ACCESS is enabled but no executor was initialized"
            )
        launch_start = time.perf_counter()
        self.async_wave_batch_access_future = self.async_wave_batch_access_executor.submit(
            self._run_async_wave_batch_access_worker,
            layer_idx,
            device_idx,
        )
        self.async_wave_batch_access_launch_elapsed_ms += (
            time.perf_counter() - launch_start
        ) * 1000.0
        self.async_wave_batch_access_launch_count += 1
        self.async_wave_batch_access_pending_layer = layer_idx
        self.async_wave_batch_access_pending_device = device_idx


    def _sync_async_wave_batch_access(self, reason: str):
        if not self.async_wave_batch_access_enabled:
            return
        future = self.async_wave_batch_access_future
        if future is None:
            return
        join_start = time.perf_counter()
        try:
            timing = future.result()
        except Exception:
            self.async_wave_batch_access_exception_count += 1
            self.async_wave_batch_access_future = None
            self.async_wave_batch_access_pending_layer = None
            self.async_wave_batch_access_pending_device = None
            raise
        self.async_wave_batch_access_join_wait_ms += (
            time.perf_counter() - join_start
        ) * 1000.0
        self.async_wave_batch_access_worker_elapsed_ms += timing["worker_elapsed_ms"]
        self.async_wave_batch_access_cluster_id_wait_ms += timing["cluster_id_wait_ms"]
        self.async_wave_batch_access_batch_access_elapsed_ms += timing["batch_access_elapsed_ms"]
        self.async_wave_batch_access_sync_count += 1
        self.async_wave_batch_access_last_sync_reason = reason
        self.async_wave_batch_access_future = None
        self.async_wave_batch_access_pending_layer = None
        self.async_wave_batch_access_pending_device = None


    def _finish_wave_batch_access(self, layer_idx, device_idx, reason: str):
        if self.async_wave_batch_access_enabled and self.async_wave_batch_access_future is not None:
            self._sync_async_wave_batch_access(reason)
            return
        self._sync_cluster_ids_copy(device_idx)
        self.wave_buffer[layer_idx].batch_access()


    def _async_wave_batch_access_metadata(self):
        return {
            "async_wave_batch_access_enabled": self.async_wave_batch_access_enabled,
            "async_wave_batch_access_env": self.async_wave_batch_access_env or "unset",
            "async_wave_batch_access_mode": (
                "python_thread_event_wait_then_wave_buffer_batch_access"
                if self.async_wave_batch_access_enabled
                else "main_thread_inline_after_estimation_zone"
            ),
            "async_wave_batch_access_thread_count": (
                1 if self.async_wave_batch_access_executor is not None else 0
            ),
            "async_wave_batch_access_launch_count": self.async_wave_batch_access_launch_count,
            "async_wave_batch_access_sync_count": self.async_wave_batch_access_sync_count,
            "async_wave_batch_access_exception_count": self.async_wave_batch_access_exception_count,
            "async_wave_batch_access_pending_count": (
                1 if self.async_wave_batch_access_future is not None else 0
            ),
            "async_wave_batch_access_pending_layer": self.async_wave_batch_access_pending_layer,
            "async_wave_batch_access_pending_device": self.async_wave_batch_access_pending_device,
            "async_wave_batch_access_launch_elapsed_ms": self.async_wave_batch_access_launch_elapsed_ms,
            "async_wave_batch_access_join_wait_ms": self.async_wave_batch_access_join_wait_ms,
            "async_wave_batch_access_worker_elapsed_ms": self.async_wave_batch_access_worker_elapsed_ms,
            "async_wave_batch_access_cluster_id_wait_ms": self.async_wave_batch_access_cluster_id_wait_ms,
            "async_wave_batch_access_batch_access_elapsed_ms": (
                self.async_wave_batch_access_batch_access_elapsed_ms
            ),
            "async_wave_batch_access_cluster_id_copy_async": self.async_cluster_id_copy_enabled,
            "async_wave_batch_access_launch_point": (
                "after_topk_cluster_id_copy_launch_before_estimation_zone"
                if self.async_wave_batch_access_enabled
                else "not_applicable"
            ),
            "async_wave_batch_access_sync_point": (
                "before_gather_copy_and_concat_reads_wave_buffer_indices"
                if self.async_wave_batch_access_enabled
                else "inline_batch_access_before_gather_copy_and_concat"
            ),
            "async_wave_batch_access_overlap_window": (
                "estimation_zone_gather_and_weighted_decoding"
                if self.async_wave_batch_access_enabled
                else "none"
            ),
            "async_wave_batch_access_last_sync_reason": self.async_wave_batch_access_last_sync_reason,
            "async_wave_batch_access_path_executed": self.async_wave_batch_access_launch_count > 0,
        }


    def _sync_async_cache_admission_for_device(self, device_idx, reason: str):
        if not self.async_cache_admission_enabled:
            return
        if device_idx not in self.async_cache_admission_pending_devices:
            return

        wait_start = time.perf_counter()
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        current_stream.wait_event(self.async_cache_admission_events[device_idx])
        self.async_cache_admission_wait_enqueue_elapsed_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0
        self.async_cache_admission_sync_count += 1
        self.async_cache_admission_last_sync_reason = reason
        self.async_cache_admission_pending_devices.discard(device_idx)
        self.async_cache_admission_pending_layers.pop(device_idx, None)


    def _sync_all_async_cache_admissions(self, reason: str):
        if not self.async_cache_admission_enabled:
            return
        for device_idx in sorted(list(self.async_cache_admission_pending_devices)):
            self._sync_async_cache_admission_for_device(device_idx, reason)


    def _sync_pending_admission_before_execution_buffer_read(self, layer_idx, reason: str):
        device_idx = self.layer_mapping[str(layer_idx)]
        self._sync_async_cache_admission_for_device(device_idx, reason)


    def _run_cache_admission_update(self, layer_idx, use_cuda_graph_update: bool):
        if use_cuda_graph_update:
            self.update_cudagraphs[layer_idx].replay()
            return
        gather_copy_and_scatter(
            self.execution_buffer_keys, self.cache_keys[layer_idx],
            self.execution_buffer_values, self.cache_values[layer_idx],
            self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx],
            self.update_cache_indices[layer_idx], self.update_num_units[layer_idx],
            self.batch_groups, self.execution_stride, self.cache_strides[layer_idx], self.buffer_size,
            self.static_len_tensor
        )


    def _launch_cache_admission_update(self, layer_idx, use_cuda_graph_update: bool):
        if not self.async_cache_admission_enabled:
            self._run_cache_admission_update(layer_idx, use_cuda_graph_update)
            return

        device_idx = self.layer_mapping[str(layer_idx)]
        self._sync_async_cache_admission_for_device(
            device_idx,
            "before_new_async_cache_admission_same_device",
        )
        launch_start = time.perf_counter()
        admission_stream = self.async_cache_admission_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        admission_stream.wait_stream(current_stream)
        with torch.cuda.stream(admission_stream):
            self._run_cache_admission_update(layer_idx, use_cuda_graph_update)
            self.async_cache_admission_events[device_idx].record(admission_stream)
        self.async_cache_admission_launch_elapsed_ms += (
            time.perf_counter() - launch_start
        ) * 1000.0
        self.async_cache_admission_launch_count += 1
        self.async_cache_admission_pending_devices.add(device_idx)
        self.async_cache_admission_pending_layers[device_idx] = layer_idx


    def _async_cache_admission_metadata(self):
        return {
            "async_cache_admission_enabled": self.async_cache_admission_enabled,
            "async_cache_admission_env": self.async_cache_admission_env or "unset",
            "async_cache_admission_mode": (
                "side_stream_gather_copy_and_scatter"
                if self.async_cache_admission_enabled
                else "default_stream_inline"
            ),
            "async_cache_admission_stream_count": len(self.async_cache_admission_streams),
            "async_cache_admission_launch_count": self.async_cache_admission_launch_count,
            "async_cache_admission_sync_count": self.async_cache_admission_sync_count,
            "async_cache_admission_pending_device_count": len(self.async_cache_admission_pending_devices),
            "async_cache_admission_pending_layers": {
                str(device_idx): layer_idx
                for device_idx, layer_idx in self.async_cache_admission_pending_layers.items()
            },
            "async_cache_admission_launch_elapsed_ms": self.async_cache_admission_launch_elapsed_ms,
            "async_cache_admission_wait_enqueue_elapsed_ms": (
                self.async_cache_admission_wait_enqueue_elapsed_ms
            ),
            "async_cache_admission_sync_mode": (
                "current_stream_wait_event"
                if self.async_cache_admission_enabled
                else "not_applicable"
            ),
            "async_cache_admission_sync_point": (
                "before_next_execution_buffer_reuse_or_same_layer_cache_read"
                if self.async_cache_admission_enabled
                else "inline_update_before_return"
            ),
            "async_cache_admission_overlap_window": (
                "next_layer_topk_estimation_and_cpu_wave_buffer_batch_access"
                if self.async_cache_admission_enabled
                else "none"
            ),
            "async_cache_admission_last_sync_reason": self.async_cache_admission_last_sync_reason,
            "async_cache_admission_uses_cuda_graph_update": bool(self.use_cuda_graph),
        }


    def _sync_async_stream_only_gather_for_device(self, device_idx, reason: str):
        if not self.async_stream_only_gather_enabled:
            return
        if device_idx not in self.async_stream_only_gather_pending_devices:
            return

        wait_start = time.perf_counter()
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        current_stream.wait_event(self.async_stream_only_gather_events[device_idx])
        self.async_stream_only_gather_wait_enqueue_elapsed_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0
        self.async_stream_only_gather_sync_count += 1
        self.async_stream_only_gather_last_sync_reason = reason
        self.async_stream_only_gather_pending_devices.discard(device_idx)
        self.async_stream_only_gather_pending_layers.pop(device_idx, None)


    def _sync_all_async_stream_only_gathers(self, reason: str):
        if not self.async_stream_only_gather_enabled:
            return
        for device_idx in sorted(list(self.async_stream_only_gather_pending_devices)):
            self._sync_async_stream_only_gather_for_device(device_idx, reason)


    def _run_stream_only_execution_buffer_gather(
        self,
        layer_idx,
        cluster_size_cumsum,
        use_cuda_graph_gather: bool,
    ):
        if use_cuda_graph_gather:
            if self.stream_only_gather_cudagraphs is None:
                raise RuntimeError(
                    "RETROINFER_ASYNC_STREAM_ONLY_GATHER requested CUDA graph gather replay "
                    "before stream-only gather graphs were captured"
                )
            self.stream_only_gather_cudagraphs[layer_idx].replay()
            return

        gather_copy_cluster_and_concat_fuse(
            self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.execution_buffer_keys,
            self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.execution_buffer_values,
            cluster_size_cumsum, self.cI, self.valid_lengths,
            self.batch_groups, self.static_stride, self.list_stride, self.execution_stride,
            self.nprobe, self.nprobe_tensor, self.static_len_tensor
        )


    def _launch_stream_only_execution_buffer_gather(
        self,
        layer_idx,
        cluster_size_cumsum,
        use_cuda_graph_gather: bool,
    ):
        self._sync_pending_admission_before_execution_buffer_read(
            layer_idx,
            "before_stream_only_execution_buffer_write",
        )
        if not self.async_stream_only_gather_enabled:
            self._run_stream_only_execution_buffer_gather(
                layer_idx,
                cluster_size_cumsum,
                use_cuda_graph_gather,
            )
            return

        device_idx = self.layer_mapping[str(layer_idx)]
        self._sync_async_stream_only_gather_for_device(
            device_idx,
            "before_new_async_stream_only_gather_same_device",
        )
        launch_start = time.perf_counter()
        gather_stream = self.async_stream_only_gather_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        gather_stream.wait_stream(current_stream)
        with torch.cuda.stream(gather_stream):
            self._run_stream_only_execution_buffer_gather(
                layer_idx,
                cluster_size_cumsum,
                use_cuda_graph_gather,
            )
            self.async_stream_only_gather_events[device_idx].record(gather_stream)
        self.async_stream_only_gather_launch_elapsed_ms += (
            time.perf_counter() - launch_start
        ) * 1000.0
        self.async_stream_only_gather_launch_count += 1
        if use_cuda_graph_gather:
            self.async_stream_only_gather_cudagraph_launch_count += 1
        else:
            self.async_stream_only_gather_eager_launch_count += 1
        self.async_stream_only_gather_pending_devices.add(device_idx)
        self.async_stream_only_gather_pending_layers[device_idx] = layer_idx


    def _async_stream_only_gather_metadata(self):
        return {
            "async_stream_only_gather_enabled": self.async_stream_only_gather_enabled,
            "async_stream_only_gather_requested": self.async_stream_only_gather_requested,
            "async_stream_only_gather_env": self.async_stream_only_gather_env or "unset",
            "async_stream_only_gather_disabled_reason": (
                self.async_stream_only_gather_disabled_reason
            ),
            "async_stream_only_gather_mode": (
                "side_stream_gather_copy_cluster_and_concat_fuse"
                if self.async_stream_only_gather_enabled
                else "default_stream_inline"
            ),
            "async_stream_only_gather_stream_count": len(self.async_stream_only_gather_streams),
            "async_stream_only_gather_launch_count": self.async_stream_only_gather_launch_count,
            "async_stream_only_gather_eager_launch_count": (
                self.async_stream_only_gather_eager_launch_count
            ),
            "async_stream_only_gather_cudagraph_launch_count": (
                self.async_stream_only_gather_cudagraph_launch_count
            ),
            "async_stream_only_gather_sync_count": self.async_stream_only_gather_sync_count,
            "async_stream_only_gather_pending_device_count": (
                len(self.async_stream_only_gather_pending_devices)
            ),
            "async_stream_only_gather_pending_layers": {
                str(device_idx): layer_idx
                for device_idx, layer_idx in self.async_stream_only_gather_pending_layers.items()
            },
            "async_stream_only_gather_launch_elapsed_ms": (
                self.async_stream_only_gather_launch_elapsed_ms
            ),
            "async_stream_only_gather_wait_enqueue_elapsed_ms": (
                self.async_stream_only_gather_wait_enqueue_elapsed_ms
            ),
            "async_stream_only_gather_sync_mode": (
                "current_stream_wait_event"
                if self.async_stream_only_gather_enabled
                else "not_applicable"
            ),
            "async_stream_only_gather_launch_point": (
                "after_topk_before_estimation_zone"
                if self.async_stream_only_gather_enabled
                else "after_estimation_zone_before_weighted_flash_decoding"
            ),
            "async_stream_only_gather_sync_point": (
                "before_weighted_flash_decoding_reads_execution_buffer"
                if self.async_stream_only_gather_enabled
                else "inline_gather_before_weighted_flash_decoding"
            ),
            "async_stream_only_gather_overlap_window": (
                "estimation_zone_gather_and_weighted_decoding"
                if self.async_stream_only_gather_enabled
                else "none"
            ),
            "async_stream_only_gather_last_sync_reason": self.async_stream_only_gather_last_sync_reason,
            "async_stream_only_gather_uses_cuda_graph": bool(
                self.use_cuda_graph and self.stream_only_gather_cudagraphs is not None
            ),
            "async_stream_only_gather_split_cuda_graph": bool(
                self.async_stream_only_gather_enabled
                and self.use_cuda_graph
                and self.stream_only_gather_cudagraphs is not None
            ),
            "async_stream_only_gather_path_executed": self.async_stream_only_gather_launch_count > 0,
        }


    def _metadata_tensor_nbytes(self, tensor):
        if tensor is None:
            return 0
        return int(tensor.numel() * tensor.element_size())


    def _index_metadata_layer_bytes(self, layer_idx):
        return (
            self._metadata_tensor_nbytes(self.centroids[layer_idx])
            + self._metadata_tensor_nbytes(self.value_sum[layer_idx])
            + self._metadata_tensor_nbytes(self.centroids_mask[layer_idx])
            + self._metadata_tensor_nbytes(self.cluster_size[layer_idx])
            + self._metadata_tensor_nbytes(self.cluster_size_cumsum[layer_idx])
        )


    def _current_index_metadata_residency_bytes(self):
        gpu_bytes = 0
        cpu_bytes = 0
        host_pinned_bytes = 0
        for layer_idx in range(self.layer_num):
            tensors = (
                self.centroids[layer_idx],
                self.value_sum[layer_idx],
                self.centroids_mask[layer_idx],
                self.cluster_size[layer_idx],
                self.cluster_size_cumsum[layer_idx],
            )
            for tensor in tensors:
                nbytes = self._metadata_tensor_nbytes(tensor)
                if nbytes == 0:
                    continue
                if tensor.device.type == "cuda":
                    gpu_bytes += nbytes
                else:
                    cpu_bytes += nbytes
                    if tensor.is_pinned():
                        host_pinned_bytes += nbytes
        return gpu_bytes, cpu_bytes, host_pinned_bytes


    def _late_index_metadata_migration_copy_mode(self):
        if self.index_metadata_late_migration_policy == "pinned_side_stream":
            return "pinned_h2d_non_blocking_side_stream"
        if self.index_metadata_late_migration_policy == "pinned_non_blocking":
            return "pinned_h2d_non_blocking_current_stream"
        if self.index_metadata_late_migration_policy == "pinned_blocking":
            return "pinned_h2d_blocking_current_stream"
        return "pageable_h2d_blocking_current_stream"


    def _late_index_metadata_migration_sync_point(self):
        if self.index_metadata_late_migration_policy == "pinned_side_stream":
            return "after_allocate_computation_buffer_before_decode"
        if self.index_metadata_late_migration_policy == "pinned_non_blocking":
            return "after_layer_metadata_copies_before_allocate_computation_buffer"
        return "copy_call"


    def _migrate_index_metadata_tensor_to_gpu(self, tensor, device_idx):
        if tensor is None:
            return None
        if tensor.device.type == "cuda":
            return tensor.contiguous()

        nbytes = self._metadata_tensor_nbytes(tensor)
        self.index_metadata_late_migration_copy_count += 1
        self.index_metadata_late_migration_copy_bytes += nbytes
        if tensor.is_pinned():
            self.index_metadata_late_migration_source_host_pinned_bytes += nbytes
        self.index_metadata_late_migration_source_cpu_bytes += nbytes

        copy_launch_start = time.perf_counter()
        if self.index_metadata_late_migration_policy == "pinned_side_stream":
            destination = torch.empty(tensor.shape, dtype=tensor.dtype, device=device_idx).contiguous()
            copy_stream = self.index_metadata_late_migration_streams[device_idx]
            current_stream = torch.cuda.current_stream(torch.device(device_idx))
            copy_stream.wait_stream(current_stream)
            self.index_metadata_late_migration_pending_sources.append(tensor)
            with torch.cuda.stream(copy_stream):
                destination.copy_(tensor, non_blocking=True)
                self.index_metadata_late_migration_events[device_idx].record(copy_stream)
            self.index_metadata_late_migration_pending_devices.add(device_idx)
            self.index_metadata_late_migration_copy_launch_elapsed_ms += (
                time.perf_counter() - copy_launch_start
            ) * 1000.0
            return destination

        non_blocking = self.index_metadata_late_migration_policy == "pinned_non_blocking"
        if non_blocking:
            self.index_metadata_late_migration_pending_sources.append(tensor)
            self.index_metadata_late_migration_pending_devices.add(device_idx)
        destination = tensor.to(device_idx, non_blocking=non_blocking).contiguous()
        self.index_metadata_late_migration_copy_launch_elapsed_ms += (
            time.perf_counter() - copy_launch_start
        ) * 1000.0
        return destination


    def _sync_late_index_metadata_migration(self):
        sync_start = time.perf_counter()
        if self.index_metadata_late_migration_policy == "pinned_side_stream":
            for device_idx in sorted(self.index_metadata_late_migration_pending_devices):
                self.index_metadata_late_migration_events[device_idx].synchronize()
                self.index_metadata_late_migration_sync_count += 1
        elif self.index_metadata_late_migration_policy == "pinned_non_blocking":
            for device_idx in sorted(self.index_metadata_late_migration_pending_devices):
                torch.cuda.synchronize(device_idx)
                self.index_metadata_late_migration_sync_count += 1
        self.index_metadata_late_migration_pending_devices.clear()
        self.index_metadata_late_migration_pending_sources.clear()
        self.index_metadata_late_migration_sync_wait_ms += (
            time.perf_counter() - sync_start
        ) * 1000.0
        self.index_metadata_late_migration_elapsed_ms = (
            self.index_metadata_late_migration_copy_launch_elapsed_ms
            + self.index_metadata_late_migration_sync_wait_ms
        )


    def _migrate_index_metadata_layer_to_gpu(self, layer_idx):
        device_idx = self.layer_mapping[str(layer_idx)]
        self.centroids[layer_idx] = self._migrate_index_metadata_tensor_to_gpu(
            self.centroids[layer_idx],
            device_idx,
        )
        self.value_sum[layer_idx] = self._migrate_index_metadata_tensor_to_gpu(
            self.value_sum[layer_idx],
            device_idx,
        )
        self.centroids_mask[layer_idx] = self._migrate_index_metadata_tensor_to_gpu(
            self.centroids_mask[layer_idx],
            device_idx,
        )
        self.cluster_size[layer_idx] = self._migrate_index_metadata_tensor_to_gpu(
            self.cluster_size[layer_idx],
            device_idx,
        )
        if self._is_stream_only_layer(layer_idx):
            self.cluster_size_cumsum[layer_idx] = self._migrate_index_metadata_tensor_to_gpu(
                self.cluster_size_cumsum[layer_idx],
                device_idx,
            )


    def _late_index_metadata_migration_metadata(self):
        return {
            "index_metadata_late_migration_policy": self.index_metadata_late_migration_policy,
            "index_metadata_late_migration_policy_env": self.index_metadata_late_migration_policy_env,
            "index_metadata_late_migration_enabled": (
                self.index_metadata_late_migration_policy != "pageable_blocking"
            ),
            "index_metadata_late_migration_host_pinned": self.index_metadata_late_migration_host_pinned,
            "index_metadata_late_migration_non_blocking": self.index_metadata_late_migration_non_blocking,
            "index_metadata_late_migration_copy_mode": self._late_index_metadata_migration_copy_mode(),
            "index_metadata_late_migration_sync_point": self._late_index_metadata_migration_sync_point(),
            "index_metadata_late_migration_stream_count": len(self.index_metadata_late_migration_streams),
            "index_metadata_late_migration_copy_count": self.index_metadata_late_migration_copy_count,
            "index_metadata_late_migration_copy_bytes": self.index_metadata_late_migration_copy_bytes,
            "index_metadata_late_migration_source_cpu_bytes": self.index_metadata_late_migration_source_cpu_bytes,
            "index_metadata_late_migration_source_host_pinned_bytes": (
                self.index_metadata_late_migration_source_host_pinned_bytes
            ),
            "index_metadata_late_migration_sync_count": self.index_metadata_late_migration_sync_count,
            "index_metadata_late_migration_elapsed_ms": self.index_metadata_late_migration_elapsed_ms,
            "index_metadata_late_migration_copy_launch_elapsed_ms": (
                self.index_metadata_late_migration_copy_launch_elapsed_ms
            ),
            "index_metadata_late_migration_sync_wait_ms": self.index_metadata_late_migration_sync_wait_ms,
            "index_metadata_late_migration_prepare_window_elapsed_ms": (
                self.index_metadata_late_migration_prepare_window_elapsed_ms
            ),
            "index_metadata_late_migration_elapsed_scope": (
                "host_copy_launch_plus_explicit_sync_wait_excludes_interleaved_prepare_allocations"
            ),
        }


    def _index_metadata_stage_buffer_bytes(self):
        total = 0
        for buffers in self.index_metadata_stage_buffers.values():
            for tensor in buffers.values():
                total += self._metadata_tensor_nbytes(tensor)
        return total


    def _index_metadata_staging_metadata(self):
        layer_bytes = [self._index_metadata_layer_bytes(layer_idx) for layer_idx in range(self.layer_num)]
        current_gpu_bytes, current_cpu_bytes, current_host_pinned_bytes = self._current_index_metadata_residency_bytes()
        staged_layers = sorted(self.stage_index_metadata_layers)
        nominal_gpu_bytes = sum(layer_bytes)
        host_pinned_bytes = sum(layer_bytes[layer_idx] for layer_idx in staged_layers)
        persistent_gpu_bytes = sum(
            layer_bytes[layer_idx]
            for layer_idx in range(self.layer_num)
            if layer_idx not in self.stage_index_metadata_layers
        )
        stage_buffer_bytes = self._index_metadata_stage_buffer_bytes()
        resident_gpu_bytes = persistent_gpu_bytes + stage_buffer_bytes
        return {
            "index_metadata_staging_enabled": bool(staged_layers),
            "index_metadata_prefill_residency": self.index_metadata_prefill_residency,
            "index_metadata_prefill_residency_env": self.index_metadata_prefill_residency_env,
            "index_metadata_prefill_residency_effective": self.index_metadata_prefill_residency_effective,
            "index_metadata_prefill_residency_reason": self.index_metadata_prefill_residency_reason,
            "index_metadata_prefill_gpu_bytes": self.index_metadata_prefill_gpu_bytes,
            "index_metadata_prefill_cpu_bytes": self.index_metadata_prefill_cpu_bytes,
            "index_metadata_prefill_host_pinned_bytes": self.index_metadata_prefill_host_pinned_bytes,
            "index_metadata_current_gpu_bytes": current_gpu_bytes,
            "index_metadata_current_cpu_bytes": current_cpu_bytes,
            "index_metadata_current_host_pinned_bytes": current_host_pinned_bytes,
            "index_metadata_staged_layers_env": self.stage_index_metadata_layers_spec or "off",
            "index_metadata_staged_layer_count": len(staged_layers),
            "index_metadata_staged_layers": staged_layers,
            "index_metadata_layer_bytes": layer_bytes,
            "index_metadata_nominal_persistent_gpu_bytes": nominal_gpu_bytes,
            "index_metadata_persistent_gpu_bytes": persistent_gpu_bytes,
            "index_metadata_stage_buffer_bytes": stage_buffer_bytes,
            "index_metadata_gpu_resident_bytes": resident_gpu_bytes,
            "index_metadata_gpu_bytes_saved_vs_nominal": nominal_gpu_bytes - resident_gpu_bytes,
            "index_metadata_host_pinned_bytes": host_pinned_bytes,
            "index_metadata_stage_stream_count": len(self.index_metadata_stage_streams),
            "index_metadata_stage_loaded_layers": {
                str(device_idx): layer_idx
                for device_idx, layer_idx in self.index_metadata_stage_loaded_layers.items()
                if layer_idx is not None
            },
            "index_metadata_stage_prefetch_count": self.index_metadata_stage_prefetch_count,
            "index_metadata_stage_sync_count": self.index_metadata_stage_sync_count,
            "index_metadata_stage_copy_bytes": self.index_metadata_stage_copy_bytes,
            "index_metadata_stage_mode": (
                "host_pinned_cpu_metadata_with_per_device_gpu_stage_buffer"
                if staged_layers
                else "persistent_gpu_metadata"
            ),
            "index_metadata_stage_copy_mode": (
                "async_h2d_non_blocking_prefetch_stream"
                if staged_layers
                else "none"
            ),
            "index_metadata_stage_sync_point": (
                "before_topk_batch_gemm_softmax"
                if staged_layers
                else "none"
            ),
            "index_metadata_stage_prefetch_policy": (
                "prefetch_next_staged_layer_after_current_layer_metadata_use"
                if staged_layers
                else "off"
            ),
            "index_metadata_cuda_graph_behavior": (
                "cuda_graph_replay_reads_fixed_stage_buffers_after_h2d_prefetch"
                if staged_layers and self.use_cuda_graph
                else (
                    "eager_sparse_attention_reads_stage_buffers_after_h2d_prefetch"
                    if staged_layers
                    else "default_persistent_metadata_graph_behavior"
                )
            ),
        }


    def pre_allocate_decision(self):
        """Decide whether to pre-allocate GPU cache and buffers before prefilling"""
        # estimate GPU memory consumption for cache and buffers
        layer_cache_vectors = sum(cache_pages * self.page_size for cache_pages in self.cache_sizes)
        if self.block_cache_slot_rotation_enabled:
            layer_cache_vectors = self.block_cache_slot_rotation_actual_slots * self.cache_size * self.page_size
        layer_index_vectors = self.layer_num * (self.n_centroids + self.static_stride)
        self.estimated_gpu_memory = 2 * self.batch_size * self.kv_head * (layer_cache_vectors + layer_index_vectors) * self.head_dim * 2
        self.estimated_gpu_memory += 2 * self.batch_size * self.kv_head * (self.buffer_size*self.page_size + self.static_stride) * self.head_dim * 2
        self.estimated_gpu_memory += 2 * self.batch_size * self.kv_head * self.es_cluster_num * self.head_dim * 2
        self.estimated_gpu_memory += 6 * self.batch_size * self.kv_head * self.group_size * self.n_centroids * 2
        self.estimated_gpu_memory /= 1024 * 1024 * 1024
        self.esitimate_gpu_memory = self.estimated_gpu_memory
        self.block_cache_preallocation_threshold_multiplier = 1.5
        self.block_cache_preallocation_threshold_gib = (
            self.estimated_gpu_memory * self.block_cache_preallocation_threshold_multiplier
        )
        auto_preallocate = self.free_memory > self.block_cache_preallocation_threshold_gib
        self.block_cache_auto_preallocate_before_prefill = auto_preallocate
        if self.block_cache_slot_rotation_enabled:
            preallocate = False
            reason = "slot_rotation_allocates_cpu_owner_and_gpu_slots_after_prefill"
        elif self.block_cache_allocation_policy == "late":
            preallocate = False
            reason = "forced_late_after_prefill"
        elif self.block_cache_allocation_policy == "preallocate":
            preallocate = True
            reason = "forced_preallocate_before_prefill"
        else:
            preallocate = auto_preallocate
            reason = (
                "auto_free_memory_above_threshold"
                if auto_preallocate
                else "auto_free_memory_not_above_threshold"
            )
        self.block_cache_allocation_decision = (
            "preallocate_before_prefill" if preallocate else "allocate_after_prefill"
        )
        self.block_cache_allocation_decision_reason = reason
        print(
            "Block cache allocation: "
            f"policy={self.block_cache_allocation_policy}, "
            f"env={self.block_cache_allocation_policy_env}, "
            f"decision={self.block_cache_allocation_decision}, "
            f"free_memory={self.free_memory:.4f} GiB, "
            f"estimated_gpu_memory={self.estimated_gpu_memory:.4f} GiB, "
            f"threshold={self.block_cache_preallocation_threshold_gib:.4f} GiB"
        )
        # print(f"Estimate GPU memory consumption for cache and buffers: {self.esitimate_gpu_memory:.4f} GB")
        return preallocate


    def block_cache_metadata(self):
        """Return structured capacity metadata for the per-layer GPU block cache."""
        self._sync_async_wave_batch_access("before_block_cache_metadata")
        self._sync_all_async_stream_only_gathers("before_block_cache_metadata")
        self._sync_all_async_cache_admissions("before_block_cache_metadata")
        self._slot_rotation_sync_all_transfers("before_block_cache_metadata")
        dtype_bytes = torch.empty((), dtype=self.dtype).element_size()
        nominal_vectors_per_layer = self.cache_size * self.page_size
        nominal_bytes_per_layer = 2 * self.batch_size * self.kv_head * nominal_vectors_per_layer * self.head_dim * dtype_bytes
        layer_vectors = [cache_pages * self.page_size for cache_pages in self.cache_sizes]
        layer_bytes = [
            2 * self.batch_size * self.kv_head * vectors * self.head_dim * dtype_bytes
            for vectors in layer_vectors
        ]
        logical_total_pages = sum(self.cache_sizes)
        logical_total_vectors = sum(layer_vectors)
        logical_total_bytes = sum(layer_bytes)
        if self.block_cache_slot_rotation_enabled:
            total_pages = self.block_cache_slot_rotation_actual_slots * self.cache_size
            total_vectors = total_pages * self.page_size
            total_bytes = self.block_cache_slot_rotation_gpu_slot_total_bytes
        else:
            total_pages = logical_total_pages
            total_vectors = logical_total_vectors
            total_bytes = logical_total_bytes
        metadata = {
            "block_cache_source": self.block_cache_capacity["source"],
            "block_cache_ratio": self.cache_ratio,
            "block_cache_index_clusters": self.block_cache_capacity["index_cluster_num"],
            "block_cache_retrieval_clusters": self.block_cache_capacity["retrieval_cluster_num"],
            "block_cache_clusters_per_layer": self.cache_cluster_num,
            "block_cache_pages_per_cluster": self.pages_per_cluster,
            "block_cache_pages_per_layer": self.cache_size,
            "block_cache_page_size_vectors": self.page_size,
            "block_cache_vectors_per_layer": nominal_vectors_per_layer,
            "block_cache_layer_count": self.layer_num,
            "block_cache_total_pages": total_pages,
            "block_cache_total_vectors": total_vectors,
            "block_cache_dtype_bytes": dtype_bytes,
            "block_cache_bytes_per_layer": nominal_bytes_per_layer,
            "block_cache_total_bytes": total_bytes,
            "block_cache_nominal_pages_per_layer": self.cache_size,
            "block_cache_nominal_bytes_per_layer": nominal_bytes_per_layer,
            "block_cache_logical_total_pages": logical_total_pages,
            "block_cache_logical_total_vectors": logical_total_vectors,
            "block_cache_logical_total_bytes": logical_total_bytes,
            "block_cache_residency_mode": self.layer_cache_residency_mode,
            "block_cache_capacity_scale_spec": self.layer_cache_capacity_scale_spec or "default",
            "block_cache_active_layer_count": self.block_cache_active_layer_count(),
            "block_cache_layer_pages": self.cache_sizes,
            "block_cache_layer_capacity_scales": self.cache_capacity_scales,
            "block_cache_layer_bytes": layer_bytes,
            "buffer_nprobe_multiplier": self.buffer_nprobe_multiplier,
            "buffer_nprobe_multiplier_env": self.buffer_nprobe_multiplier_env,
            "buffer_pages_per_cluster": self.buffer_pages_per_cluster,
            "buffer_pages_per_cluster_floor": self.buffer_pages_per_cluster_floor,
            "buffer_pages_per_cluster_floor_effective": self.buffer_pages_per_cluster,
            "buffer_pages_per_cluster_floor_env": self.buffer_pages_per_cluster_floor_env,
            "buffer_pages_per_cluster_floor_active": self.buffer_pages_per_cluster_floor_active,
            "buffer_pages_per_cluster_source": self.buffer_pages_per_cluster_source,
            "buffer_pages": self.buffer_size,
            "buffer_min_pages": self.buffer_min_pages,
            "buffer_probe_pages": self.buffer_probe_pages,
            "buffer_retrieval_clusters": self.nprobe + self.nprobe_new,
            "wave_buffer_cpu_core_config_value": self.wave_buffer_cpu_core_config_value,
            "wave_buffer_cpu_core_effective": self.wave_buffer_cpu_core_effective,
            "wave_buffer_cpu_core_override_env": self.wave_buffer_cpu_core_override_env,
            "wave_buffer_cpu_core_override_active": self.wave_buffer_cpu_core_override_active,
            "wave_buffer_cpu_core_source": self.wave_buffer_cpu_core_source,
            "execution_stride": getattr(self, "execution_stride", None),
            "block_cache_allocation_policy": self.block_cache_allocation_policy,
            "block_cache_allocation_policy_env": self.block_cache_allocation_policy_env,
            "block_cache_allocation_decision": self.block_cache_allocation_decision,
            "block_cache_allocation_decision_reason": self.block_cache_allocation_decision_reason,
            "block_cache_free_memory_gib": self.free_memory,
            "block_cache_estimated_gpu_memory_gib": self.estimated_gpu_memory,
            "block_cache_preallocation_threshold_multiplier": self.block_cache_preallocation_threshold_multiplier,
            "block_cache_preallocation_threshold_gib": self.block_cache_preallocation_threshold_gib,
            "block_cache_auto_preallocate_before_prefill": self.block_cache_auto_preallocate_before_prefill,
            "block_cache_preallocated_before_prefill": self.block_cache_preallocated_before_prefill,
            "block_cache_allocated_after_prefill": self.block_cache_allocated_after_prefill,
            "block_cache_allocation_prepare_cache_calls": self.block_cache_allocation_prepare_cache_calls,
            "block_cache_late_init_policy": self.late_block_cache_init_policy,
            "block_cache_late_init_policy_env": self.late_block_cache_init_policy_env,
            "block_cache_late_init_effective": self.block_cache_late_init_effective,
            "block_cache_late_init_mode": self.block_cache_late_init_mode,
            "block_cache_late_init_reason": self.block_cache_late_init_reason,
            "block_cache_late_init_safety": self.block_cache_late_init_safety,
            "block_cache_late_uninitialized_tensor_count": self.block_cache_late_uninitialized_tensor_count,
            "block_cache_late_uninitialized_bytes": self.block_cache_late_uninitialized_bytes,
            "block_cache_late_init_torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "block_cache_late_init_torch_fill_uninitialized_memory": bool(
                getattr(getattr(torch.utils, "deterministic", None), "fill_uninitialized_memory", False)
            ),
            "scratch_buffer_init_policy": self.scratch_buffer_init_policy,
            "scratch_buffer_init_policy_env": self.scratch_buffer_init_policy_env,
            "scratch_buffer_init_effective": self.scratch_buffer_init_effective,
            "scratch_buffer_init_mode": self.scratch_buffer_init_mode,
            "scratch_buffer_init_reason": self.scratch_buffer_init_reason,
            "scratch_buffer_init_safety": self.scratch_buffer_init_safety,
            "scratch_buffer_uninitialized_tensor_count": self.scratch_buffer_uninitialized_tensor_count,
            "scratch_buffer_uninitialized_bytes": self.scratch_buffer_uninitialized_bytes,
            "scratch_buffer_uninitialized_by_name": self.scratch_buffer_uninitialized_by_name,
            "scratch_buffer_init_torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "scratch_buffer_init_torch_fill_uninitialized_memory": bool(
                getattr(getattr(torch.utils, "deterministic", None), "fill_uninitialized_memory", False)
            ),
            "allocation_policy": self.block_cache_allocation_policy,
            "allocation_decision": self.block_cache_allocation_decision,
            "allocation_free_memory_gib": self.free_memory,
            "allocation_estimated_gpu_memory_gib": self.estimated_gpu_memory,
            "allocation_preallocated_before_prefill": self.block_cache_preallocated_before_prefill,
            "stream_only_layers_enabled": bool(self.stream_only_layers),
            "stream_only_layers_env": self.stream_only_layers_spec or "off",
            "stream_only_layer_count": len(self.stream_only_layers),
            "stream_only_layers": sorted(self.stream_only_layers),
            "stream_only_mode": (
                "direct_fused_cluster_gather_from_list_kv"
                if self.stream_only_layers
                else "off"
            ),
            "stream_only_wave_buffer_admission": (
                "bypass_batch_access_lru_update_and_gpu_cache_scatter"
                if self.stream_only_layers
                else "default_block_cache_admission"
            ),
        }
        metadata.update(self._block_cache_telemetry_metadata())
        metadata.update(self._async_cluster_id_copy_metadata())
        metadata.update(self._async_wave_batch_access_metadata())
        metadata.update(self._async_cache_admission_metadata())
        metadata.update(self._async_stream_only_gather_metadata())
        metadata.update(self._index_metadata_staging_metadata())
        metadata.update(self._late_index_metadata_migration_metadata())
        metadata.update(self._slot_rotation_metadata())
        return metadata


    def _stage_buffer_needs_cumsum(self):
        return any(
            layer_idx in self.stage_index_metadata_layers
            and self.cluster_size_cumsum[layer_idx] is not None
            for layer_idx in range(self.layer_num)
        )


    def _allocate_index_metadata_stage_buffers(self):
        self.index_metadata_stage_buffers = {}
        self.index_metadata_stage_streams = {}
        self.index_metadata_stage_events = {}
        self.index_metadata_stage_loaded_layers = {}
        if not self.stage_index_metadata_layers:
            return

        needs_cumsum = self._stage_buffer_needs_cumsum()
        for device_idx in self.device_list:
            if not any(self.layer_mapping[str(layer_idx)] == device_idx for layer_idx in self.stage_index_metadata_layers):
                continue
            buffers = {
                "centroids": _gpu_zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    self.dtype,
                    device_idx,
                ),
                "value_sum": _gpu_zeros(
                    (self.batch_groups, self.n_centroids, self.head_dim),
                    self.dtype,
                    device_idx,
                ),
                "centroids_mask": _gpu_zeros(
                    (self.batch_groups, self.n_centroids),
                    torch.bool,
                    device_idx,
                ),
                "cluster_size": _gpu_zeros(
                    (self.batch_groups, self.n_centroids),
                    self.dtype,
                    device_idx,
                ),
            }
            if needs_cumsum:
                buffers["cluster_size_cumsum"] = _gpu_zeros(
                    (self.batch_groups, self.n_centroids),
                    torch.int32,
                    device_idx,
                )
            self.index_metadata_stage_buffers[device_idx] = buffers
            self.index_metadata_stage_streams[device_idx] = torch.cuda.Stream(device=device_idx)
            self.index_metadata_stage_events[device_idx] = torch.cuda.Event(blocking=False)
            self.index_metadata_stage_loaded_layers[device_idx] = None


    def _invalidate_index_metadata_stage(self, layer_idx=None):
        if not self.stage_index_metadata_layers:
            return
        if layer_idx is None:
            for device_idx in self.index_metadata_stage_loaded_layers:
                self.index_metadata_stage_loaded_layers[device_idx] = None
            return
        device_idx = self.layer_mapping[str(layer_idx)]
        if self.index_metadata_stage_loaded_layers.get(device_idx) == layer_idx:
            self.index_metadata_stage_loaded_layers[device_idx] = None


    def _copy_index_metadata_to_stage(self, layer_idx, wait: bool):
        device_idx = self.layer_mapping[str(layer_idx)]
        if self.index_metadata_stage_loaded_layers.get(device_idx) == layer_idx:
            if wait:
                self.index_metadata_stage_events[device_idx].synchronize()
                self.index_metadata_stage_sync_count += 1
            return

        buffers = self.index_metadata_stage_buffers[device_idx]
        copy_stream = self.index_metadata_stage_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        copy_stream.wait_stream(current_stream)
        copied_bytes = (
            self._metadata_tensor_nbytes(self.centroids[layer_idx])
            + self._metadata_tensor_nbytes(self.value_sum[layer_idx])
            + self._metadata_tensor_nbytes(self.centroids_mask[layer_idx])
            + self._metadata_tensor_nbytes(self.cluster_size[layer_idx])
        )
        with torch.cuda.stream(copy_stream):
            buffers["centroids"].copy_(self.centroids[layer_idx], non_blocking=True)
            buffers["value_sum"].copy_(self.value_sum[layer_idx], non_blocking=True)
            buffers["centroids_mask"].copy_(self.centroids_mask[layer_idx], non_blocking=True)
            buffers["cluster_size"].copy_(self.cluster_size[layer_idx], non_blocking=True)
            source_cumsum = self.cluster_size_cumsum[layer_idx]
            if source_cumsum is not None and "cluster_size_cumsum" in buffers:
                buffers["cluster_size_cumsum"].copy_(source_cumsum, non_blocking=True)
                copied_bytes += self._metadata_tensor_nbytes(source_cumsum)
            self.index_metadata_stage_events[device_idx].record(copy_stream)
        self.index_metadata_stage_loaded_layers[device_idx] = layer_idx
        self.index_metadata_stage_prefetch_count += 1
        self.index_metadata_stage_copy_bytes += copied_bytes
        if wait:
            self.index_metadata_stage_events[device_idx].synchronize()
            self.index_metadata_stage_sync_count += 1


    def _prefetch_next_index_metadata_layer(self, layer_idx):
        if not self.stage_index_metadata_layers:
            return
        for next_layer_idx in sorted(self.stage_index_metadata_layers):
            if next_layer_idx > layer_idx:
                self._copy_index_metadata_to_stage(next_layer_idx, wait=False)
                return


    def _index_metadata_for_attention(self, layer_idx):
        if not self._is_index_metadata_staged_layer(layer_idx):
            self._prefetch_next_index_metadata_layer(layer_idx)
            return (
                self.centroids[layer_idx],
                self.value_sum[layer_idx],
                self.centroids_mask[layer_idx],
                self.cluster_size[layer_idx],
                self.cluster_size_cumsum[layer_idx],
            )

        self._copy_index_metadata_to_stage(layer_idx, wait=True)
        buffers = self.index_metadata_stage_buffers[self.layer_mapping[str(layer_idx)]]
        return (
            buffers["centroids"],
            buffers["value_sum"],
            buffers["centroids_mask"],
            buffers["cluster_size"],
            buffers.get("cluster_size_cumsum"),
        )

    def allocate_computation_buffer(self):
        """Allocate layer-share buffers, dict for different GPUs"""
        self.gemm_o_dict, self.softmax_o_dict, self.norm_dict, self.sum_dict, self.dist_dict = {}, {}, {}, {}, {}
        self.cI_dict, self.cV_dict = {}, {}
        self.es_centroids_dict, self.es_value_sum_dict, self.es_cluster_size_dict= {}, {}, {}
        self.execution_buffer_keys_dict, self.execution_buffer_values_dict, self.valid_lengths_dict = {}, {}, {}
        self.static_len_tensor_dict = {}
        self.nprobe_tensor_dict = {}
        if self.use_cuda_graph:
            self.query_buffer_dict = {}
            self.es_out_dict, self.es_lse_dict = {}, {}
            self.attn_out_dict = {}
        
        for device_idx in self.device_list:
            # for batch_gemm_softmax kernel
            self.gemm_o_dict[device_idx] = self._new_scratch_tensor(
                "gemm_o",
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
                self.dtype,
                device_idx,
            )
            self.softmax_o_dict[device_idx] = self._new_scratch_tensor(
                "softmax_o",
                (self.batch_size*self.kv_head, self.group_size, self.n_centroids),
                self.dtype,
                device_idx,
            )
            self.norm_dict[device_idx] = self._new_scratch_tensor(
                "norm",
                (self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                torch.float32,
                device_idx,
            )
            self.sum_dict[device_idx] = self._new_scratch_tensor(
                "sum",
                (self.batch_size*self.kv_head, self.group_size, (self.n_centroids+256-1)//256),
                torch.float32,
                device_idx,
            )
            self.dist_dict[device_idx] = self._new_scratch_tensor(
                "dist",
                (self.batch_size*self.kv_head, self.n_centroids),
                self.dtype,
                device_idx,
            )
            
            # for topk
            self.cI_dict[device_idx] = self._new_scratch_tensor(
                "cI",
                (self.batch_size*self.kv_head, self.max_compute_cluster_num),
                torch.int64,
                device_idx,
            )
            self.cV_dict[device_idx] = self._new_scratch_tensor(
                "cV",
                (self.batch_size*self.kv_head, self.max_compute_cluster_num),
                self.dtype,
                device_idx,
            )
            
            # estimation zone
            self.es_centroids_dict[device_idx] = self._new_scratch_tensor(
                "es_centroids",
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.es_value_sum_dict[device_idx] = self._new_scratch_tensor(
                "es_value_sum",
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.es_cluster_size_dict[device_idx] = self._new_scratch_tensor(
                "es_cluster_size",
                (self.batch_groups, 1, 1, self.es_cluster_num),
                self.dtype,
                device_idx,
            )
            
            # execution buffer
            self.execution_buffer_keys_dict[device_idx] = self._new_scratch_tensor(
                "execution_buffer_keys",
                (self.batch_groups, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.execution_buffer_values_dict[device_idx] = self._new_scratch_tensor(
                "execution_buffer_values",
                (self.batch_groups, self.buffer_size*self.page_size+self.static_stride, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.valid_lengths_dict[device_idx] = self._new_scratch_tensor(
                "valid_lengths",
                (self.batch_groups),
                torch.int32,
                device_idx,
            )
            self.static_len_tensor_dict[device_idx] = torch.tensor(
                self.static_pattern_total, dtype=torch.int32, device=device_idx)
            self.nprobe_tensor_dict[device_idx] = torch.tensor(
                self.nprobe, dtype=torch.int32, device=device_idx)
            
            # allocate buffers used when enable CUDA graphs
            if self.use_cuda_graph:
                self.query_buffer_dict[device_idx] = self._new_scratch_tensor(
                    "query_buffer",
                    (self.batch_groups, 1, self.group_size, self.head_dim),
                    self.dtype,
                    device_idx,
                )
                if self.es_cluster_num > 0:
                    self.es_out_dict[device_idx] = self._new_scratch_tensor(
                        "es_out",
                        (self.batch_groups, 1, self.group_size, self.head_dim),
                        self.dtype,
                        device_idx,
                    )
                    self.es_lse_dict[device_idx] = self._new_scratch_tensor(
                        "es_lse",
                        (self.batch_groups, self.group_size, 1),
                        torch.float32,
                        device_idx,
                    )
                else:
                    self.es_out_dict[device_idx] = None
                    self.es_lse_dict[device_idx] = None
                self.attn_out_dict[device_idx] = self._new_scratch_tensor(
                    "attn_out",
                    (self.batch_size, 1, self.num_heads, self.head_dim),
                    self.dtype,
                    device_idx,
                )
                
        self.execution_stride = self.buffer_size * self.page_size + self.static_stride

        # point to the buffer of current layer's device
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.static_len_tensor = self.static_len_tensor_dict[self.layer_mapping[str(0)]]
        self.nprobe_tensor = self.nprobe_tensor_dict[self.layer_mapping[str(0)]]
        if self.use_cuda_graph:
            self.query_buffer = self.query_buffer_dict[self.layer_mapping[str(0)]]
            self.attn_out = self.attn_out_dict[self.layer_mapping[str(0)]]
        else:
            self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
            self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
            self.norm = self.norm_dict[self.layer_mapping[str(0)]]
            self.sum = self.sum_dict[self.layer_mapping[str(0)]]
            self.dist = self.dist_dict[self.layer_mapping[str(0)]]
            self.cV = self.cV_dict[self.layer_mapping[str(0)]]
            self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
            self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
            self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]
            self.execution_buffer_keys = self.execution_buffer_keys_dict[self.layer_mapping[str(0)]]
            self.execution_buffer_values = self.execution_buffer_values_dict[self.layer_mapping[str(0)]]
            self.valid_lengths = self.valid_lengths_dict[self.layer_mapping[str(0)]]
        self._allocate_index_metadata_stage_buffers()


    def _new_late_block_cache_tensor(self, layer_idx):
        shape = (
            self.batch_size,
            self.kv_head,
            self.cache_sizes[layer_idx],
            self.page_size,
            self.head_dim,
        )
        if self.late_block_cache_init_policy == "uninitialized":
            tensor = _gpu_empty(shape, self.dtype, self.layer_mapping[str(layer_idx)])
            nbytes = int(tensor.numel() * tensor.element_size())
            if nbytes > 0:
                self.block_cache_late_uninitialized_tensor_count += 1
                self.block_cache_late_uninitialized_bytes += nbytes
            return tensor
        return _gpu_zeros(shape, self.dtype, self.layer_mapping[str(layer_idx)])


    def _block_cache_slot_rotation_shape(self):
        return (
            self.batch_size,
            self.kv_head,
            self.cache_size,
            self.page_size,
            self.head_dim,
        )


    def _tensor_nbytes(self, tensor):
        return int(tensor.numel() * tensor.element_size())


    def _slot_rotation_layer_bytes(self, layer_idx):
        return self._tensor_nbytes(self.block_cache_cpu_keys[layer_idx]) + self._tensor_nbytes(
            self.block_cache_cpu_values[layer_idx]
        )


    def _slot_rotation_slot_for_layer(self, layer_idx):
        return layer_idx % self.block_cache_slot_rotation_actual_slots


    def _slot_rotation_page_bytes(self):
        dtype_bytes = torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.page_size * self.head_dim * dtype_bytes


    def _slot_rotation_collect_page_ids(self, layer_idx, ids_tensor, counts_tensor, source_name):
        pages_by_group = []
        listed_count = 0
        unique_count = 0
        for group_idx in range(self.batch_groups):
            count = int(counts_tensor[group_idx].item())
            if count < 0 or count > self.buffer_size:
                self.block_cache_slot_rotation_page_index_violation_count += 1
                self._slot_rotation_violation(
                    f"{source_name} for layer {layer_idx} group {group_idx} has invalid count {count}"
                )
            page_ids = []
            if count > 0:
                for raw_page_id in ids_tensor[group_idx, :count].tolist():
                    page_id = int(raw_page_id)
                    if page_id < 0 or page_id >= self.cache_size:
                        self.block_cache_slot_rotation_page_index_violation_count += 1
                        self._slot_rotation_violation(
                            f"{source_name} for layer {layer_idx} group {group_idx} "
                            f"contains invalid page id {page_id}"
                        )
                    page_ids.append(page_id)
            listed_count += len(page_ids)
            unique_page_ids = sorted(set(page_ids))
            unique_count += len(unique_page_ids)
            pages_by_group.append(unique_page_ids)
        return pages_by_group, listed_count, unique_count


    def _slot_rotation_copy_page_ranges(self, dst_tensor, src_tensor, pages_by_group):
        dst_view = dst_tensor.view(self.batch_groups, self.cache_size, self.page_size, self.head_dim)
        src_view = src_tensor.view(self.batch_groups, self.cache_size, self.page_size, self.head_dim)
        for group_idx, page_ids in enumerate(pages_by_group):
            if not page_ids:
                continue
            start_page = page_ids[0]
            last_page = start_page
            for page_id in page_ids[1:]:
                if page_id == last_page + 1:
                    last_page = page_id
                    continue
                dst_view[group_idx, start_page:last_page + 1].copy_(
                    src_view[group_idx, start_page:last_page + 1],
                    non_blocking=True,
                )
                start_page = page_id
                last_page = page_id
            dst_view[group_idx, start_page:last_page + 1].copy_(
                src_view[group_idx, start_page:last_page + 1],
                non_blocking=True,
            )


    def _slot_rotation_record_transition(self, slot_id, state, layer_idx, generation, reason):
        self.block_cache_slot_rotation_transition_count += 1
        entry = {
            "transition": self.block_cache_slot_rotation_transition_count,
            "slot_id": slot_id,
            "state": state,
            "layer_idx": layer_idx,
            "generation": generation,
            "reason": reason,
        }
        self.block_cache_slot_rotation_transition_tail.append(entry)
        if len(self.block_cache_slot_rotation_transition_tail) > 128:
            self.block_cache_slot_rotation_transition_tail.pop(0)


    def _slot_rotation_set_state(self, slot_id, state, layer_idx=None, generation=None, reason="state_update"):
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if layer_idx is not None:
            slot["resident_layer"] = layer_idx
        if generation is not None:
            slot["generation"] = generation
        slot["state"] = state
        self._slot_rotation_record_transition(
            slot_id,
            state,
            slot["resident_layer"],
            slot["generation"],
            reason,
        )


    def _slot_rotation_violation(self, message):
        self.block_cache_slot_rotation_violation_count += 1
        raise RuntimeError(f"RETROINFER_BLOCK_CACHE_SLOT_ROTATION ownership violation: {message}")


    def _allocate_slot_rotation_block_cache(self):
        if not self.block_cache_slot_rotation_enabled:
            return False

        self.block_cache_slot_rotation_allocation_status = "allocating"
        shape = self._block_cache_slot_rotation_shape()
        device_idx = self.device_list[0]
        try:
            self.block_cache_cpu_keys = [
                _cpu_zeros(shape, dtype=self.dtype, pin_memory=True) for _ in range(self.layer_num)
            ]
            self.block_cache_cpu_values = [
                _cpu_zeros(shape, dtype=self.dtype, pin_memory=True) for _ in range(self.layer_num)
            ]
        except RuntimeError as exc:
            self.block_cache_slot_rotation_allocation_status = "pinned_host_allocation_failed"
            self.block_cache_slot_rotation_allocation_error = str(exc)
            raise RuntimeError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION failed to allocate the pinned CPU full-K/V "
                "block-cache owner; classify this as an environment/resource blocker, not an idea failure"
            ) from exc

        if not all(tensor.is_pinned() for tensor in self.block_cache_cpu_keys + self.block_cache_cpu_values):
            self.block_cache_slot_rotation_allocation_status = "pinned_host_allocation_not_pinned"
            raise RuntimeError(
                "RETROINFER_BLOCK_CACHE_SLOT_ROTATION CPU owner allocation returned pageable memory"
            )

        self.block_cache_slot_rotation_cpu_owner_pinned = True
        self.block_cache_slot_rotation_cpu_owner_bytes = sum(
            self._tensor_nbytes(tensor) for tensor in self.block_cache_cpu_keys + self.block_cache_cpu_values
        )
        self.block_cache_slot_rotation_cpu_owner_pinned_bytes = self.block_cache_slot_rotation_cpu_owner_bytes
        self.block_cache_slot_rotation_cpu_owner_pageable_bytes = 0

        self.block_cache_gpu_slot_keys = [
            _gpu_zeros(shape, dtype=self.dtype, device=device_idx)
            for _ in range(self.block_cache_slot_rotation_actual_slots)
        ]
        self.block_cache_gpu_slot_values = [
            _gpu_zeros(shape, dtype=self.dtype, device=device_idx)
            for _ in range(self.block_cache_slot_rotation_actual_slots)
        ]
        if self.block_cache_gpu_slot_keys:
            self.block_cache_slot_rotation_gpu_slot_bytes = (
                self._tensor_nbytes(self.block_cache_gpu_slot_keys[0])
                + self._tensor_nbytes(self.block_cache_gpu_slot_values[0])
            )
        self.block_cache_slot_rotation_gpu_slot_total_bytes = (
            self.block_cache_slot_rotation_gpu_slot_bytes
            * self.block_cache_slot_rotation_actual_slots
        )
        self.cache_keys = [
            self.block_cache_gpu_slot_keys[self._slot_rotation_slot_for_layer(layer_idx)]
            for layer_idx in range(self.layer_num)
        ]
        self.cache_values = [
            self.block_cache_gpu_slot_values[self._slot_rotation_slot_for_layer(layer_idx)]
            for layer_idx in range(self.layer_num)
        ]
        self.block_cache_slot_rotation_h2d_streams = {
            device_idx: torch.cuda.Stream(device=device_idx)
        }
        self.block_cache_slot_rotation_d2h_streams = {
            device_idx: torch.cuda.Stream(device=device_idx)
        }
        self.block_cache_slot_rotation_h2d_events = [
            torch.cuda.Event(blocking=False)
            for _ in range(self.block_cache_slot_rotation_actual_slots)
        ]
        self.block_cache_slot_rotation_d2h_events = [
            torch.cuda.Event(blocking=False)
            for _ in range(self.block_cache_slot_rotation_actual_slots)
        ]
        self.block_cache_slot_rotation_slots = [
            {
                "slot_id": slot_id,
                "resident_layer": None,
                "generation": 0,
                "state": "free",
                "h2d_pending": False,
                "d2h_pending_layer": None,
            }
            for slot_id in range(self.block_cache_slot_rotation_actual_slots)
        ]
        self.block_cache_slot_rotation_layer_generation = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_layer_pending_d2h = {}
        self.block_cache_slot_rotation_layer_to_slot = [None for _ in range(self.layer_num)]
        self.cache_stride = self.cache_size
        self.block_cache_slot_rotation_allocation_status = "pinned_cpu_full_kv_and_gpu_slots_allocated"
        return True


    def _slot_rotation_sync_pending_layer_d2h(self, layer_idx, reason):
        pending = self.block_cache_slot_rotation_layer_pending_d2h.get(layer_idx)
        if pending is None:
            return
        sync_start = time.perf_counter()
        pending["event"].synchronize()
        self.block_cache_slot_rotation_explicit_sync_elapsed_ms += (
            time.perf_counter() - sync_start
        ) * 1000.0
        self.block_cache_slot_rotation_explicit_sync_count += 1
        self.block_cache_slot_rotation_layer_generation[layer_idx] = pending["generation"]
        slot_id = pending["slot_id"]
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["d2h_pending_layer"] == layer_idx:
            slot["d2h_pending_layer"] = None
            if slot["resident_layer"] == layer_idx and not slot["h2d_pending"]:
                self._slot_rotation_set_state(
                    slot_id,
                    "resident",
                    layer_idx=layer_idx,
                    generation=pending["generation"],
                    reason=reason,
                )
        self.block_cache_slot_rotation_layer_pending_d2h.pop(layer_idx, None)


    def _slot_rotation_complete_slot_d2h_if_done(self, slot_id, reason):
        slot = self.block_cache_slot_rotation_slots[slot_id]
        layer_idx = slot["d2h_pending_layer"]
        if layer_idx is None:
            return
        if not self.block_cache_slot_rotation_d2h_events[slot_id].query():
            return
        pending = self.block_cache_slot_rotation_layer_pending_d2h.pop(layer_idx, None)
        if pending is not None:
            self.block_cache_slot_rotation_layer_generation[layer_idx] = pending["generation"]
        slot["d2h_pending_layer"] = None
        if slot["resident_layer"] == layer_idx and not slot["h2d_pending"]:
            self._slot_rotation_set_state(
                slot_id,
                "resident",
                layer_idx=layer_idx,
                generation=self.block_cache_slot_rotation_layer_generation[layer_idx],
                reason=reason,
            )


    def _slot_rotation_sync_slot_h2d(self, slot_id, reason):
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if not slot["h2d_pending"]:
            return
        sync_start = time.perf_counter()
        self.block_cache_slot_rotation_h2d_events[slot_id].synchronize()
        self.block_cache_slot_rotation_explicit_sync_elapsed_ms += (
            time.perf_counter() - sync_start
        ) * 1000.0
        self.block_cache_slot_rotation_explicit_sync_count += 1
        slot["h2d_pending"] = False
        self._slot_rotation_complete_slot_d2h_if_done(slot_id, reason)
        self._slot_rotation_set_state(
            slot_id,
            "resident",
            layer_idx=slot["resident_layer"],
            generation=slot["generation"],
            reason=reason,
        )


    def _slot_rotation_prepare_delta_layer(self, layer_idx):
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["h2d_pending"]:
            self._slot_rotation_sync_slot_h2d(slot_id, f"delta_layer_{layer_idx}:pending_h2d")
        pending_layer = slot["d2h_pending_layer"]
        if pending_layer is not None:
            self.block_cache_slot_rotation_overwrite_prevention_count += 1
            if pending_layer == layer_idx:
                self.block_cache_slot_rotation_dirty_read_prevention_count += 1
            self._slot_rotation_sync_pending_layer_d2h(
                pending_layer,
                f"delta_layer_{layer_idx}:before_slot_reuse",
            )
        self._slot_rotation_sync_pending_layer_d2h(
            layer_idx,
            f"delta_layer_{layer_idx}:before_cpu_owner_read",
        )
        self.block_cache_slot_rotation_layer_to_slot[layer_idx] = slot_id
        self._slot_rotation_set_state(
            slot_id,
            "compute_in_use",
            layer_idx=layer_idx,
            generation=self.block_cache_slot_rotation_layer_generation[layer_idx],
            reason="delta_consume_layer",
        )


    def _slot_rotation_materialize_hit_pages(self, layer_idx):
        if not self.block_cache_slot_rotation_delta_enabled:
            return
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["resident_layer"] != layer_idx:
            self._slot_rotation_violation(
                f"delta materialization for layer {layer_idx} found slot {slot_id} "
                f"resident_layer={slot['resident_layer']}"
            )
        if slot["d2h_pending_layer"] is not None:
            self._slot_rotation_violation(
                f"delta materialization for layer {layer_idx} found pending D2H for "
                f"layer {slot['d2h_pending_layer']} in slot {slot_id}"
            )
        pages_by_group, listed_pages, unique_pages = self._slot_rotation_collect_page_ids(
            layer_idx,
            self.hit_unit_idices[layer_idx],
            self.hit_num_units[layer_idx],
            "hit pages",
        )
        self.block_cache_slot_rotation_page_h2d_listed_pages[layer_idx] += listed_pages
        self.block_cache_slot_rotation_page_h2d_unique_pages[layer_idx] += unique_pages
        self.block_cache_slot_rotation_page_h2d_total_listed_pages += listed_pages
        self.block_cache_slot_rotation_page_h2d_total_unique_pages += unique_pages
        if unique_pages == 0:
            return

        device_idx = self.layer_mapping[str(layer_idx)]
        h2d_stream = self.block_cache_slot_rotation_h2d_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        h2d_stream.wait_stream(current_stream)
        page_bytes = self._slot_rotation_page_bytes()
        with torch.cuda.stream(h2d_stream):
            self._slot_rotation_copy_page_ranges(
                self.block_cache_gpu_slot_keys[slot_id],
                self.block_cache_cpu_keys[layer_idx],
                pages_by_group,
            )
            self._slot_rotation_copy_page_ranges(
                self.block_cache_gpu_slot_values[slot_id],
                self.block_cache_cpu_values[layer_idx],
                pages_by_group,
            )
            self.block_cache_slot_rotation_h2d_events[slot_id].record(h2d_stream)
        slot["h2d_pending"] = True
        transfer_bytes = unique_pages * page_bytes
        self.block_cache_slot_rotation_page_h2d_counts[layer_idx] += 1
        self.block_cache_slot_rotation_page_h2d_bytes[layer_idx] += transfer_bytes
        self.block_cache_slot_rotation_page_h2d_total_count += 1
        self.block_cache_slot_rotation_page_h2d_total_bytes += transfer_bytes
        self.block_cache_slot_rotation_hit_page_materialization_count += 1
        self._slot_rotation_set_state(
            slot_id,
            "h2d_pending",
            layer_idx=layer_idx,
            generation=self.block_cache_slot_rotation_layer_generation[layer_idx],
            reason="delta_hit_page_materialization",
        )

        wait_start = time.perf_counter()
        current_stream.wait_event(self.block_cache_slot_rotation_h2d_events[slot_id])
        self.block_cache_slot_rotation_wait_enqueue_elapsed_ms += (
            time.perf_counter() - wait_start
        ) * 1000.0
        self.block_cache_slot_rotation_event_wait_count += 1
        slot["h2d_pending"] = False
        self._slot_rotation_set_state(
            slot_id,
            "compute_in_use",
            layer_idx=layer_idx,
            generation=self.block_cache_slot_rotation_layer_generation[layer_idx],
            reason="delta_hit_pages_ready",
        )


    def _slot_rotation_after_layer_dirty_pages(self, layer_idx):
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["resident_layer"] != layer_idx:
            self._slot_rotation_violation(
                f"{self.block_cache_slot_rotation_copy_mode} layer {layer_idx} finished with slot {slot_id} "
                f"resident_layer={slot['resident_layer']}"
            )
        if slot["h2d_pending"]:
            self._slot_rotation_violation(
                f"{self.block_cache_slot_rotation_copy_mode} layer {layer_idx} finished while H2D is pending"
            )

        device_idx = self.layer_mapping[str(layer_idx)]
        self._sync_async_cache_admission_for_device(device_idx, "before_slot_rotation_delta_d2h")
        pages_by_group, listed_pages, unique_pages = self._slot_rotation_collect_page_ids(
            layer_idx,
            self.update_cache_indices[layer_idx],
            self.update_num_units[layer_idx],
            "dirty update pages",
        )
        self.block_cache_slot_rotation_page_d2h_listed_pages[layer_idx] += listed_pages
        self.block_cache_slot_rotation_page_d2h_unique_pages[layer_idx] += unique_pages
        self.block_cache_slot_rotation_page_d2h_total_listed_pages += listed_pages
        self.block_cache_slot_rotation_page_d2h_total_unique_pages += unique_pages
        if unique_pages == 0:
            self._slot_rotation_set_state(
                slot_id,
                "resident",
                layer_idx=layer_idx,
                generation=slot["generation"],
                reason=f"{self.block_cache_slot_rotation_copy_mode}_after_layer_no_dirty_pages",
            )
            return

        d2h_stream = self.block_cache_slot_rotation_d2h_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        d2h_stream.wait_stream(current_stream)
        next_generation = slot["generation"] + 1
        page_bytes = self._slot_rotation_page_bytes()
        with torch.cuda.stream(d2h_stream):
            self._slot_rotation_copy_page_ranges(
                self.block_cache_cpu_keys[layer_idx],
                self.block_cache_gpu_slot_keys[slot_id],
                pages_by_group,
            )
            self._slot_rotation_copy_page_ranges(
                self.block_cache_cpu_values[layer_idx],
                self.block_cache_gpu_slot_values[slot_id],
                pages_by_group,
            )
            self.block_cache_slot_rotation_d2h_events[slot_id].record(d2h_stream)
        transfer_bytes = unique_pages * page_bytes
        self.block_cache_slot_rotation_layer_pending_d2h[layer_idx] = {
            "slot_id": slot_id,
            "generation": next_generation,
            "event": self.block_cache_slot_rotation_d2h_events[slot_id],
        }
        slot["d2h_pending_layer"] = layer_idx
        self.block_cache_slot_rotation_page_d2h_counts[layer_idx] += 1
        self.block_cache_slot_rotation_page_d2h_bytes[layer_idx] += transfer_bytes
        self.block_cache_slot_rotation_page_d2h_total_count += 1
        self.block_cache_slot_rotation_page_d2h_total_bytes += transfer_bytes
        self.block_cache_slot_rotation_dirty_page_flush_count += 1
        self._slot_rotation_set_state(
            slot_id,
            "d2h_pending",
            layer_idx=layer_idx,
            generation=next_generation,
            reason=f"{self.block_cache_slot_rotation_copy_mode}_dirty_page_flush",
        )


    def _slot_rotation_after_layer_delta(self, layer_idx):
        self._slot_rotation_after_layer_dirty_pages(layer_idx)


    def _slot_rotation_launch_h2d(self, layer_idx, reason):
        if not self.block_cache_slot_rotation_enabled:
            return False
        self._slot_rotation_sync_pending_layer_d2h(layer_idx, f"{reason}:before_h2d_cpu_owner")
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        slot = self.block_cache_slot_rotation_slots[slot_id]
        expected_generation = self.block_cache_slot_rotation_layer_generation[layer_idx]
        if (
            slot["resident_layer"] == layer_idx
            and slot["generation"] == expected_generation
            and not slot["h2d_pending"]
        ):
            return False
        if slot["state"] == "compute_in_use":
            self._slot_rotation_violation(
                f"attempted to overwrite slot {slot_id} while layer {slot['resident_layer']} is compute_in_use"
            )
        if slot["h2d_pending"]:
            self._slot_rotation_sync_slot_h2d(slot_id, f"{reason}:before_overwriting_pending_h2d")
        device_idx = self.layer_mapping[str(layer_idx)]
        h2d_stream = self.block_cache_slot_rotation_h2d_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        h2d_stream.wait_stream(current_stream)
        if slot["d2h_pending_layer"] is not None:
            h2d_stream.wait_event(self.block_cache_slot_rotation_d2h_events[slot_id])
            self.block_cache_slot_rotation_overwrite_prevention_count += 1
        key_bytes = self._tensor_nbytes(self.block_cache_cpu_keys[layer_idx])
        value_bytes = self._tensor_nbytes(self.block_cache_cpu_values[layer_idx])
        with torch.cuda.stream(h2d_stream):
            self.block_cache_gpu_slot_keys[slot_id].copy_(
                self.block_cache_cpu_keys[layer_idx],
                non_blocking=True,
            )
            self.block_cache_gpu_slot_values[slot_id].copy_(
                self.block_cache_cpu_values[layer_idx],
                non_blocking=True,
            )
            self.block_cache_slot_rotation_h2d_events[slot_id].record(h2d_stream)
        slot["h2d_pending"] = True
        self.block_cache_slot_rotation_layer_to_slot[layer_idx] = slot_id
        self.block_cache_slot_rotation_h2d_counts[layer_idx] += 1
        self.block_cache_slot_rotation_h2d_bytes[layer_idx] += key_bytes + value_bytes
        self.block_cache_slot_rotation_h2d_total_count += 1
        self.block_cache_slot_rotation_h2d_total_bytes += key_bytes + value_bytes
        self._slot_rotation_set_state(
            slot_id,
            "h2d_pending",
            layer_idx=layer_idx,
            generation=expected_generation,
            reason=reason,
        )
        return True


    def _slot_rotation_wait_for_layer(self, layer_idx):
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        expected_generation = self.block_cache_slot_rotation_layer_generation[layer_idx]
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["resident_layer"] != layer_idx or slot["generation"] != expected_generation:
            if slot["resident_layer"] == layer_idx:
                self.block_cache_slot_rotation_generation_mismatch_count += 1
            self.block_cache_slot_rotation_dirty_read_prevention_count += 1
            self._slot_rotation_launch_h2d(layer_idx, "consume_layer_missing_or_stale")
            slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["resident_layer"] != layer_idx or slot["generation"] != expected_generation:
            self._slot_rotation_violation(
                f"slot {slot_id} has layer={slot['resident_layer']} generation={slot['generation']} "
                f"but layer {layer_idx} generation {expected_generation} is required"
            )
        if slot["h2d_pending"]:
            wait_start = time.perf_counter()
            current_stream = torch.cuda.current_stream(torch.device(self.layer_mapping[str(layer_idx)]))
            current_stream.wait_event(self.block_cache_slot_rotation_h2d_events[slot_id])
            self.block_cache_slot_rotation_wait_enqueue_elapsed_ms += (
                time.perf_counter() - wait_start
            ) * 1000.0
            self.block_cache_slot_rotation_event_wait_count += 1
            slot["h2d_pending"] = False
            self._slot_rotation_complete_slot_d2h_if_done(slot_id, "consume_layer_h2d_wait")
        if slot["d2h_pending_layer"] == layer_idx:
            self._slot_rotation_violation(
                f"layer {layer_idx} was about to read slot {slot_id} while its D2H update is pending"
            )
        self._slot_rotation_set_state(
            slot_id,
            "compute_in_use",
            layer_idx=layer_idx,
            generation=expected_generation,
            reason="consume_layer",
        )


    def _slot_rotation_prefetch_future(self, layer_idx):
        if self.block_cache_slot_rotation_actual_slots <= 1:
            return
        future_layer = (layer_idx + self.block_cache_slot_rotation_actual_slots - 1) % self.layer_num
        launched = self._slot_rotation_launch_h2d(future_layer, f"prefetch_from_layer_{layer_idx}")
        if launched:
            self.block_cache_slot_rotation_prefetch_count += 1
            if future_layer <= layer_idx:
                self.block_cache_slot_rotation_wraparound_prefetch_count += 1


    def _slot_rotation_before_layer(self, layer_idx):
        if not self.block_cache_slot_rotation_enabled:
            return
        self.block_cache_slot_rotation_path_executed = True
        if self.block_cache_slot_rotation_delta_enabled:
            self._slot_rotation_prepare_delta_layer(layer_idx)
            return
        self._slot_rotation_wait_for_layer(layer_idx)
        self._slot_rotation_prefetch_future(layer_idx)


    def _slot_rotation_after_layer(self, layer_idx):
        if not self.block_cache_slot_rotation_enabled:
            return
        if (
            self.block_cache_slot_rotation_delta_enabled
            or self.block_cache_slot_rotation_dirty_page_d2h_enabled
        ):
            self._slot_rotation_after_layer_dirty_pages(layer_idx)
            return
        slot_id = self._slot_rotation_slot_for_layer(layer_idx)
        slot = self.block_cache_slot_rotation_slots[slot_id]
        if slot["resident_layer"] != layer_idx:
            self._slot_rotation_violation(
                f"layer {layer_idx} finished with slot {slot_id} resident_layer={slot['resident_layer']}"
            )
        if slot["h2d_pending"]:
            self._slot_rotation_violation(f"layer {layer_idx} finished while H2D is still pending")

        device_idx = self.layer_mapping[str(layer_idx)]
        self._sync_async_cache_admission_for_device(device_idx, "before_slot_rotation_d2h")
        d2h_stream = self.block_cache_slot_rotation_d2h_streams[device_idx]
        current_stream = torch.cuda.current_stream(torch.device(device_idx))
        d2h_stream.wait_stream(current_stream)
        next_generation = slot["generation"] + 1
        key_bytes = self._tensor_nbytes(self.block_cache_cpu_keys[layer_idx])
        value_bytes = self._tensor_nbytes(self.block_cache_cpu_values[layer_idx])
        with torch.cuda.stream(d2h_stream):
            self.block_cache_cpu_keys[layer_idx].copy_(
                self.block_cache_gpu_slot_keys[slot_id],
                non_blocking=True,
            )
            self.block_cache_cpu_values[layer_idx].copy_(
                self.block_cache_gpu_slot_values[slot_id],
                non_blocking=True,
            )
            self.block_cache_slot_rotation_d2h_events[slot_id].record(d2h_stream)
        self.block_cache_slot_rotation_layer_pending_d2h[layer_idx] = {
            "slot_id": slot_id,
            "generation": next_generation,
            "event": self.block_cache_slot_rotation_d2h_events[slot_id],
        }
        slot["d2h_pending_layer"] = layer_idx
        self.block_cache_slot_rotation_d2h_counts[layer_idx] += 1
        self.block_cache_slot_rotation_d2h_bytes[layer_idx] += key_bytes + value_bytes
        self.block_cache_slot_rotation_d2h_total_count += 1
        self.block_cache_slot_rotation_d2h_total_bytes += key_bytes + value_bytes
        self._slot_rotation_set_state(
            slot_id,
            "d2h_pending",
            layer_idx=layer_idx,
            generation=next_generation,
            reason="after_layer_update",
        )


    def _slot_rotation_sync_all_transfers(self, reason):
        if not self.block_cache_slot_rotation_enabled:
            return
        for slot_id, slot in enumerate(self.block_cache_slot_rotation_slots):
            if slot["h2d_pending"]:
                self._slot_rotation_sync_slot_h2d(slot_id, f"{reason}:pending_h2d")
        for layer_idx in sorted(list(self.block_cache_slot_rotation_layer_pending_d2h)):
            self._slot_rotation_sync_pending_layer_d2h(layer_idx, f"{reason}:pending_d2h")


    def finish_decode(self):
        self._slot_rotation_sync_all_transfers("decode_end")


    def _slot_rotation_reset_after_cuda_graph_capture(self):
        if not self.block_cache_slot_rotation_enabled:
            return
        self._slot_rotation_sync_all_transfers("after_cuda_graph_capture_before_reset")
        for slot in self.block_cache_slot_rotation_slots:
            slot["resident_layer"] = None
            slot["generation"] = 0
            slot["state"] = "free"
            slot["h2d_pending"] = False
            slot["d2h_pending_layer"] = None
        self.block_cache_slot_rotation_layer_pending_d2h.clear()
        self.block_cache_slot_rotation_layer_generation = [0 for _ in range(self.layer_num)]
        self.block_cache_slot_rotation_layer_to_slot = [None for _ in range(self.layer_num)]
        if self.block_cache_slot_rotation_delta_enabled:
            return
        for layer_idx in range(min(self.block_cache_slot_rotation_actual_slots, self.layer_num)):
            if self._slot_rotation_launch_h2d(layer_idx, "after_cuda_graph_capture_initial_prefetch"):
                self.block_cache_slot_rotation_initial_prefetch_count += 1
        self._slot_rotation_sync_all_transfers("after_cuda_graph_capture_initial_prefetch")


    def _slot_rotation_metadata(self):
        if not self.block_cache_slot_rotation_enabled:
            return {
                "block_cache_slot_rotation_enabled": False,
                "block_cache_slot_rotation_env": self.block_cache_slot_rotation_env or "unset",
                "block_cache_slot_rotation_delta_enabled": False,
                "block_cache_slot_rotation_delta_env": self.block_cache_slot_rotation_delta_env or "unset",
                "block_cache_slot_rotation_dirty_page_d2h_enabled": False,
                "block_cache_slot_rotation_dirty_page_d2h_env": (
                    self.block_cache_slot_rotation_dirty_page_d2h_env or "unset"
                ),
                "block_cache_slot_rotation_copy_mode": "disabled",
                "block_cache_slot_rotation_path_executed": False,
                "block_cache_gpu_slots_requested": self.block_cache_slot_rotation_requested_slots,
                "block_cache_gpu_slots_env": self.block_cache_gpu_slots_env,
                "block_cache_actual_gpu_slot_count": 0,
                "block_cache_actual_gpu_slot_count_le_m": True,
                "block_cache_slot_rotation_cuda_graph_mode": "disabled",
            }
        pending_h2d = sum(1 for slot in self.block_cache_slot_rotation_slots if slot["h2d_pending"])
        pending_d2h = len(self.block_cache_slot_rotation_layer_pending_d2h)
        slot_states = [
            {
                "slot_id": slot["slot_id"],
                "resident_layer": slot["resident_layer"],
                "generation": slot["generation"],
                "state": slot["state"],
                "h2d_pending": slot["h2d_pending"],
                "d2h_pending_layer": slot["d2h_pending_layer"],
            }
            for slot in self.block_cache_slot_rotation_slots
        ]
        layer_mapping = [
            {
                "layer_idx": layer_idx,
                "slot_id": self._slot_rotation_slot_for_layer(layer_idx),
                "last_loaded_slot_id": self.block_cache_slot_rotation_layer_to_slot[layer_idx],
                "cpu_generation": self.block_cache_slot_rotation_layer_generation[layer_idx],
                "pending_d2h": layer_idx in self.block_cache_slot_rotation_layer_pending_d2h,
                "h2d_count": self.block_cache_slot_rotation_h2d_counts[layer_idx],
                "d2h_count": self.block_cache_slot_rotation_d2h_counts[layer_idx],
                "page_h2d_count": self.block_cache_slot_rotation_page_h2d_counts[layer_idx],
                "page_h2d_listed_pages": self.block_cache_slot_rotation_page_h2d_listed_pages[layer_idx],
                "page_h2d_unique_pages": self.block_cache_slot_rotation_page_h2d_unique_pages[layer_idx],
                "page_d2h_count": self.block_cache_slot_rotation_page_d2h_counts[layer_idx],
                "page_d2h_listed_pages": self.block_cache_slot_rotation_page_d2h_listed_pages[layer_idx],
                "page_d2h_unique_pages": self.block_cache_slot_rotation_page_d2h_unique_pages[layer_idx],
            }
            for layer_idx in range(self.layer_num)
        ]
        return {
            "block_cache_slot_rotation_enabled": True,
            "block_cache_slot_rotation_env": self.block_cache_slot_rotation_env or "unset",
            "block_cache_slot_rotation_delta_enabled": self.block_cache_slot_rotation_delta_enabled,
            "block_cache_slot_rotation_delta_env": self.block_cache_slot_rotation_delta_env or "unset",
            "block_cache_slot_rotation_dirty_page_d2h_enabled": (
                self.block_cache_slot_rotation_dirty_page_d2h_enabled
            ),
            "block_cache_slot_rotation_dirty_page_d2h_env": (
                self.block_cache_slot_rotation_dirty_page_d2h_env or "unset"
            ),
            "block_cache_slot_rotation_copy_mode": self.block_cache_slot_rotation_copy_mode,
            "block_cache_slot_rotation_path_executed": self.block_cache_slot_rotation_path_executed,
            "block_cache_slot_rotation_allocation_status": self.block_cache_slot_rotation_allocation_status,
            "block_cache_slot_rotation_allocation_error": self.block_cache_slot_rotation_allocation_error,
            "block_cache_slot_rotation_initial_m_rationale": self.block_cache_slot_rotation_initial_m_rationale,
            "block_cache_gpu_slots_requested": self.block_cache_slot_rotation_requested_slots,
            "block_cache_gpu_slots_env": self.block_cache_gpu_slots_env,
            "block_cache_actual_gpu_slot_count": self.block_cache_slot_rotation_actual_slots,
            "block_cache_actual_gpu_slot_count_le_m": (
                self.block_cache_slot_rotation_actual_slots <= self.block_cache_slot_rotation_requested_slots
            ),
            "block_cache_slot_rotation_cpu_owner_bytes": self.block_cache_slot_rotation_cpu_owner_bytes,
            "block_cache_slot_rotation_cpu_owner_pinned": self.block_cache_slot_rotation_cpu_owner_pinned,
            "block_cache_slot_rotation_cpu_owner_pinned_bytes": (
                self.block_cache_slot_rotation_cpu_owner_pinned_bytes
            ),
            "block_cache_slot_rotation_cpu_owner_pageable_bytes": (
                self.block_cache_slot_rotation_cpu_owner_pageable_bytes
            ),
            "block_cache_slot_rotation_host_meminfo": self.block_cache_slot_rotation_host_meminfo,
            "block_cache_slot_rotation_gpu_slot_bytes": self.block_cache_slot_rotation_gpu_slot_bytes,
            "block_cache_slot_rotation_gpu_slot_total_bytes": (
                self.block_cache_slot_rotation_gpu_slot_total_bytes
            ),
            "block_cache_slot_rotation_h2d_stream_count": len(self.block_cache_slot_rotation_h2d_streams),
            "block_cache_slot_rotation_d2h_stream_count": len(self.block_cache_slot_rotation_d2h_streams),
            "block_cache_slot_rotation_h2d_total_count": self.block_cache_slot_rotation_h2d_total_count,
            "block_cache_slot_rotation_h2d_total_bytes": self.block_cache_slot_rotation_h2d_total_bytes,
            "block_cache_slot_rotation_d2h_total_count": self.block_cache_slot_rotation_d2h_total_count,
            "block_cache_slot_rotation_d2h_total_bytes": self.block_cache_slot_rotation_d2h_total_bytes,
            "block_cache_slot_rotation_full_layer_h2d_total_count": (
                self.block_cache_slot_rotation_h2d_total_count
            ),
            "block_cache_slot_rotation_full_layer_h2d_total_bytes": (
                self.block_cache_slot_rotation_h2d_total_bytes
            ),
            "block_cache_slot_rotation_full_layer_d2h_total_count": (
                self.block_cache_slot_rotation_d2h_total_count
            ),
            "block_cache_slot_rotation_full_layer_d2h_total_bytes": (
                self.block_cache_slot_rotation_d2h_total_bytes
            ),
            "block_cache_slot_rotation_h2d_counts_by_layer": self.block_cache_slot_rotation_h2d_counts,
            "block_cache_slot_rotation_h2d_bytes_by_layer": self.block_cache_slot_rotation_h2d_bytes,
            "block_cache_slot_rotation_d2h_counts_by_layer": self.block_cache_slot_rotation_d2h_counts,
            "block_cache_slot_rotation_d2h_bytes_by_layer": self.block_cache_slot_rotation_d2h_bytes,
            "block_cache_slot_rotation_page_h2d_total_count": (
                self.block_cache_slot_rotation_page_h2d_total_count
            ),
            "block_cache_slot_rotation_page_h2d_total_bytes": (
                self.block_cache_slot_rotation_page_h2d_total_bytes
            ),
            "block_cache_slot_rotation_page_h2d_total_listed_pages": (
                self.block_cache_slot_rotation_page_h2d_total_listed_pages
            ),
            "block_cache_slot_rotation_page_h2d_total_unique_pages": (
                self.block_cache_slot_rotation_page_h2d_total_unique_pages
            ),
            "block_cache_slot_rotation_page_d2h_total_count": (
                self.block_cache_slot_rotation_page_d2h_total_count
            ),
            "block_cache_slot_rotation_page_d2h_total_bytes": (
                self.block_cache_slot_rotation_page_d2h_total_bytes
            ),
            "block_cache_slot_rotation_page_d2h_total_listed_pages": (
                self.block_cache_slot_rotation_page_d2h_total_listed_pages
            ),
            "block_cache_slot_rotation_page_d2h_total_unique_pages": (
                self.block_cache_slot_rotation_page_d2h_total_unique_pages
            ),
            "block_cache_slot_rotation_page_h2d_counts_by_layer": (
                self.block_cache_slot_rotation_page_h2d_counts
            ),
            "block_cache_slot_rotation_page_h2d_bytes_by_layer": (
                self.block_cache_slot_rotation_page_h2d_bytes
            ),
            "block_cache_slot_rotation_page_h2d_listed_pages_by_layer": (
                self.block_cache_slot_rotation_page_h2d_listed_pages
            ),
            "block_cache_slot_rotation_page_h2d_unique_pages_by_layer": (
                self.block_cache_slot_rotation_page_h2d_unique_pages
            ),
            "block_cache_slot_rotation_page_d2h_counts_by_layer": (
                self.block_cache_slot_rotation_page_d2h_counts
            ),
            "block_cache_slot_rotation_page_d2h_bytes_by_layer": (
                self.block_cache_slot_rotation_page_d2h_bytes
            ),
            "block_cache_slot_rotation_page_d2h_listed_pages_by_layer": (
                self.block_cache_slot_rotation_page_d2h_listed_pages
            ),
            "block_cache_slot_rotation_page_d2h_unique_pages_by_layer": (
                self.block_cache_slot_rotation_page_d2h_unique_pages
            ),
            "block_cache_slot_rotation_hit_page_materialization_count": (
                self.block_cache_slot_rotation_hit_page_materialization_count
            ),
            "block_cache_slot_rotation_dirty_page_flush_count": (
                self.block_cache_slot_rotation_dirty_page_flush_count
            ),
            "block_cache_slot_rotation_page_index_violation_count": (
                self.block_cache_slot_rotation_page_index_violation_count
            ),
            "block_cache_slot_rotation_prefetch_count": self.block_cache_slot_rotation_prefetch_count,
            "block_cache_slot_rotation_wraparound_prefetch_count": (
                self.block_cache_slot_rotation_wraparound_prefetch_count
            ),
            "block_cache_slot_rotation_initial_prefetch_count": (
                self.block_cache_slot_rotation_initial_prefetch_count
            ),
            "block_cache_slot_rotation_event_wait_count": self.block_cache_slot_rotation_event_wait_count,
            "block_cache_slot_rotation_wait_enqueue_elapsed_ms": (
                self.block_cache_slot_rotation_wait_enqueue_elapsed_ms
            ),
            "block_cache_slot_rotation_explicit_sync_count": self.block_cache_slot_rotation_explicit_sync_count,
            "block_cache_slot_rotation_explicit_sync_elapsed_ms": (
                self.block_cache_slot_rotation_explicit_sync_elapsed_ms
            ),
            "block_cache_slot_rotation_final_pending_h2d_count": pending_h2d,
            "block_cache_slot_rotation_final_pending_d2h_count": pending_d2h,
            "block_cache_slot_rotation_final_pending_transfer_count": pending_h2d + pending_d2h,
            "block_cache_slot_rotation_overwrite_prevention_count": (
                self.block_cache_slot_rotation_overwrite_prevention_count
            ),
            "block_cache_slot_rotation_dirty_read_prevention_count": (
                self.block_cache_slot_rotation_dirty_read_prevention_count
            ),
            "block_cache_slot_rotation_generation_mismatch_count": (
                self.block_cache_slot_rotation_generation_mismatch_count
            ),
            "block_cache_slot_rotation_violation_count": self.block_cache_slot_rotation_violation_count,
            "block_cache_slot_rotation_slot_states": slot_states,
            "block_cache_slot_rotation_layer_slot_generation": layer_mapping,
            "block_cache_slot_rotation_transition_count": self.block_cache_slot_rotation_transition_count,
            "block_cache_slot_rotation_transition_tail": self.block_cache_slot_rotation_transition_tail,
            "block_cache_slot_rotation_cuda_graph_mode": self.block_cache_slot_rotation_cuda_graph_mode,
            "block_cache_slot_rotation_cuda_graph_blocker_reason": (
                self.block_cache_slot_rotation_cuda_graph_blocker_reason
            ),
        }


    def _new_scratch_tensor(self, name: str, shape, dtype: torch.dtype, device):
        if self.scratch_buffer_init_effective:
            tensor = _gpu_empty(shape, dtype, device)
            nbytes = int(tensor.numel() * tensor.element_size())
            if nbytes > 0:
                self.scratch_buffer_uninitialized_tensor_count += 1
                self.scratch_buffer_uninitialized_bytes += nbytes
                stats = self.scratch_buffer_uninitialized_by_name.setdefault(
                    name,
                    {"count": 0, "bytes": 0},
                )
                stats["count"] += 1
                stats["bytes"] += nbytes
            return tensor
        return _gpu_zeros(shape, dtype, device)


    def prepare_cache(self):
        """Ensure GPU cache and buffers are allocated before decoding"""
        self.block_cache_allocation_prepare_cache_calls += 1
        if self.build_index_when_prefilling:
            # sync the last batch of the last layer
            torch.cuda.synchronize()
            self.wave_buffer[self.layer_num-1].construction_sync()
            # clear temp memory
            self.clusters_cpu, self.cluster_size_cpu = None, None
            self.temp_keys, self.temp_values = None, None
            torch.cuda.empty_cache()

        if not self.allocated:  # allocate GPU cache and buffers after prefilling
            prepare_window_start = time.perf_counter()
            self.cache_keys, self.cache_values = [], []
            self.block_cache_late_uninitialized_tensor_count = 0
            self.block_cache_late_uninitialized_bytes = 0
            self.block_cache_late_init_effective = (
                self.late_block_cache_init_policy == "uninitialized"
                and not self.block_cache_slot_rotation_enabled
            )
            if self.block_cache_slot_rotation_enabled:
                self.block_cache_late_init_mode = "not_applicable_slot_rotation"
                self.block_cache_late_init_reason = "slot_rotation_uses_cpu_owner_and_bounded_gpu_slots"
                self.block_cache_late_init_safety = (
                    "CPU full logical block-cache owner is zero-filled; GPU slots copy from CPU owner "
                    "before any slot-backed cache-hit read"
                )
            else:
                self.block_cache_late_init_mode = (
                    "uninitialized_empty" if self.block_cache_late_init_effective else "zero_fill"
                )
            if self.block_cache_late_init_effective:
                self.block_cache_late_init_reason = "late_allocation_wave_buffer_miss_before_admission_write"
                self.block_cache_late_init_safety = (
                    "WaveBufferCPU starts all clusters as cache misses and marks cache hits only after "
                    "gather_copy_and_scatter writes admitted K/V blocks"
                )
            elif not self.block_cache_slot_rotation_enabled:
                self.block_cache_late_init_reason = (
                    "policy_unset_zero_fill"
                    if self.late_block_cache_init_policy_env == "unset"
                    else "explicit_zero_fill_policy"
                )
                self.block_cache_late_init_safety = "not_applicable_zero_fill"
            if self.block_cache_slot_rotation_enabled:
                self._allocate_slot_rotation_block_cache()
            else:
                for ldx in range(self.layer_num):
                    self.cache_keys.append(self._new_late_block_cache_tensor(ldx))
                    self.cache_values.append(self._new_late_block_cache_tensor(ldx))

            for ldx in range(self.layer_num):
                # move meta index to GPU
                if self._is_index_metadata_staged_layer(ldx):
                    self.centroids[ldx] = self._pin_if_staged(ldx, self.centroids[ldx])
                    self.value_sum[ldx] = self._pin_if_staged(ldx, self.value_sum[ldx])
                    self.centroids_mask[ldx] = self._pin_if_staged(ldx, self.centroids_mask[ldx])
                    self.cluster_size[ldx] = self._pin_if_staged(ldx, self.cluster_size[ldx])
                    if self.cluster_size_cumsum[ldx] is not None:
                        self.cluster_size_cumsum[ldx] = self._pin_if_staged(ldx, self.cluster_size_cumsum[ldx])
                else:
                    self._migrate_index_metadata_layer_to_gpu(ldx)
            self.cache_stride = self.cache_size
            if self.index_metadata_late_migration_policy == "pinned_non_blocking":
                self._sync_late_index_metadata_migration()
            self.allocate_computation_buffer()
            if self.index_metadata_late_migration_policy != "pinned_non_blocking":
                self._sync_late_index_metadata_migration()
            if self.block_cache_slot_rotation_enabled and not self.block_cache_slot_rotation_delta_enabled:
                for ldx in range(min(self.block_cache_slot_rotation_actual_slots, self.layer_num)):
                    if self._slot_rotation_launch_h2d(ldx, "initial_prefetch_after_prepare_cache"):
                        self.block_cache_slot_rotation_initial_prefetch_count += 1
                self._slot_rotation_sync_all_transfers("initial_prefetch_after_prepare_cache")
            self.index_metadata_late_migration_prepare_window_elapsed_ms = (
                time.perf_counter() - prepare_window_start
            ) * 1000.0
            self.block_cache_allocated_after_prefill = True
            self.allocated = True
    

    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx): 
        """
        Update the key & value cache per layer during prefilling.
        Args:
            query_states: [bsz, seq_len, head_num, head_dim]
            key_states: [bsz, seq_len, group_num, head_dim]
            value_states: [bsz, seq_len, group_num, head_dim]
            layer_idx: layer index
            start_bdx: start batch index
        """    
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz <= self.prefill_bsz, f"Prefilling batch size ({bsz}) should <= {self.prefill_bsz}."
        assert seq_len <= self.input_length, f"seq_len({seq_len}) should <= input_length({self.input_length})"
        # assert group_num == self.kv_head, f"kv_head({self.kv_head}) should equal to group_num({group_num})"
        # assert head_dim == self.head_dim, f"head_dim({head_dim}) should equal to self.head_dim({self.head_dim})"

        valid_start = self.valid_start_list[start_bdx]
        
        if self.build_index_when_prefilling:
            # sync for the previous layer and batch finish their page organization
            if layer_idx > 0:
                self.wave_buffer[layer_idx-1].construction_sync()
            elif start_bdx > 0: # layer_idx == 0
                self.wave_buffer[self.layer_num-1].construction_sync()
            
            # store in `self` to avoid deleting when async offload to CPU, shape: (bsz*group_num, seq_len, dim)
            self.temp_keys = key_states[:, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(1, 2).reshape(bsz*self.kv_head, -1, self.head_dim).contiguous()
            self.temp_values = value_states[:, valid_start+self.static_pattern_start:seq_len-self.static_pattern_end, :, :].transpose(1, 2).reshape(bsz*self.kv_head, -1, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(layer_idx)]].record()

            # async offload keys & values to CPU
            valid_length = seq_len - self.static_pattern_total - valid_start
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(layer_idx)]].wait()
                if valid_length == self.offload_keys.shape[1]:
                    self.offload_keys[:bsz*self.kv_head, :, :].copy_(self.temp_keys, non_blocking=True)
                    self.offload_values[:bsz*self.kv_head, :, :].copy_(self.temp_values, non_blocking=True)
                else:   # loop to preserve pinned for fast copy
                    for i in range(bsz*self.kv_head):
                        self.offload_keys[i, :valid_length, :].copy_(self.temp_keys[i], non_blocking=True)
                        self.offload_values[i, :valid_length, :].copy_(self.temp_values[i], non_blocking=True)
                self.copyevents[self.layer_mapping[str(layer_idx)]].record()
            
            # copy steady zone KV
            end_bdx = start_bdx + bsz
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :self.static_pattern_start, :] = \
                key_states[:, valid_start:valid_start+self.static_pattern_start, :, :].transpose(1, 2)
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, self.static_pattern_start:self.static_pattern_total, :] = \
                key_states[:, seq_len-self.static_pattern_end:seq_len, :, :].transpose(1, 2)
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :self.static_pattern_start, :] = \
                value_states[:, valid_start:valid_start+self.static_pattern_start, :, :].transpose(1, 2)
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, self.static_pattern_start:self.static_pattern_total, :] = \
                value_states[:, seq_len-self.static_pattern_end:seq_len, :, :].transpose(1, 2)

            # compute key mean, shape (bsz*group_num, 1, head_dim)
            mean_key = torch.mean(self.temp_keys, dim=1, keepdim=True)

            # segmented clustering
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=self.temp_keys-mean_key,    # centering to 0
                value=self.temp_values,
                num_centroids=self.n_centroids,
                num_segments=self.n_segment,
            )
            # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.n_centroids

            # copy meta index
            self.centroids[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :, :].copy_(_centroids + mean_key)         # (bsz*group_num, n_centroids, dim)
            self.value_sum[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :, :].copy_(_value_sum)                    # (bsz*group_num, n_centroids, dim)
            self.centroids_mask[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :].copy_(_cluster_size == 0)          # (bsz*group_num, n_centroids)
            self.cluster_size[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :].copy_(_cluster_size.to(self.dtype))  # (bsz*group_num, n_centroids)
            if self._is_stream_only_layer(layer_idx):
                self.cluster_size_cumsum[layer_idx][start_bdx*self.kv_head:end_bdx*self.kv_head, :].copy_(
                    torch.cumsum(_cluster_size, dim=-1, dtype=torch.int32)
                )

            # cluster results will be used to organize the offload KV cache
            self.cluster_size_cpu = _cluster_size.cpu().contiguous()    # (bsz*group_num, n_centroids)
            self.clusters_cpu = _clusters.cpu().contiguous()            # (bsz*group_num, n_centroids, max_cluster_size)
        else:   # do not build index during prefilling
            assert valid_start == 0, f"Requests in the same batch should have the same length."
            end_bdx = start_bdx + bsz
            # copy input KV to steady zone
            self.steady_zone_keys[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(key_states.transpose(1, 2))
            self.steady_zone_values[layer_idx][start_bdx:end_bdx, :, :seq_len, :].copy_(value_states.transpose(1, 2))
            
        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len

            if self.build_index_when_prefilling:
                if self.use_cuda_graph:
                    self.attn_func = self.sparse_attention_with_cudagraph
                else:
                    self.attn_func = self.sparse_attention
            else:
                self.static_pattern_total = seq_len

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]   # ignore mask tokens, shape: (bsz, seq_len, kv_head, dim)

    def sync(self, layer_idx, start_bdx):  
        """Wait async offloading on copystream -> organize KV on wave buffer"""
        if self.build_index_when_prefilling:
            # wait for offload finish
            self.copyevents[self.layer_mapping[str(layer_idx)]].synchronize()
            # async organize kv
            self.wave_buffer[layer_idx].async_construction(
                self.clusters_cpu,      # (bsz*group_num, n_centroids, max_cluster_size)
                self.cluster_size_cpu,  # (bsz*group_num, n_centroids)
                start_bdx
            )


    def _update_kv_cache(self):
        """Update KV cache when generate tokens exceed UPDATE_SEGMENT"""
        self._sync_all_async_cache_admissions("before_index_update")
        self.nprobe += self.UPDATE_NPROBE
        self.cluster_ids = torch.empty((self.batch_groups, self.nprobe), dtype=torch.int64, pin_memory=True).contiguous()

        for ldx in range(self.layer_num):
            torch.cuda.set_device(self.layer_mapping[str(ldx)])
            # extract update segment, shape: (batch_size*kv_head, UPDATE_SEGMENT, head_dim)
            update_keys = self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim).contiguous()
            update_values = self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_total-self.static_pattern_end, :].clone().reshape(self.batch_groups, self.UPDATE_SEGMENT, self.head_dim).contiguous()
            self.mainevents[self.layer_mapping[str(ldx)]].record()

            # move local window
            self.steady_zone_keys[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_keys[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]
            self.steady_zone_values[ldx][:, :, self.static_pattern_start:self.static_pattern_start+self.static_pattern_end, :] = \
                self.steady_zone_values[ldx][:, :, self.static_pattern_total-self.static_pattern_end:self.static_pattern_total, :]

            # async offload
            with torch.cuda.stream(self.copystream):
                self.mainevents[self.layer_mapping[str(ldx)]].wait()
                self.offload_update_keys.copy_(update_keys, non_blocking=True)
                self.offload_update_values.copy_(update_values, non_blocking=True)
                self.copyevents[self.layer_mapping[str(ldx)]].record()
            
            # compute key mean, shape (batch_size*kv_head, 1, head_dim)
            mean_key = torch.mean(update_keys, dim=1, keepdim=True)
            
            # segmented k-means
            _centroids, _value_sum, _clusters, _cluster_size = segment_k_means(
                key=update_keys-mean_key,   # centering to 0, (batch_size*kv_head, UPDATE_SEGMENT, dim)
                value=update_values,        # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                num_centroids=self.UPDATE_CENTROIDS,
                num_segments=1,
            )
            _centroids += mean_key
            # assert _centroids.shape[-2] == _value_sum.shape[-2] == _cluster_size.shape[-1] == _clusters.shape[-2] == self.UPDATE_CENTROIDS

            # append to meta index
            if self._is_index_metadata_staged_layer(ldx):
                self.centroids[ldx] = self._pin_if_staged(
                    ldx,
                    torch.cat((self.centroids[ldx], _centroids.cpu().contiguous()), dim=1),
                )
                self.value_sum[ldx] = self._pin_if_staged(
                    ldx,
                    torch.cat((self.value_sum[ldx], _value_sum.cpu().contiguous()), dim=1),
                )
                self.centroids_mask[ldx] = self._pin_if_staged(
                    ldx,
                    torch.cat((self.centroids_mask[ldx], (_cluster_size == 0).cpu().contiguous()), dim=1),
                )
                self.cluster_size[ldx] = self._pin_if_staged(
                    ldx,
                    torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype).cpu().contiguous()), dim=1),
                )
            else:
                self.centroids[ldx] = torch.cat((self.centroids[ldx], _centroids), dim=1)  # (batch_size*kv_head, new_n_centroids, dim)
                self.value_sum[ldx] = torch.cat((self.value_sum[ldx], _value_sum), dim=1)  # (batch_size*kv_head, new_n_centroids, dim)
                self.centroids_mask[ldx] = torch.cat((self.centroids_mask[ldx], _cluster_size == 0), dim=1) # (batch_size*kv_head, new_n_centroids)
                self.cluster_size[ldx] = torch.cat((self.cluster_size[ldx], _cluster_size.to(self.dtype)), dim=1) # (batch_size*kv_head, new_n_centroids)
            if self._is_stream_only_layer(ldx):
                new_cluster_cumsum = torch.cumsum(_cluster_size, dim=-1, dtype=torch.int32)
                if self._is_index_metadata_staged_layer(ldx):
                    if self.n_centroids > 0:
                        new_cluster_cumsum += self.cluster_size_cumsum[ldx][:, -1:].to(new_cluster_cumsum.device)
                    self.cluster_size_cumsum[ldx] = self._pin_if_staged(
                        ldx,
                        torch.cat((self.cluster_size_cumsum[ldx], new_cluster_cumsum.cpu().contiguous()), dim=1),
                    )
                else:
                    if self.n_centroids > 0:
                        new_cluster_cumsum += self.cluster_size_cumsum[ldx][:, -1:]
                    self.cluster_size_cumsum[ldx] = torch.cat(
                        (self.cluster_size_cumsum[ldx], new_cluster_cumsum),
                        dim=1,
                    )
            self._invalidate_index_metadata_stage(ldx)
            # assert self.centroids[ldx].shape[-2] == self.value_sum[ldx].shape[-2] == self.centroids_mask[ldx].shape[-1] == self.cluster_size[ldx].shape[-1] == self.n_centroids + self.UPDATE_CENTROIDS

            # update wave buffer
            self.copyevents[self.layer_mapping[str(ldx)]].synchronize()
            self.wave_buffer[ldx].update_kv(
                self.offload_update_keys,           # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                self.offload_update_values,         # (batch_size*kv_head, UPDATE_SEGMENT, dim)
                _clusters.cpu().contiguous(),       # (batch_size*kv_head, UPDATE_CENTROIDS, max_cluster_size)
                _cluster_size.cpu().contiguous(),   # (batch_size*kv_head, UPDATE_CENTROIDS)
                self.cluster_ids                    # (batch_size*kv_head, new_nprobe)
            )
        
        # reset current device (layer 0)
        torch.cuda.set_device(self.layer_mapping[str(0)])
        # switch to sparse attention, and update index will disable cudagraph
        assert not self.use_cuda_graph, "CUDA Graph does not support index updating."
        self.attn_func = self.sparse_attention
        
        # update n_centroids, es_cluster_num
        self.n_centroids += self.UPDATE_CENTROIDS
        self.es_cluster_num += self.UPDATE_ES
        self.max_compute_cluster_num += (self.UPDATE_NPROBE + self.UPDATE_ES)
        
        # re-allocate layer-share buffers
        for device_idx in self.device_list:
            self.gemm_o_dict[device_idx] = self._new_scratch_tensor(
                "gemm_o",
                (self.batch_size, self.kv_head, self.group_size, self.n_centroids),
                self.dtype,
                device_idx,
            )
            self.softmax_o_dict[device_idx] = self._new_scratch_tensor(
                "softmax_o",
                (self.batch_groups, self.group_size, self.n_centroids),
                self.dtype,
                device_idx,
            )
            self.norm_dict[device_idx] = self._new_scratch_tensor(
                "norm",
                (self.batch_groups, self.group_size, (self.n_centroids+256-1)//256),
                torch.float32,
                device_idx,
            )
            self.sum_dict[device_idx] = self._new_scratch_tensor(
                "sum",
                (self.batch_groups, self.group_size, (self.n_centroids+256-1)//256),
                torch.float32,
                device_idx,
            )
            self.dist_dict[device_idx] = self._new_scratch_tensor(
                "dist",
                (self.batch_groups, self.n_centroids),
                self.dtype,
                device_idx,
            )
            self.cI_dict[device_idx] = self._new_scratch_tensor(
                "cI",
                (self.batch_groups, self.max_compute_cluster_num),
                torch.int64,
                device_idx,
            )
            self.cV_dict[device_idx] = self._new_scratch_tensor(
                "cV",
                (self.batch_groups, self.max_compute_cluster_num),
                self.dtype,
                device_idx,
            )
            self.es_centroids_dict[device_idx] = self._new_scratch_tensor(
                "es_centroids",
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.es_value_sum_dict[device_idx] = self._new_scratch_tensor(
                "es_value_sum",
                (self.batch_groups, self.es_cluster_num, 1, self.head_dim),
                self.dtype,
                device_idx,
            )
            self.es_cluster_size_dict[device_idx] = self._new_scratch_tensor(
                "es_cluster_size",
                (self.batch_groups, 1, 1, self.es_cluster_num),
                self.dtype,
                device_idx,
            )
            self.nprobe_tensor_dict[device_idx].fill_(self.nprobe)
        self._allocate_index_metadata_stage_buffers()
        
        # set pointers to current device (layer 0)
        self.gemm_o = self.gemm_o_dict[self.layer_mapping[str(0)]]
        self.softmax_o = self.softmax_o_dict[self.layer_mapping[str(0)]]
        self.norm = self.norm_dict[self.layer_mapping[str(0)]]
        self.sum = self.sum_dict[self.layer_mapping[str(0)]]
        self.dist = self.dist_dict[self.layer_mapping[str(0)]]
        self.cI = self.cI_dict[self.layer_mapping[str(0)]]
        self.cV = self.cV_dict[self.layer_mapping[str(0)]]
        self.es_centroids = self.es_centroids_dict[self.layer_mapping[str(0)]]
        self.es_value_sum = self.es_value_sum_dict[self.layer_mapping[str(0)]]
        self.es_cluster_size = self.es_cluster_size_dict[self.layer_mapping[str(0)]]
        self.nprobe_tensor = self.nprobe_tensor_dict[self.layer_mapping[str(0)]]
        
        # reset static pattern length
        self.static_pattern_total = self.static_pattern_start + self.static_pattern_end

        print(f"nprobe: {self.nprobe}, es_cluster_num: {self.es_cluster_num}, max_compute_cluster_num: {self.max_compute_cluster_num}, n_centroids: {self.n_centroids}")


    def _gather_stream_only_execution_buffer(self, layer_idx, cluster_size_cumsum=None):
        if cluster_size_cumsum is None:
            _, _, _, _, cluster_size_cumsum = self._index_metadata_for_attention(layer_idx)
        self._launch_stream_only_execution_buffer_gather(
            layer_idx,
            cluster_size_cumsum,
            use_cuda_graph_gather=False,
        )


    def decode_update_kv_cache(
        self,
        key_states,         # (bsz, seq_len(=1), group_num, dim)
        value_states,       # (bsz, seq_len(=1), group_num, dim)
        layer_idx
    ):
        # index update when generate tokens exceed UPDATE_SEGMENT
        if self.static_pattern_total == self.static_pattern_start + self.static_pattern_end + self.UPDATE_SEGMENT:
            self._update_kv_cache()

        self._slot_rotation_before_layer(layer_idx)

        # append newly generated token to the steady zone
        self.steady_zone_keys[layer_idx][:, :, self.static_pattern_total, :] = key_states[:, 0, :, :]
        self.steady_zone_values[layer_idx][:, :, self.static_pattern_total, :] = value_states[:, 0, :, :]

        if layer_idx == self.layer_num - 1:
            self.context += 1
            self.static_pattern_total += 1

        return None, None   # not use the return value


    def dense_attention(self, queries, layer_idx, static_len):
        """
        Full Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        attn_out = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                self.steady_zone_keys[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                self.steady_zone_values[layer_idx].view(self.batch_groups, -1, 1, self.head_dim),
                previous_out=None, previous_lse=None,
                cache_seqlens=static_len,
                return_softmax_lse=False
            )
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)

    
    def sparse_attention(self, queries, layer_idx, static_len):
        """
        Sparse Attention
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)
        centroids, value_sum, centroids_mask, cluster_size, cluster_size_cumsum = (
            self._index_metadata_for_attention(layer_idx)
        )

        # Softmax(QC^T) -> [batch_size*group_num, group_size, n_centroids]
        batch_gemm_softmax(queries, centroids, self.gemm_o, self.norm, self.sum, self.softmax_o,
                           self.batch_groups, self.group_size, self.n_centroids, self.head_dim, self.RSQRT_DIM, 0)
        torch.sum(self.softmax_o, dim=1, out=self.dist)  # Merge groups -> [batch_size*group_num, n_centroids]
        self.dist.masked_fill_(centroids_mask, self.DTYPE_MIN)  # mask empty clusters
        torch.topk(self.dist, self.max_compute_cluster_num, dim=-1, largest=True, sorted=True, out=(self.cV, self.cI))
        device_idx = self.layer_mapping[str(layer_idx)]
        stream_only_layer = self._is_stream_only_layer(layer_idx)
        if not stream_only_layer:
            self._launch_cluster_ids_copy(self.cI[..., :self.nprobe], device_idx)
            self._launch_async_wave_batch_access(layer_idx, device_idx)
        elif self.async_stream_only_gather_enabled:
            self._launch_stream_only_execution_buffer_gather(
                layer_idx,
                cluster_size_cumsum,
                use_cuda_graph_gather=False,
            )

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            gather_copy_vectors(
                centroids, self.es_centroids,
                value_sum, self.es_value_sum,
                cluster_size, self.es_cluster_size,
                self.cI, self.batch_groups, self.n_centroids, self.es_cluster_num, 
                self.max_compute_cluster_num, self.nprobe, self.es_cluster_num
            )
            
            es_out, es_lse = weighted_flash_decoding(
                                queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
                                self.es_centroids,       # [batch_size*group_num, es_cluster_num, 1, dim]
                                self.es_value_sum,       # [batch_size*group_num, es_cluster_num, 1, dim]
                                self.es_cluster_size,    # [batch_size*group_num, 1, 1, es_cluster_num]
                                previous_out=None, previous_lse=None,
                                return_softmax_lse=True
                            )
        else:
            es_out, es_lse = None, None
        if not stream_only_layer:
            self._prefetch_next_index_metadata_layer(layer_idx)

        if stream_only_layer:
            if self.async_stream_only_gather_enabled:
                self._sync_async_stream_only_gather_for_device(
                    device_idx,
                    "before_stream_only_weighted_flash_decoding",
                )
            else:
                self._gather_stream_only_execution_buffer(layer_idx, cluster_size_cumsum)
            self._prefetch_next_index_metadata_layer(layer_idx)
            attn_out = weighted_flash_decoding(
                queries.view(self.batch_groups, 1, self.group_size, self.head_dim),
                self.execution_buffer_keys,
                self.execution_buffer_values,
                previous_out=es_out,
                previous_lse=es_lse,
                cache_seqlens=self.valid_lengths,
                return_softmax_lse=False
            )
            return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)
        
        # access cache and submit cache update jobs to thread pool
        self._finish_wave_batch_access(
            layer_idx,
            device_idx,
            "before_sparse_attention_execution_buffer_write",
        )

        # assemble the execution buffer
        self._sync_pending_admission_before_execution_buffer_read(
            layer_idx,
            "before_sparse_attention_execution_buffer_write",
        )
        self._slot_rotation_materialize_hit_pages(layer_idx)
        gather_copy_and_concat(
            self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx], self.execution_buffer_keys, 
            self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx], self.execution_buffer_values,
            self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
            self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
            self.valid_lengths, self.batch_groups, self.static_stride, self.list_stride, self.cache_strides[layer_idx], self.execution_stride,
            self.buffer_size, self.static_len_tensor
        )

        # attention for retrieve zone and steady zone, merge the estimation zone results at the same time
        attn_out = weighted_flash_decoding(
            queries.view(self.batch_groups, 1, self.group_size, self.head_dim), 
            self.execution_buffer_keys,    # (batch_size*group_num, execution_stride, 1, dim)
            self.execution_buffer_values,  # (batch_size*group_num, execution_stride, 1, dim)
            previous_out=es_out,
            previous_lse=es_lse,
            cache_seqlens=self.valid_lengths,  # valid lengths of retrieve zone + steady zone for each group
            return_softmax_lse=False
        )

        # admit pages from execution buffer to GPU block cache
        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        self._record_block_cache_telemetry(layer_idx)
        self._launch_cache_admission_update(layer_idx, use_cuda_graph_update=False)
        self._slot_rotation_after_layer(layer_idx)
        
        return attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim)


    def sparse_attention_with_cudagraph(self, queries, layer_idx, static_len):
        """
        Sparse Attention with CUDA graph
        Args:
            queries: query vector, shape: (batch_size, 1, head_num, dim), gpu torch tensor
            layer_idx: layer index
            static_len: valid length of steady zone
        """
        self.static_len_tensor.fill_(static_len)
        self._index_metadata_for_attention(layer_idx)
        self.query_buffer.copy_(queries.view(self.batch_groups, 1, self.group_size, self.head_dim), non_blocking=True)
        
        # get topk clusters
        self.topk_cudagraphs[layer_idx].replay()
        device_idx = self.layer_mapping[str(layer_idx)]
        stream_only_layer = self._is_stream_only_layer(layer_idx)
        if not stream_only_layer:
            self._launch_cluster_ids_copy(self.cI[..., :self.nprobe], device_idx)
            self._launch_async_wave_batch_access(layer_idx, device_idx)
        elif self.async_stream_only_gather_enabled:
            self._launch_stream_only_execution_buffer_gather(
                layer_idx,
                cluster_size_cumsum=None,
                use_cuda_graph_gather=True,
            )

        # estimation zone attention computation
        if self.es_cluster_num > 0:
            self.es_cudagraphs[layer_idx].replay()

        if stream_only_layer:
            if self.async_stream_only_gather_enabled:
                self._sync_async_stream_only_gather_for_device(
                    device_idx,
                    "before_stream_only_cudagraph_weighted_flash_decoding",
                )
            else:
                self._sync_pending_admission_before_execution_buffer_read(
                    layer_idx,
                    "before_stream_only_cudagraph_execution_buffer_write",
                )
            self.attn_cudagraphs[layer_idx].replay()
            self._prefetch_next_index_metadata_layer(layer_idx)
            return self.attn_out
        self._prefetch_next_index_metadata_layer(layer_idx)
        
        # access cache and submit cache update jobs to thread pool
        self._finish_wave_batch_access(
            layer_idx,
            device_idx,
            "before_sparse_attention_cudagraph_execution_buffer_write",
        )

        # compute attention for retrieve zone and steady zone, merge estimation zone results
        self._sync_pending_admission_before_execution_buffer_read(
            layer_idx,
            "before_sparse_attention_cudagraph_execution_buffer_write",
        )
        self._slot_rotation_materialize_hit_pages(layer_idx)
        self.attn_cudagraphs[layer_idx].replay()

        self.wave_buffer[layer_idx].sync()  # wait for update LRU finish
        self._record_block_cache_telemetry(layer_idx)
        # admit pages from execution buffer to GPU cache
        self._launch_cache_admission_update(layer_idx, use_cuda_graph_update=True)
        self._slot_rotation_after_layer(layer_idx)

        return self.attn_out
    

    def capture_cuda_graph(self):
        """Capture CUDA Graph"""
        if not self.use_cuda_graph:
            return
        
        print("Capture CUDA graph ...")
        for layer_idx in range(self.layer_num):
            with torch.cuda.device(self.layer_mapping[str(layer_idx)]):
                capture_stream = torch.cuda.Stream(device=self.layer_mapping[str(layer_idx)])
                centroids, value_sum, centroids_mask, cluster_size, cluster_size_cumsum = self._index_metadata_for_attention(layer_idx)

                # TopK search CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(self.topk_cudagraphs[layer_idx], stream=capture_stream):
                    batch_gemm_softmax(
                        self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                        centroids,
                        self.gemm_o_dict[self.layer_mapping[str(layer_idx)]], 
                        self.norm_dict[self.layer_mapping[str(layer_idx)]], 
                        self.sum_dict[self.layer_mapping[str(layer_idx)]], 
                        self.softmax_o_dict[self.layer_mapping[str(layer_idx)]],
                        self.batch_groups, self.group_size, self.n_centroids, self.head_dim, 
                        self.RSQRT_DIM, 0
                    )
                    torch.sum(self.softmax_o_dict[self.layer_mapping[str(layer_idx)]], dim=1, 
                              out=self.dist_dict[self.layer_mapping[str(layer_idx)]])
                    self.dist_dict[self.layer_mapping[str(layer_idx)]].masked_fill_(centroids_mask, self.DTYPE_MIN)
                    torch.topk(self.dist_dict[self.layer_mapping[str(layer_idx)]], self.max_compute_cluster_num, 
                               dim=-1, largest=True, sorted=True, 
                               out=(self.cV_dict[self.layer_mapping[str(layer_idx)]], self.cI_dict[self.layer_mapping[str(layer_idx)]]))

                # Estimation zone CUDA graph
                if self.es_cluster_num > 0:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self.es_cudagraphs[layer_idx], stream=capture_stream):
                        gather_copy_vectors(
                            centroids, self.es_centroids_dict[self.layer_mapping[str(layer_idx)]],
                            value_sum, self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]],
                            cluster_size, self.es_cluster_size_dict[self.layer_mapping[str(layer_idx)]],
                            self.cI_dict[self.layer_mapping[str(layer_idx)]], 
                            self.batch_groups, self.n_centroids, self.es_cluster_num, 
                            self.max_compute_cluster_num, self.nprobe, self.es_cluster_num
                        )
                        # TODO: add output API in this kernel
                        es_out, es_lse = weighted_flash_decoding(
                                            self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                                            self.es_centroids_dict[self.layer_mapping[str(layer_idx)]],
                                            self.es_value_sum_dict[self.layer_mapping[str(layer_idx)]],
                                            self.es_cluster_size_dict[self.layer_mapping[str(layer_idx)]],
                                            previous_out=None, previous_lse=None,
                                            return_softmax_lse=True
                                        )
                        self.es_out_dict[self.layer_mapping[str(layer_idx)]].copy_(es_out, non_blocking=True)
                        self.es_lse_dict[self.layer_mapping[str(layer_idx)]].copy_(es_lse, non_blocking=True)

                if self._is_stream_only_layer(layer_idx) and self.async_stream_only_gather_enabled:
                    torch.cuda.synchronize()
                    with torch.cuda.graph(
                        self.stream_only_gather_cudagraphs[layer_idx],
                        stream=capture_stream,
                    ):
                        gather_copy_cluster_and_concat_fuse(
                            self.steady_zone_keys[layer_idx], self.list_keys[layer_idx],
                            self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]],
                            self.steady_zone_values[layer_idx], self.list_values[layer_idx],
                            self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                            cluster_size_cumsum,
                            self.cI_dict[self.layer_mapping[str(layer_idx)]],
                            self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]],
                            self.batch_groups, self.static_stride, self.list_stride, self.execution_stride,
                            self.nprobe, self.nprobe_tensor_dict[self.layer_mapping[str(layer_idx)]],
                            self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                        )

                # Retrieval and Steady zone CUDA graph
                torch.cuda.synchronize()
                with torch.cuda.graph(self.attn_cudagraphs[layer_idx], stream=capture_stream):
                    if self._is_stream_only_layer(layer_idx):
                        if self.async_stream_only_gather_enabled:
                            pass
                        else:
                            gather_copy_cluster_and_concat_fuse(
                                self.steady_zone_keys[layer_idx], self.list_keys[layer_idx],
                                self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]],
                                self.steady_zone_values[layer_idx], self.list_values[layer_idx],
                                self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                                cluster_size_cumsum,
                                self.cI_dict[self.layer_mapping[str(layer_idx)]],
                                self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]],
                                self.batch_groups, self.static_stride, self.list_stride, self.execution_stride,
                                self.nprobe, self.nprobe_tensor_dict[self.layer_mapping[str(layer_idx)]],
                                self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                            )
                    else:
                        gather_copy_and_concat(
                            self.steady_zone_keys[layer_idx], self.list_keys[layer_idx], self.cache_keys[layer_idx],
                            self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]],
                            self.steady_zone_values[layer_idx], self.list_values[layer_idx], self.cache_values[layer_idx],
                            self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                            self.miss_unit_idices[layer_idx], self.miss_unit_sizes[layer_idx], self.miss_unit_sizes_cumsum[layer_idx], self.miss_num_units[layer_idx],
                            self.hit_unit_idices[layer_idx], self.hit_unit_sizes[layer_idx], self.hit_unit_sizes_cumsum[layer_idx], self.hit_num_units[layer_idx],
                            self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]], self.batch_groups, self.static_stride, self.list_stride, self.cache_strides[layer_idx],
                            self.execution_stride, self.buffer_size, self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                        )
                    # TODO: add output API in this kernel
                    attn_out = weighted_flash_decoding(
                                    self.query_buffer_dict[self.layer_mapping[str(layer_idx)]], 
                                    self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]],
                                    self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]],
                                    previous_out=self.es_out_dict[self.layer_mapping[str(layer_idx)]],
                                    previous_lse=self.es_lse_dict[self.layer_mapping[str(layer_idx)]],
                                    cache_seqlens=self.valid_lengths_dict[self.layer_mapping[str(layer_idx)]],
                                    return_softmax_lse=False
                                )
                    self.attn_out_dict[self.layer_mapping[str(layer_idx)]].copy_(attn_out.view(self.batch_size, 1, self.num_heads, self.head_dim), non_blocking=True)
                        
                # Cache update CUDA graph
                if not self._is_stream_only_layer(layer_idx):
                    torch.cuda.synchronize()
                    with torch.cuda.graph(self.update_cudagraphs[layer_idx], stream=capture_stream):
                        gather_copy_and_scatter(
                            self.execution_buffer_keys_dict[self.layer_mapping[str(layer_idx)]], self.cache_keys[layer_idx],
                            self.execution_buffer_values_dict[self.layer_mapping[str(layer_idx)]], self.cache_values[layer_idx],
                            self.update_buffer_indices[layer_idx], self.update_unit_sizes[layer_idx],
                            self.update_cache_indices[layer_idx], self.update_num_units[layer_idx],
                            self.batch_groups, self.execution_stride, self.cache_strides[layer_idx], self.buffer_size,
                            self.static_len_tensor_dict[self.layer_mapping[str(layer_idx)]]
                        )
                
                torch.cuda.synchronize()
        self._slot_rotation_reset_after_cuda_graph_capture()