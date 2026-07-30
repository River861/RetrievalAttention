import json
import sys

import torch
from retroinfer_kernels import gather_copy_cluster_and_concat_fuse


DTYPE = torch.bfloat16


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for gather_copy_cluster_and_concat_fuse")

    torch.manual_seed(20260729)
    groups = 8
    dim = 128
    static_vectors = 17
    list_vectors = 96
    n_centroids = 16
    nprobe = 5
    buffer_vectors = static_vectors + 48

    key_static = torch.randn((groups, static_vectors + 3, dim), device="cuda", dtype=DTYPE).contiguous()
    value_static = torch.randn((groups, static_vectors + 3, dim), device="cuda", dtype=DTYPE).contiguous()
    key_list = torch.randn((groups, list_vectors, dim), pin_memory=True, dtype=DTYPE).contiguous()
    value_list = torch.randn((groups, list_vectors, dim), pin_memory=True, dtype=DTYPE).contiguous()
    key_buffer = torch.empty((groups, buffer_vectors, dim), device="cuda", dtype=DTYPE).contiguous()
    value_buffer = torch.empty((groups, buffer_vectors, dim), device="cuda", dtype=DTYPE).contiguous()

    cluster_sizes_cpu = torch.zeros((groups, n_centroids), dtype=torch.int32)
    for group_idx in range(groups):
        sizes = torch.tensor(
            [3, 7, 0, 11, 5, 9, 2, 8, 4, 6, 1, 10, 12, 0, 13, 5],
            dtype=torch.int32,
        )
        cluster_sizes_cpu[group_idx] = torch.roll(sizes, group_idx)
    assert int(cluster_sizes_cpu.sum(dim=1).max()) <= list_vectors
    cluster_cumsum = torch.cumsum(cluster_sizes_cpu.to("cuda"), dim=-1, dtype=torch.int32)

    select_indices = torch.tensor(
        [
            [0, 3, 5, 9, 14, 1, 2],
            [15, 7, 1, 4, 11, 0, 3],
            [2, 8, 12, 5, 0, 4, 6],
            [6, 9, 13, 3, 15, 1, 2],
            [10, 0, 5, 11, 7, 4, 8],
            [14, 2, 6, 9, 1, 12, 0],
            [4, 15, 3, 8, 11, 5, 2],
            [12, 1, 10, 6, 0, 14, 3],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    nprobe_tensor = torch.tensor(nprobe, dtype=torch.int32, device="cuda")
    static_tensor = torch.tensor(static_vectors, dtype=torch.int32, device="cuda")
    valid_lengths = torch.empty((groups,), dtype=torch.int32, device="cuda")

    gather_copy_cluster_and_concat_fuse(
        key_static,
        key_list,
        key_buffer,
        value_static,
        value_list,
        value_buffer,
        cluster_cumsum,
        select_indices,
        valid_lengths,
        groups,
        key_static.shape[1],
        list_vectors,
        buffer_vectors,
        nprobe,
        nprobe_tensor,
        static_tensor,
    )
    torch.cuda.synchronize()

    expected_keys = key_buffer.clone()
    expected_values = value_buffer.clone()
    expected_lengths = []
    cluster_cumsum_cpu = cluster_cumsum.cpu()
    select_indices_cpu = select_indices.cpu()
    for group_idx in range(groups):
        expected_keys[group_idx, :static_vectors].copy_(key_static[group_idx, :static_vectors])
        expected_values[group_idx, :static_vectors].copy_(value_static[group_idx, :static_vectors])
        copy_offset = static_vectors
        for probe_idx in range(nprobe):
            cluster_id = int(select_indices_cpu[group_idx, probe_idx])
            start = 0 if cluster_id == 0 else int(cluster_cumsum_cpu[group_idx, cluster_id - 1])
            end = int(cluster_cumsum_cpu[group_idx, cluster_id])
            copy_size = min(end - start, buffer_vectors - copy_offset)
            if copy_size == 0:
                continue
            if copy_size < 0:
                break
            expected_keys[group_idx, copy_offset:copy_offset + copy_size].copy_(
                key_list[group_idx, start:start + copy_size].to("cuda")
            )
            expected_values[group_idx, copy_offset:copy_offset + copy_size].copy_(
                value_list[group_idx, start:start + copy_size].to("cuda")
            )
            copy_offset += copy_size
            if copy_offset == buffer_vectors:
                break
        expected_lengths.append(copy_offset)

    expected_lengths_cuda = torch.tensor(expected_lengths, dtype=torch.int32, device="cuda")
    key_exact = torch.equal(key_buffer, expected_keys)
    value_exact = torch.equal(value_buffer, expected_values)
    lengths_exact = torch.equal(valid_lengths, expected_lengths_cuda)
    result = {
        "status": "pass" if key_exact and value_exact and lengths_exact else "fail",
        "dtype": str(DTYPE),
        "groups": groups,
        "dim": dim,
        "static_vectors": static_vectors,
        "list_vectors": list_vectors,
        "n_centroids": n_centroids,
        "nprobe": nprobe,
        "source2": "cpu_pinned_list_kv",
        "key_exact": key_exact,
        "value_exact": value_exact,
        "valid_lengths_exact": lengths_exact,
        "valid_lengths": valid_lengths.cpu().tolist(),
        "expected_lengths": expected_lengths,
        "installed_symbol": hasattr(sys.modules["retroinfer_kernels"], "gather_copy_cluster_and_concat_fuse"),
    }
    print("STREAM_ONLY_FUSED_GATHER_JSON=" + json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
