import argparse
import math
from typing import Literal

import numpy as np
import torch
import torch_npu

import vllm_ascend.vllm_ascend_C  # noqa: F401
from memfabric_hybrid import offload
from vllm_ascend.utils import enable_custom_op

assert enable_custom_op()
torch_npu.npu.set_device(0)
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
fused = torch.ops._C_ascend.npu_fused_sparse_attention_overlap
empty_swapped = getattr(torch_npu, "empty_with_swapped_memory", None)

CpuKvBackend = Literal["offload", "swap_memory"]

# Match KVOffloadDecodeManager CPU KV pool alignment / init.
_CPU_CACHE_ALIGNMENT = 2 * 1024 * 1024
_GIB = 1024 * 1024 * 1024

# MLA dims used by this UT (same as DeepSeek-style sparse MLA).
_BLOCK_SIZE = 128
_KVD = 512  # nope / kv_lora_rank
_KRD = 64  # rope
_DTYPE_SIZE = 2  # bf16


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset) :]


def _empty_aligned_cpu_tensor(
    shape: list[int],
    dtype: torch.dtype,
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> torch.Tensor:
    """Same allocation path as KVOffloadDecodeManager._empty_aligned_cpu_tensor."""
    num_elements = int(np.prod(shape))
    extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
    tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
    return _align_memory(tensor, alignment)[:num_elements].view(shape)


def init_offload_pool(
    dram_size_gb: float = 1.0,
    world_size: int = 1,
    rank_id: int = 0,
) -> None:
    """Mirror kv_offload_decode_manager MemFabric OffloadConfig initialize."""
    config = offload.OffloadConfig()
    config.device_id = torch_npu.npu.current_device()
    config.size = int(dram_size_gb * _GIB)
    config.world_size = world_size
    config.rank_id = rank_id
    print(
        "OFFLOAD_INIT device_id={} size_gb={} world_size={} rank_id={}".format(
            config.device_id, dram_size_gb, world_size, rank_id
        ),
        flush=True,
    )
    offload.initialize(config)


def uninit_offload_pool() -> None:
    offload.uninitialize()


def estimate_cpu_kv_bytes(max_seq_len: int, block_size: int = _BLOCK_SIZE) -> int:
    """Bytes for one-layer full K + rope CPU cache covering max_seq_len tokens."""
    num_blocks = cdiv(max_seq_len, block_size)
    k_bytes = num_blocks * block_size * _KVD * _DTYPE_SIZE
    rope_bytes = num_blocks * block_size * _KRD * _DTYPE_SIZE
    # 2MiB align padding per tensor (upper bound).
    align_pad = 2 * _CPU_CACHE_ALIGNMENT
    return k_bytes + rope_bytes + align_pad


def recommended_dram_size_gb(max_seq_len: int, block_size: int = _BLOCK_SIZE) -> float:
    need_bytes = estimate_cpu_kv_bytes(max_seq_len, block_size)
    # Round up to next 0.5 GiB with a small headroom.
    return math.ceil((need_bytes / _GIB) * 2) / 2.0 + 0.5


def _alloc_cpu_kv_tensor(
    backend: CpuKvBackend,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate host-side KV used by fused op.

    - offload: MemFabric ``offload.empty`` + 2MiB align (same as KVOffloadDecodeManager)
    - swap_memory: ``torch_npu.empty_with_swapped_memory``
    """
    if backend == "offload":
        return _empty_aligned_cpu_tensor(shape, dtype=dtype)
    if backend == "swap_memory":
        if empty_swapped is None:
            raise RuntimeError(
                "cpu-kv-backend=swap_memory requires torch_npu.empty_with_swapped_memory"
            )
        return empty_swapped(tuple(shape), dtype=dtype, device="npu")
    raise ValueError(f"unknown cpu-kv-backend: {backend}")


def _build_inputs(
    topk: int,
    q_heads: int,
    kv_heads: int = 1,
    cpu_kv_backend: CpuKvBackend = "offload",
    max_seq_len: int | None = None,
):
    assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
    assert kv_heads == 1, "fused op currently requires kv_heads=1"
    assert topk > 0
    assert cpu_kv_backend in ("offload", "swap_memory")

    bsz = 1
    seq = 1
    hd, krd, kvd = _KVD, _KRD, _KVD
    sbs = fbs = _BLOCK_SIZE
    stbs = 1
    # Default legacy sizing; for 1M long-context set --max-seq-len 1048576.
    msl = max_seq_len if max_seq_len is not None else max(topk * 4, 256)
    assert msl >= topk, f"max_seq_len ({msl}) must be >= topk ({topk})"
    fmbn = cdiv(msl, fbs)
    smbn = cdiv(topk * stbs, sbs)
    dt = torch.bfloat16
    scale = 1.0 / math.sqrt(kvd)
    torch.manual_seed(910000 + topk + q_heads * 17)
    np.random.seed((910000 + topk + q_heads * 17) % (2**32 - 1))

    cpu_kv_bytes = estimate_cpu_kv_bytes(msl, fbs)
    print(
        "CPU_KV_PLAN backend={} max_seq_len={} block_size={} num_blocks={} "
        "k_shape=[{}, {}, {}] rope_shape=[{}, {}, {}] est_bytes={:.3f} GiB".format(
            cpu_kv_backend,
            msl,
            fbs,
            fmbn,
            fmbn,
            fbs,
            kvd,
            fmbn,
            fbs,
            krd,
            cpu_kv_bytes / _GIB,
        ),
        flush=True,
    )

    q = torch.randn(bsz * seq, q_heads, hd, dtype=dt, device="npu")
    q_rope = torch.randn(bsz * seq, q_heads, krd, dtype=dt, device="npu")
    q_fused = torch.cat([q, q_rope], dim=-1).contiguous()

    kv_shape = [fmbn * bsz, fbs, kvd]
    rope_shape = [fmbn * bsz, fbs, krd]
    full_kv_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, kv_shape, dt)
    full_rope_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, rope_shape, dt)
    full_kv_fused.copy_(torch.randn(kv_shape, dtype=dt, device="cpu"))
    full_rope_fused.copy_(torch.randn(rope_shape, dtype=dt, device="cpu"))

    print(
        "FULL_KV_HOST backend={} shape={} rope_shape={} dtype={} "
        "device={} ptr_align={} bytes={:.3f} GiB".format(
            cpu_kv_backend,
            tuple(full_kv_fused.shape),
            tuple(full_rope_fused.shape),
            dt,
            full_kv_fused.device,
            (
                full_kv_fused.data_ptr() % _CPU_CACHE_ALIGNMENT == 0
                if cpu_kv_backend == "offload"
                else "n/a"
            ),
            (
                full_kv_fused.numel() * full_kv_fused.element_size()
                + full_rope_fused.numel() * full_rope_fused.element_size()
            )
            / _GIB,
        ),
        flush=True,
    )

    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(bsz, -1).contiguous()
    actual_q = torch.tensor([seq] * bsz, dtype=torch.int32, device="npu")
    actual_k = torch.tensor([msl] * bsz, dtype=torch.int32, device="npu")
    topk_np = np.zeros((bsz * seq, kv_heads, topk), dtype=np.int32)
    for row in range(bsz * seq):
        for kh in range(kv_heads):
            topk_np[row, kh] = np.sort(np.random.choice(msl, topk, replace=False).astype(np.int32))
    topk_fused = torch.tensor(topk_np, dtype=torch.int32, device="npu").reshape(
        bsz, seq, kv_heads, topk
    ).contiguous()

    total_sel = smbn * bsz * seq * kv_heads
    sel_kv = torch.zeros(total_sel, sbs, kvd, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, sbs, krd, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(bsz * seq * kv_heads, smbn)
    sel_status = torch.full((bsz, seq, kv_heads, topk + 1), -1, dtype=torch.int32, device="npu")

    return {
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "topk": topk,
        "msl": msl,
        "num_blocks": fmbn,
        "stbs": stbs,
        "scale": scale,
        "q_fused": q_fused,
        "full_kv_fused": full_kv_fused,
        "full_rope_fused": full_rope_fused,
        "full_bt": full_bt,
        "actual_q": actual_q,
        "actual_k": actual_k,
        "topk_fused": topk_fused,
        "sel_kv": sel_kv,
        "sel_rope": sel_rope,
        "sel_bt": sel_bt,
        "sel_status": sel_status,
    }


def _call_fused(inp):
    return fused(
        inp["q_fused"],
        inp["sel_rope"],
        inp["sel_kv"],
        inp["sel_bt"],
        inp["sel_status"],
        inp["topk_fused"],
        inp["full_rope_fused"],
        inp["full_kv_fused"],
        inp["full_bt"],
        inp["actual_k"],
        inp["actual_q"],
        inp["scale"],
        1,
        inp["stbs"],
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
    )


def _bench_fused(inp, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = _call_fused(inp)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = _call_fused(inp)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def run_case(
    topk: int,
    q_heads: int,
    warmup: int,
    iters: int,
    cpu_kv_backend: CpuKvBackend = "offload",
    max_seq_len: int | None = None,
) -> float:
    inp = _build_inputs(
        topk,
        q_heads,
        cpu_kv_backend=cpu_kv_backend,
        max_seq_len=max_seq_len,
    )
    print(
        "CASE topk={} q_heads={} kv_heads={} max_seq_len={} num_blocks={} "
        "cpu_kv_backend={} q_shape={} topk_shape={}".format(
            topk,
            q_heads,
            inp["kv_heads"],
            inp["msl"],
            inp["num_blocks"],
            cpu_kv_backend,
            tuple(inp["q_fused"].shape),
            tuple(inp["topk_fused"].shape),
        ),
        flush=True,
    )

    avg_ms = _bench_fused(inp, warmup=warmup, iters=iters)
    print(
        "BENCH topk={} q_heads={} max_seq_len={} warmup={} iters={} avg_ms={:.3f}".format(
            topk, q_heads, inp["msl"], warmup, iters, avg_ms
        ),
        flush=True,
    )
    return avg_ms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fused sparse attention overlap timing UT"
    )
    parser.add_argument(
        "--topk",
        default="2048",
        help="Comma-separated topk list, e.g. 64,1024,2048",
    )
    parser.add_argument(
        "--q-heads",
        default="16,4",
        help="Comma-separated q head counts, e.g. 16,4",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Full KV sequence length covered by CPU cache (tokens). "
        "Default: max(topk*4, 256). For 1M long context use 1048576.",
    )
    parser.add_argument(
        "--cpu-kv-backend",
        choices=("offload", "swap_memory"),
        default="offload",
        help="CPU/host KV init: offload=MemFabric offload.empty (default); "
        "swap_memory=torch_npu.empty_with_swapped_memory",
    )
    parser.add_argument(
        "--dram-size-gb",
        type=float,
        default=None,
        help="Only for --cpu-kv-backend=offload: MemFabric OffloadConfig.size in GB "
        "(same as kv_offload_decode_config.dram_size_per_dp_GB). "
        "Default: auto from --max-seq-len (1M -> ~2GiB).",
    )
    # Backward-compatible: bare `script.py 64,1024` still works.
    parser.add_argument("legacy_topk", nargs="?", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    topk_str = args.legacy_topk if args.legacy_topk else args.topk
    cases = [int(x) for x in topk_str.split(",") if x]
    q_heads_list = [int(x) for x in args.q_heads.split(",") if x]
    cpu_kv_backend: CpuKvBackend = args.cpu_kv_backend
    max_seq_len = args.max_seq_len
    plan_msl = max_seq_len if max_seq_len is not None else max(max(cases) * 4, 256)
    need_bytes = estimate_cpu_kv_bytes(plan_msl)
    dram_size_gb = args.dram_size_gb
    if cpu_kv_backend == "offload":
        if dram_size_gb is None:
            dram_size_gb = recommended_dram_size_gb(plan_msl)
        if dram_size_gb * _GIB < need_bytes:
            raise ValueError(
                f"dram_size_gb={dram_size_gb} GiB is too small for max_seq_len={plan_msl} "
                f"(need >= {need_bytes / _GIB:.3f} GiB; try --dram-size-gb "
                f"{recommended_dram_size_gb(plan_msl)})"
            )
    elif empty_swapped is None:
        raise RuntimeError(
            "cpu-kv-backend=swap_memory but torch_npu.empty_with_swapped_memory is unavailable"
        )

    print(
        f"{cases=} {q_heads_list=} max_seq_len={plan_msl} "
        f"num_blocks={cdiv(plan_msl, _BLOCK_SIZE)} "
        f"warmup={args.warmup} iters={args.iters} "
        f"cpu_kv_backend={cpu_kv_backend} dram_size_gb={dram_size_gb} "
        f"cpu_kv_est_gib={need_bytes / _GIB:.3f}",
        flush=True,
    )

    if cpu_kv_backend == "offload":
        init_offload_pool(dram_size_gb=dram_size_gb)

    results = []
    try:
        for topk in cases:
            for q_heads in q_heads_list:
                avg_ms = run_case(
                    topk,
                    q_heads,
                    warmup=args.warmup,
                    iters=args.iters,
                    cpu_kv_backend=cpu_kv_backend,
                    max_seq_len=max_seq_len,
                )
                results.append((topk, q_heads, avg_ms))
    finally:
        if cpu_kv_backend == "offload":
            try:
                uninit_offload_pool()
            except Exception as exc:
                print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    print("----- TIMING SUMMARY -----", flush=True)
    for topk, q_heads, avg_ms in results:
        print(
            "topk={} q_heads={} avg_ms={:.3f}".format(topk, q_heads, avg_ms),
            flush=True,
        )


if __name__ == "__main__":
    main()
