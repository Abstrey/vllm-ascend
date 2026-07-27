import argparse
import gc
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
    """Same allocation path as KVOffloadDecodeManager / model_runner CPU KV.

    Allocate raw int8 bytes from MemFabric, align, then view as ``dtype``.
    Avoids ``offload.empty(..., bfloat16)`` which some MemFabric builds reject,
    and matches production ``int8`` pool then ``.view(dtype).view(shape)``.
    """
    num_elements = int(np.prod(shape))
    nbytes = num_elements * torch.empty((), dtype=dtype).element_size()
    extra_bytes = alignment
    print(
        "OFFLOAD_EMPTY shape={} dtype={} nbytes={:.3f} GiB (+align {:.3f} MiB)".format(
            shape, dtype, nbytes / _GIB, extra_bytes / (1024 * 1024)
        ),
        flush=True,
    )
    try:
        raw = offload.empty([nbytes + extra_bytes], dtype=torch.int8, pin_memory=True)
    except Exception as exc:
        raise RuntimeError(
            f"offload.empty failed for {(nbytes + extra_bytes) / _GIB:.3f} GiB "
            f"(shape={shape}, dtype={dtype}): {type(exc).__name__}: {exc}. "
            "Check MemFabric pool size (--dram-size-gb) and host free/locked memory; "
            "avoid a second full host copy via torch.randn."
        ) from exc
    aligned = _align_memory(raw, alignment)[:nbytes]
    return aligned.view(dtype).view(shape)


def _fill_host_kv_inplace(tensor: torch.Tensor) -> None:
    """Fill host KV without a second full-size host malloc (timing UT only)."""
    tensor.zero_()


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
    try:
        offload.initialize(config)
    except Exception as exc:
        raise RuntimeError(
            f"offload.initialize failed for {dram_size_gb} GiB pool: "
            f"{type(exc).__name__}: {exc}. Host may lack free/locked memory for "
            "MemFabric pinned pool; try smaller --dram-size-gb or free host RAM."
        ) from exc


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
    # One host KV copy + ~1GiB MemFabric headroom; round up to 0.5 GiB.
    return math.ceil((need_bytes / _GIB + 1.0) * 2) / 2.0


def _alloc_cpu_kv_tensor(
    backend: CpuKvBackend,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Allocate host-side KV used by fused op.

    - offload: MemFabric ``offload.empty`` + 2MiB align (same as KVOffloadDecodeManager)
    - swap_memory: ``torch_npu.empty_with_swapped_memory``

    WARNING: on some torch_npu builds, swapped tensors SIGSEGV if you touch
    ``.shape`` / ``.device`` / ``.data_ptr()`` / ``zero_()`` / ``__repr__``.
    Only pass them into the fused op; do not print or introspect them.
    """
    if backend == "offload":
        return _empty_aligned_cpu_tensor(shape, dtype=dtype)
    if backend == "swap_memory":
        if empty_swapped is None:
            raise RuntimeError(
                "cpu-kv-backend=swap_memory requires torch_npu.empty_with_swapped_memory"
            )
        # Keep args minimal; do not print/return-repr the result here.
        print(
            "SWAP_EMPTY request_shape={} dtype={} (no tensor introspection)".format(
                shape, dtype
            ),
            flush=True,
        )
        return empty_swapped(tuple(shape), dtype=dtype, device="npu")
    raise ValueError(f"unknown cpu-kv-backend: {backend}")


def _alloc_shared_host_kv(
    topk: int,
    cpu_kv_backend: CpuKvBackend,
    max_seq_len: int | None,
) -> dict:
    """Allocate one shared full host KV for a topk/msl; reused across q_heads."""
    msl = max_seq_len if max_seq_len is not None else max(topk * 4, 256)
    assert msl >= topk, f"max_seq_len ({msl}) must be >= topk ({topk})"
    fmbn = cdiv(msl, _BLOCK_SIZE)
    dt = torch.bfloat16
    kv_shape = [fmbn, _BLOCK_SIZE, _KVD]
    rope_shape = [fmbn, _BLOCK_SIZE, _KRD]
    cpu_kv_bytes = estimate_cpu_kv_bytes(msl, _BLOCK_SIZE)
    # Print planned shapes only (Python lists). Never print swapped tensor.shape.
    print(
        "CPU_KV_PLAN backend={} max_seq_len={} block_size={} num_blocks={} "
        "k_shape={} rope_shape={} est_bytes={:.3f} GiB".format(
            cpu_kv_backend,
            msl,
            _BLOCK_SIZE,
            fmbn,
            kv_shape,
            rope_shape,
            cpu_kv_bytes / _GIB,
        ),
        flush=True,
    )
    full_kv_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, kv_shape, dt)
    full_rope_fused = _alloc_cpu_kv_tensor(cpu_kv_backend, rope_shape, dt)

    if cpu_kv_backend == "offload":
        # Timing only: inplace zero; do NOT torch.randn (extra host malloc).
        _fill_host_kv_inplace(full_kv_fused)
        _fill_host_kv_inplace(full_rope_fused)
        print(
            "FULL_KV_HOST backend=offload shape={} rope_shape={} dtype={} "
            "device={} ptr_align={} bytes={:.3f} GiB".format(
                tuple(full_kv_fused.shape),
                tuple(full_rope_fused.shape),
                dt,
                full_kv_fused.device,
                full_kv_fused.data_ptr() % _CPU_CACHE_ALIGNMENT == 0,
                (
                    full_kv_fused.numel() * full_kv_fused.element_size()
                    + full_rope_fused.numel() * full_rope_fused.element_size()
                )
                / _GIB,
            ),
            flush=True,
        )
    else:
        # swap_memory: skip zero_/shape/device/data_ptr — known SIGSEGV on some builds.
        # Leave allocation as-is for timing; values are uninitialized.
        print(
            "FULL_KV_HOST backend=swap_memory requested_k_shape={} "
            "requested_rope_shape={} dtype={} "
            "(skip tensor introspection to avoid SIGSEGV)".format(
                kv_shape, rope_shape, dt
            ),
            flush=True,
        )

    return {
        "msl": msl,
        "num_blocks": fmbn,
        "full_kv_fused": full_kv_fused,
        "full_rope_fused": full_rope_fused,
    }


def _build_inputs(
    topk: int,
    q_heads: int,
    host_kv: dict,
    kv_heads: int = 1,
):
    assert q_heads % kv_heads == 0, "q_heads must be divisible by kv_heads"
    assert kv_heads == 1, "fused op currently requires kv_heads=1"
    assert topk > 0

    bsz = 1
    seq = 1
    hd, krd, kvd = _KVD, _KRD, _KVD
    sbs = fbs = _BLOCK_SIZE
    stbs = 1
    msl = host_kv["msl"]
    fmbn = host_kv["num_blocks"]
    smbn = cdiv(topk * stbs, sbs)
    dt = torch.bfloat16
    scale = 1.0 / math.sqrt(kvd)
    torch.manual_seed(910000 + topk + q_heads * 17)
    np.random.seed((910000 + topk + q_heads * 17) % (2**32 - 1))

    q = torch.randn(bsz * seq, q_heads, hd, dtype=dt, device="npu")
    q_rope = torch.randn(bsz * seq, q_heads, krd, dtype=dt, device="npu")
    q_fused = torch.cat([q, q_rope], dim=-1).contiguous()

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
        "full_kv_fused": host_kv["full_kv_fused"],
        "full_rope_fused": host_kv["full_rope_fused"],
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


def _apply_hit_ratio(inp, hit_ratio: float) -> int:
    """Set selection status so the next fused call has the target positional hit ratio.

    Kernel hit condition is positional: ``status[i] == topk[i]``.
    - First ``round(hit_ratio * topk)`` slots are marked hit (status = topk)
    - Remaining slots are ``-1`` (miss, will H2D from full KV)

    Returns the number of hit slots per row.
    """
    if not 0.0 <= hit_ratio <= 1.0:
        raise ValueError(f"hit_ratio must be in [0, 1], got {hit_ratio}")
    topk = inp["topk_fused"]
    status = inp["sel_status"]
    k = topk.shape[-1]
    n_hit = int(round(hit_ratio * k))
    n_hit = max(0, min(k, n_hit))

    # status layout: [B, S, H, K+1]; last elem is side-channel, keep -1.
    status.fill_(-1)
    if n_hit > 0:
        status[..., :n_hit] = topk[..., :n_hit]
    return n_hit


def _prime_selection_cache(inp) -> None:
    """One full-miss call so sel_kv/sel_rope hold current topk (for realistic hits)."""
    _apply_hit_ratio(inp, 0.0)
    _ = _call_fused(inp)
    torch.npu.synchronize()


def _bench_loop(run_once, prepare, warmup: int, iters: int) -> float:
    """Time ``run_once``; ``prepare`` runs before each timed iter (not included)."""
    for _ in range(max(0, warmup)):
        if prepare is not None:
            prepare()
            torch.npu.synchronize()
        run_once()
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    total_ms = 0.0
    for _ in range(iters):
        if prepare is not None:
            prepare()
            torch.npu.synchronize()
        start.record()
        run_once()
        end.record()
        torch.npu.synchronize()
        total_ms += start.elapsed_time(end)
    return total_ms / iters


def _bench_fused_eager(inp, warmup: int, iters: int, hit_ratio: float) -> float:
    _prime_selection_cache(inp)
    n_hit = _apply_hit_ratio(inp, hit_ratio)
    print(
        "HIT_RATIO target={:.4f} n_hit={}/{}".format(
            hit_ratio, n_hit, inp["topk_fused"].shape[-1]
        ),
        flush=True,
    )

    def prepare():
        _apply_hit_ratio(inp, hit_ratio)

    def run_once():
        _ = _call_fused(inp)

    return _bench_loop(run_once, prepare, warmup=warmup, iters=iters)


def _bench_fused_graph(inp, warmup: int, iters: int, hit_ratio: float) -> float:
    """Capture fused op into ACL graph and time ``graph.replay()``."""
    if not hasattr(torch.npu, "NPUGraph"):
        raise RuntimeError("torch.npu.NPUGraph is unavailable on this torch_npu build")

    _prime_selection_cache(inp)
    n_hit = _apply_hit_ratio(inp, hit_ratio)
    print(
        "HIT_RATIO target={:.4f} n_hit={}/{}".format(
            hit_ratio, n_hit, inp["topk_fused"].shape[-1]
        ),
        flush=True,
    )

    # Eager warmup at target hit ratio (compile / settle).
    for _ in range(max(1, warmup)):
        _apply_hit_ratio(inp, hit_ratio)
        torch.npu.synchronize()
        _ = _call_fused(inp)
    torch.npu.synchronize()

    _apply_hit_ratio(inp, hit_ratio)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    print("GRAPH_CAPTURE begin", flush=True)
    try:
        with torch.npu.graph(
            graph,
            capture_error_mode="thread_local",
            auto_dispatch_capture=True,
        ):
            out = _call_fused(inp)
    except TypeError:
        # Older torch_npu may not accept auto_dispatch_capture.
        with torch.npu.graph(graph, capture_error_mode="thread_local"):
            out = _call_fused(inp)
    torch.npu.synchronize()
    print(
        "GRAPH_CAPTURE done out_shape={}".format(tuple(out.shape)),
        flush=True,
    )

    def prepare():
        # Replay reads same addresses; rewrite status so each replay keeps target hit ratio.
        _apply_hit_ratio(inp, hit_ratio)

    def run_once():
        graph.replay()

    return _bench_loop(run_once, prepare, warmup=warmup, iters=iters)


def run_case(
    topk: int,
    q_heads: int,
    host_kv: dict,
    warmup: int,
    iters: int,
    cpu_kv_backend: CpuKvBackend = "offload",
    use_graph: bool = False,
    hit_ratio: float = 1.0,
) -> float:
    inp = _build_inputs(topk, q_heads, host_kv=host_kv)
    mode = "graph" if use_graph else "eager"
    print(
        "CASE topk={} q_heads={} kv_heads={} max_seq_len={} num_blocks={} "
        "cpu_kv_backend={} mode={} hit_ratio={:.4f} q_shape={} topk_shape={}".format(
            topk,
            q_heads,
            inp["kv_heads"],
            inp["msl"],
            inp["num_blocks"],
            cpu_kv_backend,
            mode,
            hit_ratio,
            tuple(inp["q_fused"].shape),
            tuple(inp["topk_fused"].shape),
        ),
        flush=True,
    )

    if use_graph:
        avg_ms = _bench_fused_graph(inp, warmup=warmup, iters=iters, hit_ratio=hit_ratio)
    else:
        avg_ms = _bench_fused_eager(inp, warmup=warmup, iters=iters, hit_ratio=hit_ratio)
    print(
        "BENCH mode={} hit_ratio={:.4f} topk={} q_heads={} max_seq_len={} "
        "warmup={} iters={} avg_ms={:.3f}".format(
            mode, hit_ratio, topk, q_heads, inp["msl"], warmup, iters, avg_ms
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
    parser.add_argument(
        "--use-graph",
        action="store_true",
        help="Capture fused op into torch.npu.NPUGraph and time graph.replay() "
        "(ACL graph / aclgraph path). Default is eager launch timing.",
    )
    parser.add_argument(
        "--hit-ratio",
        type=float,
        default=1.0,
        help="Target selection/topk-buffer positional reuse ratio in [0, 1]. "
        "Each timed iter resets status so the first round(hit_ratio*topk) slots "
        "are hits and the rest are misses. Default 1.0 (all hit).",
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
    hit_ratio = args.hit_ratio
    if not 0.0 <= hit_ratio <= 1.0:
        raise ValueError(f"--hit-ratio must be in [0, 1], got {hit_ratio}")
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

    mode = "graph" if args.use_graph else "eager"
    print(
        f"{cases=} {q_heads_list=} max_seq_len={plan_msl} "
        f"num_blocks={cdiv(plan_msl, _BLOCK_SIZE)} "
        f"warmup={args.warmup} iters={args.iters} mode={mode} "
        f"hit_ratio={hit_ratio} "
        f"cpu_kv_backend={cpu_kv_backend} dram_size_gb={dram_size_gb} "
        f"cpu_kv_est_gib={need_bytes / _GIB:.3f}",
        flush=True,
    )

    if cpu_kv_backend == "offload":
        init_offload_pool(dram_size_gb=dram_size_gb)

    results = []
    try:
        for topk in cases:
            # One host KV per topk/msl; reuse across q_heads to avoid pool OOM.
            host_kv = _alloc_shared_host_kv(topk, cpu_kv_backend, max_seq_len)
            try:
                for q_heads in q_heads_list:
                    avg_ms = run_case(
                        topk,
                        q_heads,
                        host_kv=host_kv,
                        warmup=args.warmup,
                        iters=args.iters,
                        cpu_kv_backend=cpu_kv_backend,
                        use_graph=args.use_graph,
                        hit_ratio=hit_ratio,
                    )
                    results.append((topk, q_heads, mode, hit_ratio, avg_ms))
            finally:
                del host_kv
                gc.collect()
    finally:
        if cpu_kv_backend == "offload":
            try:
                uninit_offload_pool()
            except Exception as exc:
                print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    print("----- TIMING SUMMARY -----", flush=True)
    for topk, q_heads, bench_mode, ratio, avg_ms in results:
        print(
            "mode={} hit_ratio={:.4f} topk={} q_heads={} avg_ms={:.3f}".format(
                bench_mode, ratio, topk, q_heads, avg_ms
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
