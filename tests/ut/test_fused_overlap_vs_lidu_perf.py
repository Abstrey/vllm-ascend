"""Perf comparison: vllm-ascend fused_overlap vs nano LIDU+scatter_sfa.

Two OPP packages cannot coexist in one process, so run one approach per
process (writes JSON), then --compare reads both.

A: LI + host-side LRU planner + fused_overlap(copy+sfa in one kernel,
   consuming the host-encoded external plan).
B: LIDU(li+lru fused) + scatter_copy_sfa(copy+sfa fused).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
from typing import Callable

import numpy as np
import torch
import torch_npu  # noqa: F401

BLOCK_SIZE = 128
KVD = 512
KRD = 64
INDEX_DIM = 128
TOPK = 2048
_GIB = 1024 * 1024 * 1024
_ALIGN = 2 * 1024 * 1024


def _default_nano_path() -> str:
    """Resolve the nano torch_extension directory.

    Priority: $NANO_PATH env > sibling-relative to this UT > hardcoded fallback.
    """
    env = os.environ.get("NANO_PATH")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.normpath(
        os.path.join(here, "..", "..", "..",
                     "nanovllm-DSA-offload", "torch_extension"))
    if os.path.isdir(os.path.join(sibling, "ops_overlap")):
        return sibling
    return "/Users/zywszr/Desktop/codes/kv_offload/nanovllm-DSA-offload/torch_extension"


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def bench_events(runner, warmup, iters, reset=None):
    for _ in range(max(0, warmup)):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        runner()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        if reset is not None:
            reset()
            torch.npu.synchronize()
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        s.record()
        runner()
        e.record()
        e.synchronize()
        times.append(float(s.elapsed_time(e)))
    return statistics.mean(times)


def profile_pipeline(steps, warmup, iters, profile_dir=None):
    """Run the full pipeline under torch_npu.profiler (msprof backend).

    ``steps`` is a list of ``(name, callable)`` executed in order each iter.
    Prints an operator-level table sorted by NPU self time so kernel-level
    cost (AICore) and host dispatch cost are visible separately.
    """
    import torch_npu.profiler as prof

    for _ in range(max(0, warmup)):
        for _, fn in steps:
            fn()
        torch.npu.synchronize()

    exp = prof._ExperimentalConfig(
        export_type=prof.ExportType.Text,
        profiler_level=prof.ProfilerLevel.Level1,
        aic_metrics=prof.AiCMetrics.PipeUtilization,
        data_simplification=True,
    )
    activities = [prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU]
    trace_cb = prof.tensorboard_trace_handler(profile_dir) if profile_dir else None
    with prof.profile(activities=activities, experimental_config=exp,
                      on_trace_ready=trace_cb) as p:
        for _ in range(iters):
            for _, fn in steps:
                fn()
            torch.npu.synchronize()

    print("----- PROFILER OP-LEVEL (sorted by self NPU time) -----", flush=True)
    try:
        print(p.key_averages().table(sort_by="self_npu_time_total", row_limit=30),
              flush=True)
    except Exception:
        print(p.key_averages().table(row_limit=30), flush=True)
    if profile_dir:
        print(f"PROFILE_TRACE_DIR={profile_dir}", flush=True)


# ===================== Approach A: vllm-ascend =====================
def run_a(args) -> dict:
    import vllm_ascend.vllm_ascend_C  # noqa: F401
    from memfabric_hybrid import offload
    from vllm_ascend.utils import enable_custom_op
    from vllm_ascend.distributed.kv_transfer.kv_offload_decode import (
        kv_offload_decode_manager,
    )

    assert enable_custom_op()
    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
    FUSED_OVERLAP = torch.ops._C_ascend.npu_fused_sparse_attention_overlap

    def _align(t, a):
        p = t.data_ptr()
        ap = (p + a - 1) // a * a
        return t[(ap - p) // t.element_size():]

    def _empty_cpu(shape, dtype):
        n = int(np.prod(shape))
        nb = n * torch.empty((), dtype=dtype).element_size()
        raw = offload.empty([nb + _ALIGN], dtype=torch.int8, pin_memory=True)
        return _align(raw, _ALIGN)[:nb].view(dtype).view(shape)

    cfg = offload.OffloadConfig()
    cfg.device_id = torch_npu.npu.current_device()
    cfg.size = int(args.dram_size_gb * _GIB)
    cfg.world_size = 1
    cfg.rank_id = 0
    offload.initialize(cfg)

    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    fmbn = cdiv(seq_len, BLOCK_SIZE)
    smbn = cdiv(TOPK, BLOCK_SIZE)
    scale = 1.0 / math.sqrt(KVD)
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len)

    q = torch.randn(batch, q_heads, KVD, dtype=dt, device="npu")
    qr = torch.randn(batch, q_heads, KRD, dtype=dt, device="npu")
    q_fused = torch.cat([q, qr], dim=-1).contiguous()
    full_kv = _empty_cpu([fmbn, BLOCK_SIZE, KVD], dt)
    full_kv.zero_()
    full_rope = _empty_cpu([fmbn, BLOCK_SIZE, KRD], dt)
    full_rope.zero_()
    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(batch, -1).contiguous()
    aq = torch.tensor([1] * batch, dtype=torch.int32, device="npu")
    ak = torch.tensor([seq_len] * batch, dtype=torch.int32, device="npu")

    topk_np = np.zeros((batch, 1, TOPK), dtype=np.int32)
    for r in range(batch):
        topk_np[r, 0] = np.sort(np.random.choice(seq_len, TOPK, replace=False))
    topk = torch.tensor(topk_np, dtype=torch.int32, device="npu").reshape(batch, 1, 1, TOPK).contiguous()

    total_sel = smbn * batch
    sel_kv = torch.zeros(total_sel, BLOCK_SIZE, KVD, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, BLOCK_SIZE, KRD, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(batch, smbn)
    sel_status = torch.full((batch, 1, 1, TOPK + 1), -1, dtype=torch.int32, device="npu")

    li_q = torch.zeros((batch, idx_heads, INDEX_DIM), dtype=dt, device="npu")
    li_q[:, 0, 0] = 1
    li_q[:, 0, 1] = 64
    li_q[:, 0, 2] = 4096
    li_w = torch.zeros((batch, idx_heads), dtype=dt, device="npu")
    li_w[:, 0] = 1
    bpr = seq_len // BLOCK_SIZE
    li_key = torch.zeros((batch * bpr, BLOCK_SIZE, 1, INDEX_DIM), dtype=dt, device="npu")
    lids = torch.arange(seq_len, dtype=torch.int32, device="npu").view(1, bpr, BLOCK_SIZE)
    kr = li_key.view(batch, bpr, BLOCK_SIZE, 1, INDEX_DIM)
    kr[:, :, :, 0, 0] = (lids % 64).to(dt)
    kr[:, :, :, 0, 1] = ((lids // 64) % 64).to(dt)
    kr[:, :, :, 0, 2] = (lids // 4096).to(dt)
    li_bt = torch.arange(batch * bpr, dtype=torch.int32, device="npu").view(batch, bpr)
    li_ql = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
    li_cl = torch.full((batch,), seq_len, dtype=torch.int32, device="npu")

    # ---- host-side LRU planner (production path) -----------------------
    # The production A path runs the LRU/replacement planner on host
    # (kv_offload_decode.cpp::lru_resident_compact_with_plan_stable_rows)
    # BEFORE the fused kernel. The kernel then consumes the encoded plan
    # written into selection_membership_map (EXTERNAL_PLAN_READY_MARKER).
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include = os.path.join(ascend_home, "include")
    npu_lib = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib):
        npu_lib = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib = os.path.join(torch_npu_path, "lib")
    os.environ["TORCH_EXTENSIONS_ALWAYS_BUILD"] = "1"
    os.environ["CXX"] = "clang++"
    os.environ["CC"] = "clang"
    cpp_src = os.path.join(
        os.path.dirname(os.path.abspath(kv_offload_decode_manager.__file__)),
        "kv_offload_decode.cpp",
    )
    lru_cpp = torch.utils.cpp_extension.load(
        name="kv_offload_decode",
        sources=[cpp_src],
        extra_cflags=[
            "-O3", "-std=c++20", "-fopenmp",
            "-march=armv8.2-a+sve+fp16+bf16", "-fPIC",
            f"-I{npu_include}", f"-I{torch_npu_include}",
        ],
        extra_ldflags=[
            "-fopenmp", f"-L{npu_lib}", "-lascendcl",
            f"-L{torch_npu_lib}", "-ltorch_npu",
        ],
        verbose=False,
    )

    EXTERNAL_PLAN_READY_MARKER = 0x5A45
    PAIRED_SELECTION_COPY_MARKER = 0x5A56
    MEMBERSHIP_MAP_INT16 = 16376
    MEMBERSHIP_ALIGN_INT16 = 16
    MEMBERSHIP_CONTROL_INT16 = 8
    CONTROL_OFFSET_INT16 = (
        MEMBERSHIP_MAP_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16384
    STORAGE_INT16 = (
        CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16 + MEMBERSHIP_ALIGN_INT16
    ) // MEMBERSHIP_ALIGN_INT16 * MEMBERSHIP_ALIGN_INT16  # 16392

    topk_buffer_size = TOPK * 2
    max_rows = batch
    threads = 8
    plan_start = CONTROL_OFFSET_INT16 - TOPK
    required_columns = CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16

    membership_map = offload.empty(
        [max_rows * STORAGE_INT16], dtype=torch.int16, pin_memory=True,
    ).view([max_rows, STORAGE_INT16])
    membership_map.fill_(-1)
    control = membership_map[
        :, CONTROL_OFFSET_INT16:CONTROL_OFFSET_INT16 + MEMBERSHIP_CONTROL_INT16]
    control[:, 1] = EXTERNAL_PLAN_READY_MARKER
    control[:, 2] = TOPK
    control[:, 3] = CONTROL_OFFSET_INT16 - TOPK
    control[:, 7] = PAIRED_SELECTION_COPY_MARKER
    plan_storage = membership_map[:max_rows, plan_start:required_columns]
    encoded_plan_stride = membership_map.stride(0)

    lru_req_ids = torch.empty([max_rows], dtype=torch.int64, device="cpu", pin_memory=True)
    lru_last_req_ids = torch.full([max_rows], -1, dtype=torch.int64, device="cpu", pin_memory=True)
    lru_topk_indices = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_stable_prefix_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_visible_seq_lens = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_to_token = torch.full([max_rows, topk_buffer_size], -1, dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slots = torch.arange(topk_buffer_size, dtype=torch.int32, device="cpu").view(1, -1).repeat(max_rows, 1).pin_memory()
    lru_current_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_count = torch.empty([max_rows], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_tokens = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_slots = torch.empty([max_rows, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_mark = torch.zeros([threads, seq_len], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_token_pos = torch.full([threads, seq_len], -1, dtype=torch.int32, device="cpu", pin_memory=True)
    lru_slot_ws = torch.empty([threads, topk_buffer_size * 3], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_miss_pos_ws = torch.empty([threads, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_epochs = torch.zeros([threads], dtype=torch.int32, device="cpu", pin_memory=True)
    lru_physical_row_ws = torch.empty([max_rows * 3], dtype=torch.int32, device="cpu", pin_memory=True)

    lru_cpp.warmup_lru_resident_threads(threads)

    topk_cpu = torch.empty([batch, TOPK], dtype=torch.int32, device="cpu", pin_memory=True)
    req_ids_cpu = torch.arange(batch, dtype=torch.int64, device="cpu")

    def _a_planner():
        # refresh host-side inputs from the NPU topk indices
        topk_cpu.copy_(topk.reshape(batch, TOPK).to(torch.int32))
        lru_req_ids.copy_(req_ids_cpu)
        lru_topk_indices.copy_(topk_cpu)
        lru_stable_prefix_lens.fill_(0)
        lru_visible_seq_lens.fill_(seq_len)
        lru_cpp.lru_resident_compact_with_plan_stable_rows(
            lru_req_ids.data_ptr(),
            lru_last_req_ids.data_ptr(),
            lru_topk_indices.data_ptr(),
            lru_stable_prefix_lens.data_ptr(),
            lru_slot_to_token.data_ptr(),
            lru_slots.data_ptr(),
            lru_current_slots.data_ptr(),
            lru_miss_count.data_ptr(),
            lru_miss_tokens.data_ptr(),
            lru_miss_slots.data_ptr(),
            lru_token_mark.data_ptr(),
            lru_token_pos.data_ptr(),
            lru_slot_ws.data_ptr(),
            lru_miss_pos_ws.data_ptr(),
            lru_epochs.data_ptr(),
            lru_physical_row_ws.data_ptr(),
            max_rows,
            plan_storage.data_ptr(),
            encoded_plan_stride,
            batch,
            TOPK,
            topk_buffer_size,
            seq_len,
            threads,
            threads,
            lru_visible_seq_lens.data_ptr(),
        )

    def _a_li():
        o = torch_npu.npu_lightning_indexer(
            query=li_q, key=li_key, weights=li_w,
            actual_seq_lengths_query=li_ql, actual_seq_lengths_key=li_cl,
            block_table=li_bt, layout_query="TND", layout_key="PA_BSND",
            sparse_count=TOPK, sparse_mode=3)
        return o[0] if isinstance(o, (tuple, list)) else o

    def _a_fused():
        return FUSED_OVERLAP(
            query=q_fused,
            selection_k_rope=sel_rope,
            selection_kv_cache=sel_kv,
            selection_kv_block_table=sel_bt,
            selection_kv_block_status=sel_status,
            selection_membership_map=membership_map,
            selection_topk_indices=topk,
            full_k_rope=full_rope,
            full_kv_cache=full_kv,
            full_kv_block_table=full_bt,
            full_kv_actual_seq=ak,
            full_q_actual_seq=aq,
            scale_value=scale,
            sparse_block_size=1,
            selection_topk_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3)

    # warmup: planner then fused (production order)
    _a_planner()
    _a_fused()
    torch.npu.synchronize()
    if getattr(args, "profile", False):
        profile_pipeline(
            [("li", _a_li), ("planner", _a_planner), ("fused_overlap", _a_fused)],
            args.warmup, args.profile_iters, args.profile_dir)
        try:
            offload.uninitialize()
        except Exception as exc:
            print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)
        return {
            "approach": "A", "profile": True,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "hit_ratio": args.hit_ratio,
            "warmup": args.warmup, "iters": args.profile_iters,
        }
    li_ms = bench_events(_a_li, args.warmup, args.iters)
    planner_ms = bench_events(_a_planner, args.warmup, args.iters)
    fused_ms = bench_events(_a_fused, args.warmup, args.iters,
                           reset=_a_planner)
    try:
        offload.uninitialize()
    except Exception as exc:
        print(f"OFFLOAD_UNINIT_WARN {type(exc).__name__}: {exc}", flush=True)

    return {
        "approach": "A",
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "hit_ratio": args.hit_ratio,
        "li_ms": li_ms, "planner_ms": planner_ms, "fused_overlap_ms": fused_ms,
        "total_ms": li_ms + planner_ms + fused_ms,
        "warmup": args.warmup, "iters": args.iters,
    }


# ===================== Approach B: nano =====================
def run_b(args) -> dict:
    if args.nano_path not in sys.path:
        sys.path.insert(0, args.nano_path)

    np_dir = args.nano_path
    pkg_dir = os.path.join(np_dir, "ops_overlap")
    opapi = os.path.join(pkg_dir, "..", "..", "_custom_opp", "vendors",
                         "ops-overlap", "op_api", "lib", "libcust_opapi.so")
    so_files = [f for f in os.listdir(pkg_dir)
                if f.startswith("_C") and f.endswith(".so")] if os.path.isdir(pkg_dir) else []
    print(f"[B-diag] nano_path={np_dir!r}", flush=True)
    print(f"[B-diag] ops_overlap/ exists: {os.path.isdir(pkg_dir)}", flush=True)
    print(f"[B-diag] ops_overlap/__init__.py exists: "
          f"{os.path.isfile(os.path.join(pkg_dir, '__init__.py'))}", flush=True)
    print(f"[B-diag] _C*.so files: {so_files}", flush=True)
    print(f"[B-diag] libcust_opapi.so exists: "
          f"{os.path.isfile(opapi)} ({opapi})", flush=True)
    print(f"[B-diag] sys.path[0:3]={sys.path[:3]}", flush=True)

    try:
        import ops_overlap  # noqa: F401  sets ASCEND_CUSTOM_OPP_PATH
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise ImportError(
            f"Cannot import ops_overlap from --nano-path={np_dir!r}. "
            f"Ensure nano is built (bash build.sh in nanovllm-DSA-offload) and "
            f"the path contains ops_overlap/. Original error: {exc}"
        ) from exc

    torch_npu.npu.set_device(0)
    torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})

    def _swap(cpu, dev):
        t = torch_npu.empty_with_swapped_memory(cpu.shape, dtype=cpu.dtype, device=dev)
        t.fill_(0)
        t.add_(cpu.to(dev))
        return t

    dev = torch.device("npu:0")
    dt = torch.bfloat16
    batch = args.batch_size
    q_heads = args.q_heads
    idx_heads = args.indexer_heads
    seq_len = args.seq_len
    cache_tokens = args.cache_tokens
    miss = args.miss
    torch.manual_seed(910000 + batch + q_heads * 17 + seq_len + 1)
    cb = math.ceil(cache_tokens / BLOCK_SIZE)
    sb = seq_len // BLOCK_SIZE

    hbm_bt = torch.empty((batch, cb), dtype=torch.int32)
    for r in range(batch):
        ph = torch.arange(r * cb, (r + 1) * cb, dtype=torch.int32)
        hbm_bt[r] = ph[torch.randperm(cb)]
    dram_bt = torch.stack([torch.randperm(sb, dtype=torch.int64).to(torch.int32) for _ in range(batch)])

    slots = torch.empty((batch, TOPK), dtype=torch.int32)
    for r in range(batch):
        slots[r] = torch.randperm(cache_tokens, dtype=torch.int64)[:TOPK].to(torch.int32)
    counts = torch.full((batch,), miss, dtype=torch.int32)
    src_ids = torch.full((batch, TOPK), -1, dtype=torch.int32)
    for r in range(batch):
        c = int(counts[r])
        if c:
            src_ids[r, :c] = torch.randperm(seq_len, dtype=torch.int64)[:c].to(torch.int32)

    thb = batch * cb
    hbm_kpe = torch.randn((thb, BLOCK_SIZE, 1, KRD), dtype=dt, device=dev)
    hbm_ckv = torch.randn((thb, BLOCK_SIZE, 1, KVD), dtype=dt, device=dev)
    query = torch.randn((batch, q_heads, KVD), dtype=dt, device=dev)
    qrope = torch.randn((batch, q_heads, KRD), dtype=dt, device=dev)

    tib = batch * sb
    idx_cache = torch.randn((tib, BLOCK_SIZE, INDEX_DIM), dtype=dt, device=dev)
    idx_bt = torch.arange(tib, dtype=torch.int32, device=dev).view(batch, sb)

    g = torch.Generator().manual_seed(42)
    dk = torch.randn((sb, BLOCK_SIZE, KRD), generator=g, dtype=torch.float32).to(dt)
    dc = torch.randn((sb, BLOCK_SIZE, KVD), generator=g, dtype=torch.float32).to(dt)
    dram_kpe = _swap(dk, dev)
    dram_ckv = _swap(dc, dev)
    torch.npu.synchronize()

    pool = batch + 7
    pg = torch.Generator().manual_seed(99)
    req_e = torch.randperm(pool, generator=pg)[:batch].to(torch.int32)
    cs_cpu = torch.full((pool, seq_len), -1, dtype=torch.int32)
    for r in range(batch):
        pr = int(req_e[r])
        tt = torch.arange(seq_len - TOPK, seq_len, dtype=torch.int64)
        sl = torch.randperm(cache_tokens, generator=torch.Generator().manual_seed(r), dtype=torch.int32)
        cs_cpu[pr, tt] = sl[:TOPK]
    cache_slots = cs_cpu.to(dev)
    cache_slots_init = cache_slots.clone()

    weights = torch.zeros((batch, idx_heads), dtype=dt, device=dev)
    cache_tokens_t = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    candidate_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=dev)
    src_ids_t = src_ids.to(dev)
    dst_slots = torch.empty_like(src_ids_t)
    miss_counts = torch.empty((batch,), dtype=torch.int32, device=dev)
    sparse_slots = slots[:, None, :].to(dev)
    actual_q = torch.arange(1, batch + 1, dtype=torch.int32, device=dev)
    actual_kv = torch.full((batch,), cache_tokens, dtype=torch.int32, device=dev)
    scale = 1.0 / math.sqrt(KVD + KRD)

    def _b_reset():
        cache_slots.copy_(cache_slots_init)

    def _b_lidu():
        return torch.ops.ops_overlap.lidu_decode_update_out.default(
            query, idx_cache, weights, req_e.to(dev), cache_slots,
            cache_tokens_t, candidate_lens, idx_bt, src_ids_t, dst_slots, miss_counts)

    def _b_scatter_sfa():
        return torch.ops.ops_overlap.sparse_and_tail_attention_and_scatter_copy.default(
            query, hbm_ckv, sparse_slots, cache_tokens_t, hbm_bt.to(dev),
            actual_q, actual_kv, qrope, hbm_kpe, dram_kpe, dram_ckv,
            dram_bt.to(dev), src_ids_t, miss_counts, scale)

    _b_lidu()
    torch.npu.synchronize()
    _b_scatter_sfa()
    torch.npu.synchronize()
    if getattr(args, "profile", False):
        profile_pipeline(
            [("lidu", _b_lidu), ("scatter_sfa", _b_scatter_sfa)],
            args.warmup, args.profile_iters, args.profile_dir)
        return {
            "approach": "B", "profile": True,
            "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
            "seq_len": seq_len, "cache_tokens": cache_tokens, "miss": miss,
            "warmup": args.warmup, "iters": args.profile_iters,
        }
    lidu_ms = bench_events(_b_lidu, args.warmup, args.iters, reset=_b_reset)

    def reset_b():
        _b_reset()
        _b_lidu()
        torch.npu.synchronize()

    sfa_ms = bench_events(_b_scatter_sfa, args.warmup, args.iters, reset=reset_b)

    return {
        "approach": "B",
        "batch_size": batch, "q_heads": q_heads, "indexer_heads": idx_heads,
        "seq_len": seq_len, "cache_tokens": cache_tokens, "miss": miss,
        "lidu_ms": lidu_ms, "scatter_sfa_ms": sfa_ms,
        "total_ms": lidu_ms + sfa_ms, "warmup": args.warmup, "iters": args.iters,
    }


# ===================== Compare =====================
def compare(args) -> None:
    with open(args.out_a) as f:
        ra = json.load(f)
    with open(args.out_b) as f:
        rb = json.load(f)
    print("----- TIMING SUMMARY (ms, mean over iters) -----", flush=True)
    print(f"Approach A (vllm-ascend):  LI={ra['li_ms']:.4f}  "
          f"planner={ra.get('planner_ms', 0.0):.4f}  "
          f"fused_overlap={ra['fused_overlap_ms']:.4f}  total={ra['total_ms']:.4f}",
          flush=True)
    print(f"Approach B (nano):        LIDU={rb['lidu_ms']:.4f}  "
          f"scatter_sfa={rb['scatter_sfa_ms']:.4f}  total={rb['total_ms']:.4f}",
          flush=True)
    diff = ra["total_ms"] - rb["total_ms"]
    who = "A faster" if diff < 0 else "B faster"
    print(f"Delta (A-B): {diff:+.4f} ms  ({who} by {abs(diff):.4f} ms)", flush=True)
    print("UT_OK", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--approach", choices=("A", "B"), default=None,
                   help="Run one approach in this process (writes --out).")
    p.add_argument("--compare", action="store_true",
                   help="Compare --out-a and --out-b JSON results.")
    p.add_argument("--out", default=None, help="Output JSON path for --approach.")
    p.add_argument("--out-a", default="result_a.json")
    p.add_argument("--out-b", default="result_b.json")
    p.add_argument("--nano-path", default=_default_nano_path(),
                   help="Path to nano torch_extension dir (contains ops_overlap/). "
                        "Override via $NANO_PATH env or --nano-path.")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--q-heads", type=int, default=16)
    p.add_argument("--indexer-heads", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=65536)
    p.add_argument("--cache-tokens", type=int, default=8192)
    p.add_argument("--miss", type=int, default=300, help="Miss count per row (approach B).")
    p.add_argument("--hit-ratio", type=float, default=0.85, help="Approach A hit ratio.")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dram-size-gb", type=float, default=2.0)
    p.add_argument("--profile", action="store_true",
                   help="Profile the full pipeline with torch_npu.profiler (msprof) "
                        "and print a kernel-level op table instead of Event timing.")
    p.add_argument("--profile-iters", type=int, default=5,
                   help="Iterations to capture under --profile.")
    p.add_argument("--profile-dir", default=None,
                   help="Directory to export the full msprof trace to (optional).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.compare:
        compare(args)
        return
    if args.approach is None:
        raise SystemExit("Use --approach A|B or --compare. See --help.")
    if args.approach == "A":
        result = run_a(args)
    else:
        result = run_b(args)
    out = args.out or f"result_{args.approach.lower()}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"RESULT approach={args.approach} out={out} total_ms={result['total_ms']:.4f}",
          flush=True)
    print("UT_OK", flush=True)


if __name__ == "__main__":
    main()
