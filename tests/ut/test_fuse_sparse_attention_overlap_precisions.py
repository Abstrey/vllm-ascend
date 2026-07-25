
import os, math, numpy as np, torch, torch_npu
import vllm_ascend.vllm_ascend_C
# import vllm_ascend
from vllm_ascend.utils import enable_custom_op
# from zbal import zbal_init, zbal_uninit, zbal_h2d_init, empty_tensor
from memfabric_hybrid import offload

assert enable_custom_op()
torch_npu.npu.set_device(0)
torch.npu.set_option({"ACL_PRECISION_MODE": "must_keep_origin_dtype"})
fused = torch.ops._C_ascend.npu_fused_sparse_attention_overlap
sfa = torch.ops._C_ascend.npu_sparse_flash_attention
empty_swapped = getattr(torch_npu, "empty_with_swapped_memory", None)
cpu_tensor = False

# zbal_h2d_init(1024 * 1024 * 1024, 1 * 1 * 2048 * 2)
# tensors = empty_tensor([64 * 128 * 576], dtype=torch.bfloat16, pin_memory=True)

# offload.initialize(0, 1024 * 1024 * 1024)

_CPU_CACHE_ALIGNMENT = 1

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def _align_memory(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
    data_ptr = tensor.data_ptr()
    aligned_addr = (data_ptr + alignment - 1) // alignment * alignment
    offset = (aligned_addr - data_ptr) // tensor.element_size()
    return tensor[int(offset):]

def _empty_aligned_cpu_tensor(
    shape: list[int],
    dtype: torch.dtype,
    alignment: int = _CPU_CACHE_ALIGNMENT,
) -> torch.Tensor:
    num_elements = int(np.prod(shape))
    extra_elements = cdiv(alignment, torch.empty((), dtype=dtype).element_size())
    tensor = offload.empty([num_elements + extra_elements], dtype=dtype, pin_memory=True)
    return _align_memory(tensor, alignment)[:num_elements].view(shape)


def run_case(topk):
    bsz = 1
    seq = 1
    q_heads = 1
    kv_heads = 1
    hd, krd, kvd = 512, 64, 512
    sbs = fbs = 128
    stbs = 1
    msl = max(topk * 4, 256)
    fmbn = (msl + fbs - 1) // fbs
    smbn = (topk * stbs + sbs - 1) // sbs
    dt = torch.bfloat16
    scale = 1.0 / math.sqrt(kvd)
    torch.manual_seed(910000 + topk)
    np.random.seed((910000 + topk) % (2**32 - 1))
    q = torch.randn(bsz * seq, q_heads, hd, dtype=dt, device="npu")
    q_rope = torch.randn(bsz * seq, q_heads, krd, dtype=dt, device="npu")
    q_fused = torch.cat([q, q_rope], dim=-1).contiguous()
    full_kv_npu = torch.randn(fmbn * bsz, fbs, kvd, dtype=dt, device="npu")
    full_rope_npu = torch.randn(fmbn * bsz, fbs, krd, dtype=dt, device="npu")
    print(full_kv_npu.shape, full_rope_npu.shape, dt)
    if empty_swapped is not None:
        if cpu_tensor:
            # full_kv_fused = empty_tensor([full_kv_npu.shape[0] * full_kv_npu.shape[1] * full_kv_npu.shape[2]], dtype=dt, pin_memory=True).view(full_kv_npu.shape)
            # full_rope_fused = empty_tensor([full_rope_npu.shape[0] * full_rope_npu.shape[1] * full_rope_npu.shape[2]], dtype=dt, pin_memory=True).view(full_rope_npu.shape)
            # full_rope_fused = empty_tensor(tuple(full_rope_npu.shape), dtype=dt, pin_memory=True)
            full_kv_fused = _empty_aligned_cpu_tensor(list(full_kv_npu.shape), dtype=dt)
            full_rope_fused = _empty_aligned_cpu_tensor(list(full_rope_npu.shape), dtype=dt)
        else:
            print("what")
            full_kv_fused = empty_swapped(tuple(full_kv_npu.shape), dtype=dt, device="npu")
            full_rope_fused = empty_swapped(tuple(full_rope_npu.shape), dtype=dt, device="npu")
            print("what")
        full_kv_fused.copy_(full_kv_npu)
        full_rope_fused.copy_(full_rope_npu)
        torch.npu.synchronize()
    else:
        full_kv_fused = full_kv_npu
        full_rope_fused = full_rope_npu
    #print(f"{full_kv_fused.shape=} {full_rope_fused.shape=}\n{full_kv_fused[-1]}\n{full_kv_npu[-1]}\n{full_kv_fused.device} {full_rope_fused.device}")
    full_bt = torch.arange(fmbn, dtype=torch.int32, device="npu").unsqueeze(0).expand(bsz, -1).contiguous()
    actual_q = torch.tensor([seq] * bsz, dtype=torch.int32, device="npu")
    actual_k = torch.tensor([msl] * bsz, dtype=torch.int32, device="npu")
    all_ids = np.arange(msl, dtype=np.int32)
    topk_np = np.zeros((bsz * seq, kv_heads, topk), dtype=np.int32)
    for row in range(bsz * seq):
        topk_np[row, 0] = np.sort(np.random.choice(all_ids, topk, replace=False))
    topk_sfa = torch.tensor(topk_np, dtype=torch.int32, device="npu")
    print(f"{topk_sfa=}")
    sfa_out = sfa(
        q, full_kv_npu.unsqueeze(2), full_kv_npu.unsqueeze(2), topk_sfa, scale,
        sparse_block_size=1, block_table=full_bt, actual_seq_lengths_query=actual_q,
        actual_seq_lengths_kv=actual_k, query_rope=q_rope, key_rope=full_rope_npu.unsqueeze(2),
        layout_query="TND", layout_kv="PA_BSND", sparse_mode=3, attention_mode=2, return_softmax_lse=False)
    if isinstance(sfa_out, (tuple, list)):
        sfa_out = sfa_out[0]
    torch.npu.synchronize()
    total_sel = smbn * bsz * seq * kv_heads
    sel_kv = torch.zeros(total_sel, sbs, kvd, dtype=dt, device="npu")
    sel_rope = torch.zeros(total_sel, sbs, krd, dtype=dt, device="npu")
    sel_bt = torch.arange(total_sel, dtype=torch.int32, device="npu").reshape(bsz * seq * kv_heads, smbn)
    sel_status = torch.full((bsz, seq, kv_heads, topk + 1), -1, dtype=torch.int32, device="npu")
    topk_fused = topk_sfa.reshape(bsz, seq, kv_heads, topk).contiguous()
    print("RUN_EQUAL_HEAD_FORCE_DIRECT topk={} q_heads=1 kv_heads=1".format(topk), flush=True)
    fused_out = fused(
        q_fused, sel_rope, sel_kv, sel_bt, sel_status, topk_fused,
        full_rope_fused, full_kv_fused, full_bt, actual_k, actual_q,
        scale, 1, stbs, layout_query="TND", layout_kv="PA_BSND", sparse_mode=3)
    torch.npu.synchronize()
    ref = sfa_out[..., :kvd].float().cpu()
    out = fused_out[..., :kvd].float().cpu()
    diff = (ref - out).abs()
    close = torch.isclose(out, ref, rtol=5e-2, atol=5e-2).float().mean().item() * 100
    print("DIRECT_RESULT topk={} close={:.2f}% max={:.4e} mean={:.4e}".format(topk, close, diff.max().item(), diff.mean().item()), flush=True)
    return close >= 99.0

passed = 0
import sys
if len(sys.argv) > 1:
    cases = [int(x) for x in sys.argv[1].split(chr(44)) if x]
else:
    cases = [64, 1024, 2048]

print(f"{cases=}")

for topk in cases:
    passed += int(run_case(topk))
    
offload.uninitialize()

print("SUMMARY passed={}/{}".format(passed, len(cases)), flush=True)
raise SystemExit(0 if passed == len(cases) else 1)

print("end")
