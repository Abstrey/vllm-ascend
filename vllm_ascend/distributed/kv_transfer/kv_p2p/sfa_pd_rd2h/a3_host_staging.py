# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read remote Main KV through bounded local A3 HBM, then commit it to Host."""

from collections.abc import Iterable, Iterator

import torch

from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

Span = tuple[int, int, int]
StagedSpan = tuple[int, int, int, int]
ALIGNMENT = 2 << 20
BUFFER_BYTES = 32 << 20


def pack_spans(spans: Iterable[Span], base: int, size: int, capacity: int) -> Iterator[list[StagedSpan]]:
    batch = []
    used = 0
    for local, remote, length in spans:
        offset = local - base
        if length <= 0 or length > capacity or offset < 0 or offset + length > size or remote <= 0:
            raise ValueError("Main KV transfer span exceeds its registered range")
        if used + length > capacity:
            yield batch
            batch = []
            used = 0
        batch.append((used, offset, remote, length))
        used = (used + length + 63) & ~63
    if batch:
        yield batch


def validate_a3_staging(config, extra, *, hisparse_target: bool, is_producer: bool) -> bool:
    """Confirm both the wire layout and the topology required for replicated KV."""
    enabled = extra.get("a3_stage_main_kv", False)
    if not isinstance(enabled, bool):
        raise ValueError("a3_stage_main_kv must be a boolean")
    if not enabled:
        return False
    if (
        not hisparse_target
        or extra.get("memfabric_transfer_protocol") != "device_rdma"
        or config.cache_config.cache_dtype != "int8"
        or config.model_config.hf_text_config.model_type != "glm_moe_dsa"
        or (config.additional_config or {}).get("enable_kvpp", False)
    ):
        raise ValueError("A3 staging requires layerwise GLM DSA INT8, device_rdma and replicated KV")
    parallel = config.parallel_config
    if parallel.decode_context_parallel_size != 1 or (not is_producer and parallel.prefill_context_parallel_size != 1):
        raise ValueError("A3 staging requires unsharded Decode KV and DCP=1")
    return True


def validate_packed_main(tensors, page_bytes: int, page_tokens: int) -> None:
    views = tensors if isinstance(tensors, (tuple, list)) else (tensors,)
    if (
        len(views) != 1
        or views[0].dtype != torch.int8
        or page_tokens <= 0
        or page_bytes != page_tokens * C8_ROW_BYTES
    ):
        raise ValueError("Expected one packed INT8 C8 page for A3 layerwise transfer")


def host_byte_window(tensor, pages: int, stride: int, payload: int):
    """Expose the bytes covered by this layer, preserving its storage offset."""
    if tensor.device.type != "cpu" or tensor.dtype not in (torch.int8, torch.uint8):
        raise ValueError("A3 staging expects a packed INT8 Main Host view")
    if pages <= 0 or payload <= 0 or stride < payload:
        raise ValueError("Invalid A3 Host page geometry")
    return tensor.view(torch.uint8).as_strided(((pages - 1) * stride + payload,), (1,))


class A3HostKVStaging:
    def __init__(self, device):
        self.device = device
        self.stream = torch.npu.Stream(device=device)
        backing = torch.empty(BUFFER_BYTES + ALIGNMENT, dtype=torch.uint8, device=device)
        offset = -backing.data_ptr() % ALIGNMENT
        self.buffer = backing[offset : offset + BUFFER_BYTES]

    def read(self, engine, session, host, spans):
        if host.device.type != "cpu" or not host.is_contiguous():
            raise ValueError("A3 Main KV staging requires a contiguous Host tensor")
        host_bytes = host.view(torch.uint8).view(-1)
        batches = pack_spans(spans, host_bytes.data_ptr(), host_bytes.numel(), self.buffer.numel())
        with torch.npu.device(self.device), torch.npu.stream(self.stream):
            for batch in batches:
                result = engine.batch_transfer_sync_read(
                    session,
                    [self.buffer.data_ptr() + begin for begin, _, _, _ in batch],
                    [remote for _, _, remote, _ in batch],
                    [length for _, _, _, length in batch],
                )
                if result != 0:
                    raise RuntimeError(f"A3 Main KV staged RDMA read failed with code {result}")
                try:
                    for begin, offset, _, length in batch:
                        host_bytes[offset : offset + length].copy_(self.buffer[begin : begin + length], non_blocking=True)
                finally:
                    # Finish any queued copies before reusing the RDMA window, also on error.
                    self.stream.synchronize()
