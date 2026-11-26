# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Read remote Main KV through bounded local A3 HBM, then commit it to Host."""

from collections.abc import Iterable, Iterator


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


class A3HostKVStaging:
    def __init__(self, device):
        import torch

        self.device = device
        self.stream = torch.npu.Stream(device=device)
        backing = torch.empty(BUFFER_BYTES + ALIGNMENT, dtype=torch.uint8, device=device)
        offset = -backing.data_ptr() % ALIGNMENT
        self.buffer = backing[offset:offset + BUFFER_BYTES]

    def read(self, engine, session, host, spans):
        import torch

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
                        host_bytes[offset:offset + length].copy_(self.buffer[begin:begin + length], non_blocking=True)
                finally:
                    # Finish any queued copies before reusing the RDMA window, also on error.
                    self.stream.synchronize()
