# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import pack_spans


def test_stage_batches_keep_host_offsets_and_align_device_reads():
    spans = [(1032, 2000, 33), (1104, 2200, 64), (1072, 2400, 8)]
    assert list(pack_spans(spans, 1000, 256, 128)) == [
        [(0, 32, 2000, 33), (64, 104, 2200, 64)],
        [(0, 72, 2400, 8)],
    ]


@pytest.mark.parametrize("span", [(999, 1, 1), (1063, 1, 2), (1000, 0, 1), (1000, 1, 65), (1000, 1, 0)])
def test_invalid_stage_ranges(span):
    with pytest.raises(ValueError):
        list(pack_spans([span], 1000, 64, 64))


def test_staged_read_commits_valid_bytes_before_reusing_the_window(monkeypatch):
    import ctypes
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch

    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import A3HostKVStaging

    events = []
    window = A3HostKVStaging.__new__(A3HostKVStaging)
    window.device = "cpu"
    window.buffer = torch.empty(64, dtype=torch.uint8)
    window.stream = Mock()
    window.stream.synchronize.side_effect = lambda: events.append("commit")
    npu = SimpleNamespace(device=lambda _: nullcontext(), stream=lambda _: nullcontext())
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    host = torch.zeros(100, dtype=torch.uint8)
    source = torch.arange(1, 65, dtype=torch.uint8)

    def read(session, destinations, sources, sizes):
        assert session == "peer"
        events.append("read")
        for dest, src, size in zip(destinations, sources, sizes, strict=True):
            ctypes.memmove(dest, src, size)
        return 0

    spans = [(host.data_ptr() + 1, source.data_ptr(), 40), (host.data_ptr() + 64, source.data_ptr() + 40, 24)]
    window.read(SimpleNamespace(batch_transfer_sync_read=read), "peer", host, spans)
    assert events == ["read", "commit", "read", "commit"]
    expected = torch.zeros_like(host)
    expected[1:41] = source[:40]
    expected[64:88] = source[40:]
    assert torch.equal(host, expected)


def test_packed_host_view_preserves_layer_offset_and_page_padding():
    import torch

    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import host_byte_window

    storage = torch.zeros(48, dtype=torch.int8)
    layer = storage.as_strided((2, 4), (16, 1), 5)
    window = host_byte_window(layer, pages=2, stride=16, payload=4)
    assert window.data_ptr() == layer.data_ptr()
    assert window.numel() == 20
    window[16:20].fill_(3)
    assert torch.equal(storage[21:25], torch.full((4,), 3, dtype=torch.int8))
    assert torch.count_nonzero(storage) == 4
    with pytest.raises(ValueError):
        host_byte_window(layer, pages=2, stride=3, payload=4)


def test_validate_packed_main_checks_wire_bytes_and_dtype():
    import torch

    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import validate_packed_main
    from vllm_ascend.hisparse.c8 import C8_ROW_BYTES

    main = torch.empty(1, dtype=torch.int8)
    validate_packed_main((main,), 128 * C8_ROW_BYTES, 128)
    for invalid, size in ((main, 128), ((main, main), 128 * C8_ROW_BYTES), (main.float(), 128 * C8_ROW_BYTES)):
        with pytest.raises(ValueError):
            validate_packed_main(invalid, size, 128)


def test_a3_staging_topology_allows_replicated_prefill_only():
    from types import SimpleNamespace

    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import validate_a3_staging

    parallel = SimpleNamespace(prefill_context_parallel_size=4, decode_context_parallel_size=1)
    config = SimpleNamespace(
        parallel_config=parallel,
        cache_config=SimpleNamespace(cache_dtype="int8"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="glm_moe_dsa")),
        additional_config={},
    )
    extra = {"a3_stage_main_kv": True, "memfabric_transfer_protocol": "device_rdma"}
    assert not validate_a3_staging(None, {}, hisparse_target=False, is_producer=False)
    assert validate_a3_staging(config, extra, hisparse_target=True, is_producer=True)
    with pytest.raises(ValueError, match="unsharded"):
        validate_a3_staging(config, extra, hisparse_target=True, is_producer=False)
    parallel.prefill_context_parallel_size = 1
    assert validate_a3_staging(config, extra, hisparse_target=True, is_producer=False)
    parallel.decode_context_parallel_size = 2
    with pytest.raises(ValueError, match="DCP=1"):
        validate_a3_staging(config, extra, hisparse_target=True, is_producer=True)
    with pytest.raises(ValueError, match="device_rdma"):
        validate_a3_staging(
            config, dict(extra, memfabric_transfer_protocol="host_rdma"), hisparse_target=True, is_producer=True
        )
