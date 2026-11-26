# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.a3_host_staging import pack_spans


def test_stage_batches_keep_host_offsets_and_align_device_reads():
    spans = [(1032, 2000, 33), (1104, 2200, 64), (1072, 2400, 8)]
    assert list(pack_spans(spans, 1000, 256, 128)) == [
        [(0, 32, 2000, 33), (64, 104, 2200, 64)], [(0, 72, 2400, 8)]
    ]


@pytest.mark.parametrize("span", [(999, 1, 1), (1063, 1, 2), (1000, 0, 1), (1000, 1, 65), (1000, 1, 0)])
def test_invalid_stage_ranges(span):
    with pytest.raises(ValueError):
        list(pack_spans([span], 1000, 64, 64))
