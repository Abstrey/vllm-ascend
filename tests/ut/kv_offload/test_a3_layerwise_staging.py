# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_consumer import (
    HiSparseLayerwiseReadThread,
    HiSparseReadState,
    LayerDestination,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.hisparse_layout import PageBuffer
from vllm_ascend.hisparse.c8 import C8_ROW_BYTES


def test_packed_tail_uses_staging_and_indexer_uses_direct_rdma():
    thread = HiSparseLayerwiseReadThread.__new__(HiSparseLayerwiseReadThread)
    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.return_value = 0
    staging = MagicMock()
    staged = []
    staging.read.side_effect = lambda engine, session, host, entries: staged.extend(entries)
    width = 2 * C8_ROW_BYTES
    dst = LayerDestination(PageBuffer(10000, width, width + 32, 4), (PageBuffer(20000, 4, 8, 4),), MagicMock())
    thread._state = HiSparseReadState({"layer": dst}, {"req": ([1], [2])}, 2, 1024, 128, 0, 1, staging=staging)
    meta = {
        "main_tensor_count": 1,
        "has_indexer": True,
        "base_addrs": [30000, 40000],
        "block_len": [width, 4],
        "block_size_scale": [1, 1],
        "block_shapes": [(width,), (2, 2)],
        "num_logical_blocks": [4, 4],
        "block_tokens": [2, 2],
        "packed_int8_c8": True,
    }
    args = ("layer", [("req", [3], [1], 0, 0, 1)], "peer", {"layer": meta})
    thread._do_read_batch(*args)
    assert staged == [(10000 + width + 32, 30000 + 3 * width, C8_ROW_BYTES)]
    thread.engine.batch_transfer_sync_read.assert_called_once_with("peer", [20016], [40004], [2])
    thread._state = replace(thread._state, staging=None)
    with pytest.raises(ValueError, match="disagree"):
        thread._do_read_batch(*args)


def test_replicated_pcp_worker_does_not_publish_or_wait():
    from vllm_ascend.distributed.kv_transfer.kv_p2p.sfa_pd_rd2h.worker import SFAPDRD2HProducerWorker

    worker = SFAPDRD2HProducerWorker.__new__(SFAPDRD2HProducerWorker)
    worker._pcp_replica = True
    worker.register_kv_caches({})
    worker.bind_connector_metadata(None)
    worker.save_kv_layer("layer", [], None, None)
    worker.wait_for_layer_reuse(0)
