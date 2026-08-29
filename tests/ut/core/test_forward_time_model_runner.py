#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""V1 model runner integration tests for forward time metrics (design doc §13.2).

``NPUModelRunner.__init__`` loads a full model, so tests build the runner via
``object.__new__`` and only the attributes touched by the instrumented paths
are set up.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_ascend.ascend_config import ForwardTimeMetricsConfig
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def make_runner(metadata_attn_state=None, num_reqs=4, collector=None):
    runner = object.__new__(NPUModelRunner)
    runner.forward_time_collector = collector
    runner.metadata_attn_state = metadata_attn_state
    runner.input_batch = SimpleNamespace(num_reqs=num_reqs)
    runner._model_forward = MagicMock(return_value="hidden_states")
    return runner


def sched_output(spec_tokens=None):
    return SimpleNamespace(scheduled_spec_decode_tokens=spec_tokens or {})


# --------------------------------------------------------------------- #
# §13.2.1 普通 decode：真实 num_reqs 与 role/phase
# --------------------------------------------------------------------- #


def test_decode_step_uses_real_num_reqs():
    collector = MagicMock()
    runner = make_runner(
        metadata_attn_state=AscendAttentionState.DecodeOnly, num_reqs=7, collector=collector
    )
    result = runner._timed_model_forward(sched_output(), 16)
    assert result == "hidden_states"
    collector.start.assert_called_once_with("target", "decode", 7)
    collector.finish.assert_called_once_with(collector.start.return_value)
    runner._model_forward.assert_called_once()


# --------------------------------------------------------------------- #
# §13.2.2 / §13.2.3 / §13.2.4 阶段映射表
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,spec_tokens,expected",
    [
        (AscendAttentionState.DecodeOnly, {}, "decode"),
        (AscendAttentionState.SpecDecoding, {"r0": [1, 2]}, "verify"),
        # MTP + PD disaggregation: SpecDecoding state without scheduled draft
        # tokens is a pure decode step and must not be reported as verify.
        (AscendAttentionState.SpecDecoding, {}, "decode"),
        (AscendAttentionState.PrefillNoCache, {}, "prefill"),
        (AscendAttentionState.PrefillCacheHit, {}, "prefill"),
        (AscendAttentionState.ChunkedPrefill, {}, "mixed"),
        # A genuine mixed batch carrying decode draft tokens stays "mixed";
        # only the pre-overlay SpecDecoding state plus draft tokens is verify
        # (the non-MTP overlay rewrites self.attn_state, not metadata_attn_state).
        (AscendAttentionState.ChunkedPrefill, {"r0": [1]}, "mixed"),
        # Defensive default before the first metadata build.
        (None, {}, "prefill"),
    ],
)
def test_phase_mapping(state, spec_tokens, expected):
    runner = make_runner(metadata_attn_state=state)
    assert runner._forward_time_phase(sched_output(spec_tokens)) == expected


# --------------------------------------------------------------------- #
# §6.2 draft phase 映射：与 target 共用 metadata_attn_state 词表
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "state,expected",
    [
        (AscendAttentionState.DecodeOnly, "decode"),
        # Draft generation during verify steps is decode-like, never "verify".
        (AscendAttentionState.SpecDecoding, "decode"),
        (AscendAttentionState.PrefillNoCache, "prefill"),
        (AscendAttentionState.PrefillCacheHit, "prefill"),
        # Chunked-prefill batches may mix prefill and decode requests.
        (AscendAttentionState.ChunkedPrefill, "mixed"),
        # Unclassifiable must fall back to mixed, never decode (design §6.2).
        (None, "mixed"),
    ],
)
def test_draft_phase_mapping(state, expected):
    runner = make_runner(metadata_attn_state=state)
    assert runner._draft_time_phase() == expected


def test_draft_phase_defaults_to_mixed_without_state_attr():
    runner = object.__new__(NPUModelRunner)  # metadata_attn_state never set
    assert runner._draft_time_phase() == "mixed"


# --------------------------------------------------------------------- #
# §10 forward 异常：abort 且原异常原样传播
# --------------------------------------------------------------------- #


def test_forward_exception_aborts_and_reraises():
    collector = MagicMock()
    runner = make_runner(
        metadata_attn_state=AscendAttentionState.DecodeOnly, collector=collector
    )
    runner._model_forward = MagicMock(side_effect=RuntimeError("model boom"))
    with pytest.raises(RuntimeError, match="model boom"):
        runner._timed_model_forward(sched_output(), 16)
    collector.start.assert_called_once()
    collector.abort.assert_called_once_with(collector.start.return_value)
    collector.finish.assert_not_called()


def test_forward_runs_when_collector_start_raises():
    # Fail-open: instrumentation must never skip the model forward.
    collector = MagicMock()
    collector.start.side_effect = RuntimeError("collector broken")
    runner = make_runner(
        metadata_attn_state=AscendAttentionState.DecodeOnly, collector=collector
    )
    result = runner._timed_model_forward(sched_output(), 16)
    assert result == "hidden_states"
    runner._model_forward.assert_called_once()
    collector.finish.assert_not_called()


def test_forward_exception_preserved_when_abort_raises():
    # Fail-open: cleanup failures must not mask the business exception
    # (match discriminates "model boom" from the abort-side "abort boom").
    collector = MagicMock()
    collector.abort.side_effect = RuntimeError("abort boom")
    runner = make_runner(
        metadata_attn_state=AscendAttentionState.DecodeOnly, collector=collector
    )
    runner._model_forward = MagicMock(side_effect=RuntimeError("model boom"))
    with pytest.raises(RuntimeError, match="model boom"):
        runner._timed_model_forward(sched_output(), 16)
    collector.abort.assert_called_once()


def test_forward_result_returned_when_finish_raises():
    # Fail-open: a metrics failure after a successful forward must not turn
    # it into a failed one.
    collector = MagicMock()
    collector.finish.side_effect = RuntimeError("finish boom")
    runner = make_runner(
        metadata_attn_state=AscendAttentionState.DecodeOnly, collector=collector
    )
    assert runner._timed_model_forward(sched_output(), 16) == "hidden_states"
    collector.finish.assert_called_once()


# --------------------------------------------------------------------- #
# 功能关闭：直通、零开销
# --------------------------------------------------------------------- #


def test_disabled_collector_is_passthrough():
    runner = make_runner(metadata_attn_state=AscendAttentionState.DecodeOnly, collector=None)
    result = runner._timed_model_forward(sched_output(), 16)
    assert result == "hidden_states"
    runner._model_forward.assert_called_once()


# --------------------------------------------------------------------- #
# §7.4.7 teardown 钩子
# --------------------------------------------------------------------- #


def test_shutdown_flushes_tail_windows():
    collector = MagicMock()
    runner = make_runner(collector=collector)
    with patch.object(GPUModelRunner, "shutdown"):
        runner.shutdown()
    collector.flush_window.assert_called_once_with(final=True)


def test_shutdown_calls_super_cleanup():
    # Regression: the override used to shadow GPUModelRunner.shutdown, which
    # releases model weights / workspace / global caches for in-process
    # engine recreation; it must delegate after flushing the tail windows.
    collector = MagicMock()
    runner = make_runner(collector=collector)
    calls = []
    with patch.object(GPUModelRunner, "shutdown", side_effect=lambda: calls.append("super")) as super_shutdown:
        collector.flush_window.side_effect = lambda **_: calls.append("flush")
        runner.shutdown()
    super_shutdown.assert_called_once()
    assert calls == ["flush", "super"]  # flush first, then base cleanup


def test_shutdown_calls_super_even_when_flush_raises():
    # A metrics flush failure must not block the base cleanup (weights,
    # workspace, caches); the error still surfaces to the caller.
    collector = MagicMock()
    collector.flush_window.side_effect = RuntimeError("flush boom")
    runner = make_runner(collector=collector)
    with patch.object(GPUModelRunner, "shutdown") as super_shutdown:
        with pytest.raises(RuntimeError, match="flush boom"):
            runner.shutdown()
    super_shutdown.assert_called_once()
    collector.flush_window.assert_called_once_with(final=True)


def test_shutdown_without_collector_is_noop():
    runner = make_runner(collector=None)
    with patch.object(GPUModelRunner, "shutdown"):
        runner.shutdown()  # must not raise and must still reach super()


# --------------------------------------------------------------------- #
# additional-config 通道（spawn 环境下 env 变量到不了 worker，主配置入口）
# --------------------------------------------------------------------- #


def test_forward_time_metrics_config_defaults():
    config = ForwardTimeMetricsConfig()
    assert config.enabled is False
    assert config.window_size == 1000
    assert config.target_batch_sizes == ()


def test_forward_time_metrics_config_parses():
    config = ForwardTimeMetricsConfig(
        {"enabled": True, "window_size": 100, "target_batch_sizes": [4, 5]}
    )
    assert config.enabled is True
    assert config.window_size == 100
    assert config.target_batch_sizes == (4, 5)


@pytest.mark.parametrize(
    "bad",
    [
        {"enabled": True, "window_size": 0},
        {"enabled": True, "window_size": -1},
        {"enabled": True, "target_batch_sizes": [0]},
        {"enabled": True, "target_batch_sizes": [4, -2]},
    ],
)
def test_forward_time_metrics_config_rejects_invalid(bad):
    with pytest.raises(ValueError, match="forward_time_metrics_config"):
        ForwardTimeMetricsConfig(bad)


# --------------------------------------------------------------------- #
# §13.2.6 只有 execute_model 业务路径被埋点
# --------------------------------------------------------------------- #


def test_only_execute_model_is_instrumented():
    execute_src = inspect.getsource(NPUModelRunner.execute_model)
    dummy_src = inspect.getsource(NPUModelRunner._dummy_run)
    forward_src = inspect.getsource(NPUModelRunner._model_forward)
    timed_src = inspect.getsource(NPUModelRunner._timed_model_forward)

    assert "_timed_model_forward(" in execute_src
    # capture/warmup/profile/dummy paths call _model_forward directly and
    # must never enter business statistics.
    assert "_timed_model_forward" not in dummy_src
    assert "_timed_model_forward" not in forward_src
    # The wrapper itself delegates to _model_forward exactly once per path.
    assert timed_src.count("self._model_forward(") == 2  # disabled path + timed path

    # The pre-overlay attention state must stay exposed for phase mapping.
    build_src = inspect.getsource(NPUModelRunner._build_attn_state)
    assert "self.metadata_attn_state = attn_state" in build_src
