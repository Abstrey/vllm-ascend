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
"""Unit tests for ForwardTimeCollector (design doc §13.1).

Uses FakeEvent/FakeEventFactory so the tests run on the CPU runner. The fake
events deliberately expose ``synchronize``/``wait_event``/... methods that
raise ``AssertionError``: any forbidden blocking call fails the test.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_ascend.core.forward_time_collector import (
    DEFAULT_WINDOW_SIZE,
    ForwardTimeCollector,
    ForwardTimeConfig,
    run_timed_draft_forward,
    run_timed_forward,
)

RANK = 0


class FakeEvent:
    """Scriptable stand-in for ``torch.npu.Event``."""

    def __init__(self, factory: "FakeEventFactory", seq: int):
        self.factory = factory
        self.seq = seq
        self.record_count = 0
        self.elapsed_calls = 0
        self._complete = True

    def record(self):
        if self.seq in self.factory.record_exc:
            raise self.factory.record_exc[self.seq]
        self.record_count += 1
        self._complete = False

    def query(self):
        if self.seq in self.factory.query_exc:
            raise self.factory.query_exc[self.seq]
        return self._complete

    def elapsed_time(self, other: "FakeEvent") -> float:
        self.elapsed_calls += 1
        if self.seq in self.factory.elapsed_exc:
            raise self.factory.elapsed_exc[self.seq]
        return self.factory.elapsed_value.get(self.seq, 1.0)

    def mark_complete(self) -> None:
        """Test helper: simulate the device having executed this event."""
        self._complete = True

    # Forbidden blocking APIs (design doc §3.1.4 / §13.1.3).
    def synchronize(self):
        raise AssertionError("synchronize() must never be called")

    def wait_event(self, _):
        raise AssertionError("wait_event() must never be called")

    def wait_stream(self, _):
        raise AssertionError("wait_stream() must never be called")

    def synchronize_stream(self, _):
        raise AssertionError("synchronize_stream() must never be called")


class FakeEventFactory:
    """Creates FakeEvents; exception/value scripting is keyed by event seq."""

    def __init__(self):
        self.events: list[FakeEvent] = []
        self._seq = 0
        self.create_exc: Exception | None = None
        self.record_exc: dict[int, Exception] = {}
        self.query_exc: dict[int, Exception] = {}
        self.elapsed_exc: dict[int, Exception] = {}
        self.elapsed_value: dict[int, float] = {}

    def __call__(self) -> FakeEvent:
        if self.create_exc is not None:
            raise self.create_exc
        event = FakeEvent(self, self._seq)
        self._seq += 1
        self.events.append(event)
        return event

    @property
    def created(self) -> int:
        return len(self.events)

    def mark_all_complete(self) -> None:
        for event in self.events:
            event.mark_complete()


def make_collector(
    window_size: int = 1000,
    target_bs: frozenset[int] | set[int] | None = None,
    enabled: bool = True,
    rank: int = RANK,
):
    factory = FakeEventFactory()
    logs: list[str] = []
    collector = ForwardTimeCollector(
        rank=rank,
        window_size=window_size,
        target_batch_sizes=target_bs or frozenset(),
        event_factory=factory,
        emit_log=logs.append,
        enabled=enabled,
    )
    return collector, factory, logs


def fwd(collector, factory, *, role="target", phase="decode", bs=4, value=None, complete=False):
    """Run one instrumented forward; optionally script its elapsed value/completion."""
    handle = collector.start(role, phase, bs)
    collector.finish(handle)
    if handle is not None:
        if value is not None:
            factory.elapsed_value[handle.start_event.seq] = value
        if complete:
            factory.mark_all_complete()
    return handle


def summaries_by_key(summaries):
    return {(s.key.role, s.key.phase, s.key.batch_size): s for s in summaries}


# --------------------------------------------------------------------- #
# §13.1.1 关闭路径
# --------------------------------------------------------------------- #


def test_disabled_start_returns_none_without_events():
    collector, factory, logs = make_collector(enabled=False)
    assert collector.start("target", "decode", 4) is None
    assert factory.created == 0
    assert collector.finish(None) is None
    assert collector.flush_window(final=True) == []
    assert logs == []


# --------------------------------------------------------------------- #
# §13.1.2 正常计时
# --------------------------------------------------------------------- #


def test_normal_timing_updates_stats_and_log_format():
    collector, factory, logs = make_collector(window_size=4)
    handle = fwd(collector, factory, bs=4, value=7.5, complete=True)
    assert handle is not None
    collector.drain_ready()

    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    summary = by_key[("target", "decode", 4)]
    assert summary.window_id == 1
    assert summary.count == 1
    assert summary.avg_ms == pytest.approx(7.5)
    assert summary.min_ms == pytest.approx(7.5)
    assert summary.max_ms == pytest.approx(7.5)
    assert summary.dropped == 0
    assert summary.errors == 0
    # elapsed_time called exactly once for the start event.
    assert handle.start_event.elapsed_calls == 1

    assert len(logs) == 1
    assert logs[0] == (
        "Forward timing: window=1 rank=0 role=target phase=decode bs=4 "
        "count=1 avg_ms=7.500 min_ms=7.500 max_ms=7.500 dropped=0 errors=0"
    )


def test_avg_min_max_over_multiple_samples():
    collector, factory, logs = make_collector(window_size=4)
    for value in (2.0, 4.0, 3.0):
        fwd(collector, factory, bs=4, value=value, complete=True)
    summaries = collector.flush_window(final=True)
    summary = summaries_by_key(summaries)[("target", "decode", 4)]
    assert summary.count == 3
    assert summary.avg_ms == pytest.approx(3.0)
    assert summary.min_ms == pytest.approx(2.0)
    assert summary.max_ms == pytest.approx(4.0)


# --------------------------------------------------------------------- #
# §13.1.3 非阻塞保证
# --------------------------------------------------------------------- #


def test_incomplete_event_never_reads_elapsed_time():
    collector, factory, logs = make_collector(window_size=4)
    handle = fwd(collector, factory, bs=4, value=7.5, complete=False)
    # A drain (as triggered by the next start()) must not read elapsed_time.
    collector.drain_ready()
    assert handle.start_event.elapsed_calls == 0
    assert collector.pending_count == 1

    # Window boundary does not wait either; only teardown finalizes it as dropped.
    summaries = collector.flush_window(final=True)
    summary = summaries_by_key(summaries)[("target", "decode", 4)]
    assert summary.count == 0
    assert summary.dropped == 1
    assert handle.start_event.elapsed_calls == 0
    assert logs == []  # count=0 keys emit no timing line


def test_drain_stops_at_first_incomplete_head():
    collector, factory, logs = make_collector(window_size=4)
    h1 = fwd(collector, factory, bs=4, value=1.0)
    h2 = fwd(collector, factory, bs=4, value=2.0)
    # Complete only the second sample's end event; the FIFO head must stop the drain.
    factory.events[3].mark_complete()
    collector.drain_ready()
    assert collector.pending_count == 2
    assert h1.start_event.elapsed_calls == 0
    assert h2.start_event.elapsed_calls == 0


# --------------------------------------------------------------------- #
# §13.1.4 窗口边界
# --------------------------------------------------------------------- #


def test_window_boundary_and_async_settlement():
    collector, factory, logs = make_collector(window_size=2)
    h1 = fwd(collector, factory, bs=4, value=1.0)
    h2 = fwd(collector, factory, bs=4, value=2.0)
    assert h1.window_id == h2.window_id == 1

    # Window 1 is full with all samples in flight: an intermediate flush
    # must not report anything (window not idle).
    assert collector.flush_window(final=False) == []

    # The device catches up on window 1 only now; the freed capacity admits
    # h3 into window 2.
    factory.events[0].mark_complete()
    factory.events[1].mark_complete()
    h3 = fwd(collector, factory, bs=4, value=3.0)
    assert h3.window_id == 2

    # h2 settles only after window 2 has opened: it must still be attributed
    # to its bound window 1, which then becomes idle and is emitted.
    factory.mark_all_complete()
    collector.drain_ready()
    summaries = collector.flush_window(final=False)
    by_window = {s.window_id: s for s in summaries}
    assert by_window[1].count == 2
    assert by_window[1].avg_ms == pytest.approx(1.5)
    # Window 2 is still open (accepted=1 < window_size): not emitted yet.
    assert 2 not in by_window

    tail = collector.flush_window(final=True)
    by_window_tail = {s.window_id: s for s in tail}
    assert by_window_tail[2].count == 1


# --------------------------------------------------------------------- #
# §13.1.5 多 BS 分组
# --------------------------------------------------------------------- #


def test_multi_bs_grouping():
    collector, factory, logs = make_collector(window_size=10)
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    fwd(collector, factory, bs=8, value=2.0, complete=True)
    fwd(collector, factory, bs=4, value=3.0, complete=True)
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].count == 2
    assert by_key[("target", "decode", 8)].count == 1
    assert by_key[("target", "decode", 4)].avg_ms == pytest.approx(2.0)


# --------------------------------------------------------------------- #
# §13.1.6 角色和阶段分组
# --------------------------------------------------------------------- #


def test_role_and_phase_grouping():
    collector, factory, logs = make_collector(window_size=10)
    fwd(collector, factory, role="target", phase="decode", bs=4, value=1.0, complete=True)
    fwd(collector, factory, role="target", phase="verify", bs=4, value=2.0, complete=True)
    fwd(collector, factory, role="draft", phase="decode", bs=4, value=3.0, complete=True)
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert set(by_key) == {("target", "decode", 4), ("target", "verify", 4), ("draft", "decode", 4)}
    assert by_key[("target", "decode", 4)].avg_ms == pytest.approx(1.0)
    assert by_key[("target", "verify", 4)].avg_ms == pytest.approx(2.0)
    assert by_key[("draft", "decode", 4)].avg_ms == pytest.approx(3.0)


# --------------------------------------------------------------------- #
# §13.1.7 BS 过滤
# --------------------------------------------------------------------- #


def test_bs_filter_skips_unmatched_without_dropped():
    collector, factory, logs = make_collector(window_size=10, target_bs={4})
    assert collector.start("target", "decode", 8) is None
    assert factory.created == 0  # filtered before any event creation
    fwd(collector, factory, bs=4, value=5.0, complete=True)
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert set(by_key) == {("target", "decode", 4)}
    assert by_key[("target", "decode", 4)].dropped == 0
    assert by_key[("target", "decode", 4)].count == 1


# --------------------------------------------------------------------- #
# §13.1.8 队列上限（准入控制）
# --------------------------------------------------------------------- #


def test_admission_control_when_in_flight_full():
    collector, factory, logs = make_collector(window_size=2)
    fwd(collector, factory, bs=4, value=1.0)
    fwd(collector, factory, bs=4, value=2.0)
    assert collector.pending_count + collector.active_count == 2

    h3 = collector.start("target", "decode", 6)
    assert h3 is None  # admission failure: pending(2) + active(0) >= window_size
    assert collector.pending_count == 2
    assert factory.created == 4  # no events created for the rejected sample

    summaries = collector.flush_window(final=True)
    by_window_key = {(s.window_id, s.key.batch_size): s for s in summaries}
    # Window 1 is full (closed): the rejected sample is attributed to window 2.
    assert by_window_key[(2, 6)].dropped == 1
    assert by_window_key[(2, 6)].count == 0
    # The two never-completing samples of window 1 are dropped by the final flush.
    assert by_window_key[(1, 4)].dropped == 2


# --------------------------------------------------------------------- #
# §13.1.9 未完成样本
# --------------------------------------------------------------------- #


def test_unfinished_sample_settles_later_and_final_drops_the_rest():
    collector, factory, logs = make_collector(window_size=2)
    fwd(collector, factory, bs=4, value=1.0)
    fwd(collector, factory, bs=4, value=2.0)
    fwd(collector, factory, bs=8, value=3.0)  # window 2, stays incomplete

    factory.events[0].mark_complete()
    factory.events[1].mark_complete()
    factory.events[2].mark_complete()
    factory.events[3].mark_complete()
    collector.drain_ready()
    summaries = collector.flush_window(final=False)
    by_window = {s.window_id: s for s in summaries}
    assert by_window[1].count == 2
    assert 2 not in by_window

    pool_before_final = collector.pool_size
    tail = collector.flush_window(final=True)
    by_window_tail = {s.window_id: s for s in tail}
    assert by_window_tail[2].count == 0
    assert by_window_tail[2].dropped == 1
    # The unfinished events must not be recycled into the pool.
    assert collector.pool_size == pool_before_final


# --------------------------------------------------------------------- #
# §13.1.10 异常路径
# --------------------------------------------------------------------- #


def test_event_creation_failure_counts_error():
    collector, factory, logs = make_collector(window_size=4)
    factory.create_exc = RuntimeError("no memory for event")
    assert collector.start("target", "decode", 4) is None
    factory.create_exc = None
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 1


def test_start_record_failure_counts_error():
    collector, factory, logs = make_collector(window_size=4)
    factory.record_exc[0] = RuntimeError("record failed")
    assert collector.start("target", "decode", 4) is None
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 0


def test_end_record_failure_counts_error_without_enqueue():
    collector, factory, logs = make_collector(window_size=4)
    handle = collector.start("target", "decode", 4)
    factory.record_exc[1] = RuntimeError("end record failed")  # seq 1 = end event
    collector.finish(handle)
    assert collector.pending_count == 0
    assert collector.active_count == 0
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 0


def test_query_failure_counts_error_and_continues():
    collector, factory, logs = make_collector(window_size=4)
    fwd(collector, factory, bs=4, value=1.0)
    fwd(collector, factory, bs=4, value=2.0)
    factory.query_exc[1] = RuntimeError("query failed")  # head sample's end event
    factory.events[3].mark_complete()
    collector.drain_ready()
    assert collector.pending_count == 0  # head popped as error, second settled
    factory.mark_all_complete()
    collector.drain_ready()
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 1


def test_elapsed_time_failure_counts_error():
    collector, factory, logs = make_collector(window_size=4)
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    factory.elapsed_exc[0] = RuntimeError("elapsed_time failed")
    collector.drain_ready()
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 0


# --------------------------------------------------------------------- #
# §13.1.11 非法耗时
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_value", [-1.0, float("nan"), float("inf")])
def test_invalid_elapsed_values_counted_as_errors(bad_value):
    collector, factory, logs = make_collector(window_size=4)
    fwd(collector, factory, bs=4, value=bad_value, complete=True)
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 0


# --------------------------------------------------------------------- #
# §13.1.12 配置校验
# --------------------------------------------------------------------- #


def test_collector_rejects_non_positive_window_size():
    with pytest.raises(ValueError, match="window_size"):
        ForwardTimeCollector(rank=RANK, window_size=0)
    with pytest.raises(ValueError, match="window_size"):
        ForwardTimeCollector(rank=RANK, window_size=-3)


def test_config_defaults():
    config = ForwardTimeConfig.from_env()
    assert config.enabled is False
    assert config.window_size == DEFAULT_WINDOW_SIZE
    assert config.target_batch_sizes == frozenset()


def test_config_parses_valid_values(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS", "1")
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE", "7")
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_TARGET_BATCH_SIZES", "4,5,4")
    config = ForwardTimeConfig.from_env()
    assert config.enabled is True
    assert config.window_size == 7
    assert config.target_batch_sizes == {4, 5}


@pytest.mark.parametrize("bad_window", ["0", "-5", "abc"])
def test_config_rejects_bad_window_size(monkeypatch, bad_window):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS", "1")
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE", bad_window)
    with pytest.raises(ValueError, match="VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE"):
        ForwardTimeConfig.from_env()


def test_config_rejects_bad_enable_flag(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS", "x")
    with pytest.raises(ValueError, match="VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS"):
        ForwardTimeConfig.from_env()


@pytest.mark.parametrize("bad_bs", ["0", "-2", "4,x", "a,b"])
def test_config_rejects_bad_target_batch_sizes(monkeypatch, bad_bs):
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS", "1")
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_TARGET_BATCH_SIZES", bad_bs)
    with pytest.raises(ValueError, match="VLLM_ASCEND_FORWARD_TIME_TARGET_BATCH_SIZES"):
        ForwardTimeConfig.from_env()


@pytest.mark.parametrize("bad_window,bad_bs", [("abc", ""), ("1e3", "4,x"), ("0", "a,b")])
def test_config_disabled_ignores_malformed_values(monkeypatch, bad_window, bad_bs):
    # The feature is off, so stale garbage in its env vars must not break
    # startup (from_env is consulted on every model runner init).
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_FORWARD_TIME_METRICS", "0")
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_WINDOW_SIZE", bad_window)
    monkeypatch.setenv("VLLM_ASCEND_FORWARD_TIME_TARGET_BATCH_SIZES", bad_bs)
    config = ForwardTimeConfig.from_env()
    assert config.enabled is False
    assert config.window_size == DEFAULT_WINDOW_SIZE
    assert config.target_batch_sizes == frozenset()


# --------------------------------------------------------------------- #
# §13.1.13 forward 异常（abort）
# --------------------------------------------------------------------- #


def test_abort_counts_dropped_and_does_not_pool_start_event():
    collector, factory, logs = make_collector(window_size=4)
    handle = collector.start("target", "decode", 4)
    original = RuntimeError("model forward failed")
    try:
        raise original
    except RuntimeError:
        collector.abort(handle)

    assert collector.active_count == 0
    assert collector.pool_size == 0  # recorded start_event must not be pooled
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].dropped == 1
    assert by_key[("target", "decode", 4)].count == 0
    assert by_key[("target", "decode", 4)].errors == 0


def test_abort_none_is_noop():
    collector, factory, logs = make_collector(window_size=4)
    collector.abort(None)
    assert collector.flush_window(final=True) == []


def test_abort_of_last_sample_emits_closed_window():
    # Regression: a closed window whose last in-flight sample aborts must be
    # emitted right away, not linger until the next settled sample or teardown.
    collector, factory, logs = make_collector(window_size=1)
    handle = collector.start("target", "decode", 4)  # accepted=1 -> window closed
    try:
        raise RuntimeError("model boom")
    except RuntimeError:
        collector.abort(handle)

    summaries = collector.flush_window(final=False)
    by_key = summaries_by_key(summaries)
    assert len(summaries) == 1
    assert by_key[("target", "decode", 4)].dropped == 1
    assert by_key[("target", "decode", 4)].count == 0
    assert collector.live_window_count == 0


def test_end_record_failure_emits_closed_window():
    # Regression: same promptness requirement when the end event fails to
    # record and the sample settles into the error counter instead.
    collector, factory, logs = make_collector(window_size=1)
    handle = collector.start("target", "decode", 4)  # start event seq 0
    factory.record_exc[1] = RuntimeError("end record failed")  # end event seq 1
    collector.finish(handle)

    summaries = collector.flush_window(final=False)
    by_key = summaries_by_key(summaries)
    assert len(summaries) == 1
    assert by_key[("target", "decode", 4)].errors == 1
    assert by_key[("target", "decode", 4)].count == 0
    assert collector.live_window_count == 0


# --------------------------------------------------------------------- #
# 补充：事件池、日志排序、flush 幂等
# --------------------------------------------------------------------- #


def test_event_pool_reuse_and_cap():
    collector, factory, logs = make_collector(window_size=4)
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    collector.drain_ready()
    assert factory.created == 2
    assert collector.pool_size == 2

    h2 = fwd(collector, factory, bs=4, value=2.0)
    assert factory.created == 2  # start event came from the pool
    assert h2.start_event.seq in (0, 1)

    # Pool cap is 2 * window_size; recycling beyond the cap drops the object.
    fwd(collector, factory, bs=4, value=3.0, complete=True)
    fwd(collector, factory, bs=4, value=4.0, complete=True)
    collector.drain_ready()
    assert collector.pool_size <= 8


def test_log_lines_ordered_by_count_desc():
    collector, factory, logs = make_collector(window_size=10)
    fwd(collector, factory, bs=8, value=1.0, complete=True)
    for _ in range(3):
        fwd(collector, factory, bs=4, value=1.0, complete=True)
    for _ in range(2):
        fwd(collector, factory, bs=5, value=1.0, complete=True)
    collector.flush_window(final=True)
    assert [line.split("bs=")[1].split()[0] for line in logs] == ["4", "5", "8"]


def test_flush_returns_summaries_once():
    collector, factory, logs = make_collector(window_size=2)
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    fwd(collector, factory, bs=4, value=2.0, complete=True)
    first = collector.flush_window(final=False)
    assert len(first) == 1
    assert collector.flush_window(final=False) == []  # consumed
    assert collector.flush_window(final=True) == []


def test_finalized_collector_rejects_new_samples():
    collector, factory, logs = make_collector(window_size=2)
    collector.flush_window(final=True)
    assert collector.start("target", "decode", 4) is None
    assert factory.created == 0


def test_multiple_windows_increment_ids():
    collector, factory, logs = make_collector(window_size=1)
    for i in range(3):
        fwd(collector, factory, bs=4, value=float(i), complete=True)
        collector.drain_ready()
    summaries = collector.flush_window(final=True)
    assert sorted(s.window_id for s in summaries) == [1, 2, 3]


# --------------------------------------------------------------------- #
# fail-open 契约（design doc §10）：collector 任何异常都不得影响业务 forward
# --------------------------------------------------------------------- #


def test_run_timed_forward_runs_forward_when_start_raises():
    collector = MagicMock()
    collector.start.side_effect = RuntimeError("collector broken")
    run_forward = MagicMock(return_value="hidden_states")

    result = run_timed_forward(collector, "target", lambda: "decode", 4, run_forward)

    assert result == "hidden_states"
    run_forward.assert_called_once()
    collector.finish.assert_not_called()
    collector.abort.assert_not_called()


def test_run_timed_forward_runs_forward_when_phase_raises():
    collector = MagicMock()

    def bad_phase():
        raise RuntimeError("phase boom")

    run_forward = MagicMock(return_value="hidden_states")
    result = run_timed_forward(collector, "target", bad_phase, 4, run_forward)

    assert result == "hidden_states"
    run_forward.assert_called_once()
    collector.start.assert_not_called()


def test_run_timed_forward_preserves_business_exception_when_abort_raises():
    collector = MagicMock()
    collector.abort.side_effect = RuntimeError("abort boom")
    run_forward = MagicMock(side_effect=ValueError("business failure"))

    with pytest.raises(ValueError, match="business failure"):
        run_timed_forward(collector, "target", lambda: "decode", 4, run_forward)

    collector.abort.assert_called_once()


def test_run_timed_forward_returns_result_when_finish_raises():
    collector = MagicMock()
    collector.finish.side_effect = RuntimeError("finish boom")
    run_forward = MagicMock(return_value="ok")

    assert run_timed_forward(collector, "target", lambda: "decode", 4, run_forward) == "ok"
    collector.finish.assert_called_once()


def test_run_timed_draft_forward_runs_draft_when_start_raises():
    collector = MagicMock()
    collector.start.side_effect = RuntimeError("collector broken")
    runner = SimpleNamespace(
        forward_time_collector=collector,
        input_batch=SimpleNamespace(num_reqs=3),
        _draft_time_phase=lambda: "decode",
    )
    run_draft = MagicMock(return_value="draft_ids")

    assert run_timed_draft_forward(runner, 3, run_draft) == "draft_ids"
    run_draft.assert_called_once()


def test_emit_log_failure_does_not_break_emission():
    def broken_emit(msg):
        raise RuntimeError("log handler broken")

    factory = FakeEventFactory()
    collector = ForwardTimeCollector(rank=RANK, window_size=2, event_factory=factory, emit_log=broken_emit)
    fwd(collector, factory, bs=4, value=1.0, complete=True)
    fwd(collector, factory, bs=8, value=2.0, complete=True)

    # Must not raise, and summaries for every key still come out.
    summaries = collector.flush_window(final=True)
    by_key = summaries_by_key(summaries)
    assert by_key[("target", "decode", 4)].count == 1
    assert by_key[("target", "decode", 8)].count == 1
