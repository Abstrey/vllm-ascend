#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
"""Proposer integration tests for forward time metrics (design doc §13.2).

The proposers' ``__init__`` needs a full model runner and model, so the
shared wrapper and the per-proposer overrides are tested on bare instances
built via ``object.__new__``.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm_ascend.core.forward_time_collector import run_timed_draft_forward
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def make_runner(collector, num_reqs=5, phase="decode"):
    return SimpleNamespace(
        forward_time_collector=collector,
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        _draft_time_phase=lambda: phase,
    )


# --------------------------------------------------------------------- #
# 共享封装 run_timed_draft_forward
# --------------------------------------------------------------------- #


def test_run_timed_draft_forward_records_role_phase_bs():
    collector = MagicMock()
    runner = make_runner(collector, num_reqs=5, phase="decode")
    run_draft = MagicMock(return_value="draft_ids")

    result = run_timed_draft_forward(runner, 5, run_draft)

    assert result == "draft_ids"
    run_draft.assert_called_once()
    collector.start.assert_called_once_with("draft", "decode", 5)
    collector.finish.assert_called_once_with(collector.start.return_value)


def test_run_timed_draft_forward_derives_phase_from_runner():
    # The wrapper never guesses the phase: it comes from the runner's
    # _draft_time_phase so draft and target share one vocabulary.
    collector = MagicMock()
    runner = make_runner(collector, num_reqs=2, phase="mixed")
    run_timed_draft_forward(runner, 2, MagicMock())
    collector.start.assert_called_once_with("draft", "mixed", 2)


def test_run_timed_draft_forward_falls_back_to_mixed_phase():
    # A runner without _draft_time_phase (older runner, bare test doubles)
    # must be reported as mixed, never as decode (design doc §6.2).
    collector = MagicMock()
    runner = SimpleNamespace(forward_time_collector=collector)
    run_timed_draft_forward(runner, 3, MagicMock())
    collector.start.assert_called_once_with("draft", "mixed", 3)


def test_run_timed_draft_forward_aborts_on_exception():
    collector = MagicMock()
    runner = make_runner(collector, phase="prefill")
    run_draft = MagicMock(side_effect=RuntimeError("draft boom"))

    with pytest.raises(RuntimeError, match="draft boom"):
        run_timed_draft_forward(runner, 5, run_draft)

    collector.start.assert_called_once()
    collector.abort.assert_called_once_with(collector.start.return_value)
    collector.finish.assert_not_called()


def test_run_timed_draft_forward_passthrough_when_disabled():
    runner = make_runner(collector=None)
    run_draft = MagicMock(return_value="draft_ids")
    assert run_timed_draft_forward(runner, 5, run_draft) == "draft_ids"
    run_draft.assert_called_once()


def test_run_timed_draft_forward_runner_without_collector_attr():
    # Proposers driven by runners that predate this feature (or mocks in
    # upstream tests) must keep working.
    runner = SimpleNamespace()
    run_draft = MagicMock(return_value="draft_ids")
    assert run_timed_draft_forward(runner, 5, run_draft) == "draft_ids"
    run_draft.assert_called_once()


def test_run_timed_draft_forward_none_num_reqs_disables_sampling():
    # Explicit None num_reqs means "cannot classify": skip instead of guessing.
    collector = MagicMock()
    runner = make_runner(collector)
    run_timed_draft_forward(runner, None, MagicMock())
    collector.start.assert_not_called()


# --------------------------------------------------------------------- #
# Medusa：不继承 AscendSpecDecodeBaseProposer 的特殊 proposer
# --------------------------------------------------------------------- #


def _make_medusa_proposer(collector, num_reqs=3, phase="decode"):
    from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer

    proposer = object.__new__(AscendMedusaProposer)
    proposer.runner = make_runner(collector, num_reqs=num_reqs, phase=phase)
    return proposer


def _medusa_inputs():
    # shape[0] == len(valid_sampled_token_ids) keeps propose() on the
    # passthrough branch, so no torch tensor ops (and no device) are needed.
    return [[1], [2], [3], [4]], MagicMock(), MagicMock(), SimpleNamespace(shape=[4])


def test_medusa_propose_is_instrumented():
    from vllm.v1.spec_decode.medusa import MedusaProposer

    collector = MagicMock()
    proposer = _make_medusa_proposer(collector, num_reqs=3, phase="decode")
    valid_ids, sampling_metadata, spec_decode_metadata, hidden_states = _medusa_inputs()

    with patch.object(MedusaProposer, "propose", return_value="spec_ids") as upstream:
        result = proposer.propose(valid_ids, sampling_metadata, spec_decode_metadata, hidden_states)

    assert result == "spec_ids"
    upstream.assert_called_once()
    collector.start.assert_called_once_with("draft", "decode", 3)
    collector.finish.assert_called_once_with(collector.start.return_value)


def test_medusa_propose_aborts_on_exception():
    from vllm.v1.spec_decode.medusa import MedusaProposer

    collector = MagicMock()
    proposer = _make_medusa_proposer(collector)
    valid_ids, sampling_metadata, spec_decode_metadata, hidden_states = _medusa_inputs()

    with patch.object(MedusaProposer, "propose", side_effect=RuntimeError("upstream boom")):
        with pytest.raises(RuntimeError, match="upstream boom"):
            proposer.propose(valid_ids, sampling_metadata, spec_decode_metadata, hidden_states)

    collector.abort.assert_called_once_with(collector.start.return_value)
    collector.finish.assert_not_called()


def test_medusa_propose_passthrough_without_runner():
    from vllm.v1.spec_decode.medusa import MedusaProposer

    proposer = _make_medusa_proposer(collector=None)
    proposer.runner = None
    valid_ids, sampling_metadata, spec_decode_metadata, hidden_states = _medusa_inputs()

    with patch.object(MedusaProposer, "propose", return_value="spec_ids") as upstream:
        result = proposer.propose(valid_ids, sampling_metadata, spec_decode_metadata, hidden_states)

    assert result == "spec_ids"
    upstream.assert_called_once()


# --------------------------------------------------------------------- #
# ExtractHiddenStates：同样不继承 AscendSpecDecodeBaseProposer
# --------------------------------------------------------------------- #


def _make_ehs_proposer(collector, num_reqs=2, phase="prefill"):
    from vllm_ascend.spec_decode.extract_hidden_states_proposer import AscendExtractHiddenStatesProposer

    proposer = object.__new__(AscendExtractHiddenStatesProposer)
    proposer.runner = make_runner(collector, num_reqs=num_reqs, phase=phase)
    return proposer


def test_extract_hidden_states_propose_is_instrumented():
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

    collector = MagicMock()
    proposer = _make_ehs_proposer(collector, num_reqs=2, phase="prefill")

    with patch.object(ExtractHiddenStatesProposer, "propose", return_value="draft_ids") as upstream:
        result = proposer.propose(
            1,
            sampled_token_ids=MagicMock(),
            target_hidden_states=[],
            common_attn_metadata=MagicMock(),
        )

    assert result == "draft_ids"
    upstream.assert_called_once()
    collector.start.assert_called_once_with("draft", "prefill", 2)
    collector.finish.assert_called_once_with(collector.start.return_value)


def test_extract_hidden_states_propose_passthrough_without_runner():
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer

    proposer = _make_ehs_proposer(collector=None)
    proposer.runner = None

    with patch.object(ExtractHiddenStatesProposer, "propose", return_value="draft_ids") as upstream:
        result = proposer.propose(1, sampled_token_ids=MagicMock(), target_hidden_states=[], common_attn_metadata=MagicMock())

    assert result == "draft_ids"
    upstream.assert_called_once()


# --------------------------------------------------------------------- #
# 埋点覆盖面：业务路径全部接入，dummy/capture 路径全部排除
# --------------------------------------------------------------------- #


def test_business_propose_paths_are_instrumented():
    from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
        AscendExtractHiddenStatesProposer,
    )
    from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
    from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

    assert "run_timed_draft_forward" in inspect.getsource(AscendSpecDecodeBaseProposer._propose)
    assert "run_timed_draft_forward" in inspect.getsource(AscendStep3p5MTPProposer._propose)
    # Special proposers outside the AscendSpecDecodeBaseProposer tree.
    assert "run_timed_draft_forward" in inspect.getsource(AscendMedusaProposer.propose)
    assert "run_timed_draft_forward" in inspect.getsource(AscendExtractHiddenStatesProposer.propose)


def test_medusa_construction_receives_runner():
    # The timing wrapper reaches the collector through runner; the medusa
    # construction site must forward it (it used to be the only proposer
    # built without one).
    from vllm_ascend.spec_decode import get_spec_decode_method

    assert "AscendMedusaProposer(vllm_config, device, runner)" in inspect.getsource(get_spec_decode_method)


def test_dummy_paths_are_not_instrumented():
    from vllm_ascend.spec_decode.dflash_proposer import AscendDflashProposer
    from vllm_ascend.spec_decode.dspark_proposer import AscendDSparkProposer
    from vllm_ascend.spec_decode.extract_hidden_states_proposer import (
        AscendExtractHiddenStatesProposer,
    )
    from vllm_ascend.spec_decode.medusa_proposer import AscendMedusaProposer
    from vllm_ascend.spec_decode.step3p5 import AscendStep3p5MTPProposer

    proposer_classes = (
        AscendSpecDecodeBaseProposer,
        AscendStep3p5MTPProposer,
        AscendDflashProposer,
        AscendDSparkProposer,
        AscendMedusaProposer,
        AscendExtractHiddenStatesProposer,
    )
    for cls in proposer_classes:
        assert "run_timed_draft_forward" not in inspect.getsource(cls.dummy_run), cls.__name__
    # The inner _run_merged_draft body is intentionally NOT instrumented:
    # ACL graph replay skips the Python body, so its self.model calls are
    # only the eager/capture rendering of the same draft phase that the
    # wrapper at the _runnable boundary already brackets.
    assert "run_timed_draft_forward" not in inspect.getsource(AscendSpecDecodeBaseProposer._run_merged_draft)
