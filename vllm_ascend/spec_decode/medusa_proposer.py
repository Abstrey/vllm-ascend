import torch
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

from vllm_ascend.ascend_forward_context import set_ascend_forward_context
from vllm_ascend.core.forward_time_collector import run_timed_draft_forward


class AscendMedusaProposer(MedusaProposer):
    """
    Medusa proposer class for generating token sequences
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        # Kept before super().__init__ so propose() can reach the model
        # runner's forward-time collector (design doc §6.2).
        self.runner = runner
        super().__init__(vllm_config, device)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        with_prefill: bool = False,
        in_graph_capturing: bool = False,
        num_reqs: int = 0,
        num_tokens_across_dp: torch.Tensor | None = None,
        aclgraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor=None,
        dummy_compute_logits=lambda hidden_states: None,
        is_profile=False,
    ):
        hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with set_ascend_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_tokens,
            num_actual_tokens=0,
            in_profile_run=is_profile,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=aclgraph_runtime_mode,
            is_draft_model=True,
        ):
            self.model(hidden_states)
            dummy_compute_logits(hidden_states)

    def propose(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        sample_hidden_states: torch.Tensor,
    ):
        if sample_hidden_states.shape[0] == len(valid_sampled_token_ids):
            # The input to the target model does not include draft tokens.
            hidden_states = sample_hidden_states
        else:
            num_accepted_tokens = torch.tensor(
                [len(t) for t in valid_sampled_token_ids], device=self.device, dtype=torch.long
            )
            num_draft_tokens = torch.tensor(spec_decode_metadata.num_draft_tokens, device=self.device, dtype=torch.long)

            offsets = torch.cumsum(num_draft_tokens + 1, dim=0) - (num_draft_tokens + 1)
            indices = offsets + num_accepted_tokens - 1
            hidden_states = sample_hidden_states[indices]

        # The model calls live inside the upstream propose; the timing bracket
        # therefore also covers its tree-sampling overhead and cannot be
        # compared 1:1 with the eagle-family draft samples. (``upstream`` is
        # bound first: zero-arg super() does not work inside a lambda.)
        upstream = super()

        def run_upstream_propose():
            return upstream.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )

        return run_timed_draft_forward(
            self.runner,
            getattr(getattr(self.runner, "input_batch", None), "num_reqs", None),
            run_upstream_propose,
        )
