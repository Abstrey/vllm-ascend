from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def dump_op_inputs(
    dump_dir: str | os.PathLike[str],
    *,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    inputs: dict[str, Any],
    rank: int,
    pid: int,
    step: int = 0,
) -> Path:
    payload = {
        "mode": mode,
        "layer_name": layer_name,
        "op_name": op_name,
        "step": step,
        "inputs": {
            name: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for name, value in inputs.items()
        },
    }
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"sfa_{mode}_inputs_layer{layer_id}_step{step}_rank{rank}_pid{pid}.pt"
    temporary = output.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    return output


def dump_op_output(
    dump_dir: str | os.PathLike[str],
    *,
    mode: str,
    layer_name: str,
    layer_id: int,
    op_name: str,
    output: torch.Tensor,
    rank: int,
    pid: int,
    step: int = 0,
) -> Path:
    payload = {
        "mode": mode,
        "layer_name": layer_name,
        "op_name": op_name,
        "step": step,
        "output": output.detach().cpu(),
    }
    output_dir = Path(dump_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sfa_{mode}_output_layer{layer_id}_step{step}_rank{rank}_pid{pid}.pt"
    temporary = output_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output_path)
    return output_path
