"""Custom Prefect cache policies for the video pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from prefect.cache_policies import Inputs
from prefect.context import TaskRunContext


@dataclass
class FileValidatedInputs(Inputs):
    """
    Like INPUTS, but invalidates when the ``dst`` file is missing.

    Tasks that produce a file at a ``dst`` Path will only serve cached
    results when that file still exists on disk.  If the work directory
    was cleaned between runs the task re-executes automatically.
    """

    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: Any,
    ) -> Optional[str]:
        dst = inputs.get("dst")
        if dst is not None and not Path(dst).exists():
            return None
        return super().compute_key(task_ctx, inputs, flow_parameters, **kwargs)


FILE_VALIDATED_INPUTS = FileValidatedInputs()
