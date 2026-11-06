# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Optional

from BeingH.npu_capture_replay import NPUCaptureProcessUnhealthyError

from .service import BaseInferenceClient, BaseInferenceServer


class BeingHInferenceServer(BaseInferenceServer):
    """ZeroMQ server for Being-H action inference and graph health."""

    def __init__(self, policy: Any, host: str = "*", port: int = 5555, api_token: Optional[str] = None):
        super().__init__(host, port, api_token)
        self.policy = policy
        self._npu_capture_unhealthy_reason: Optional[str] = None
        self.register_endpoint("get_action", self._get_action)
        self.register_endpoint("get_inference_stats", self._get_inference_stats, requires_input=False)

    def _get_inference_stats(self) -> Dict[str, Any]:
        """Expose read-only graph health/counters for acceptance tests."""
        model = getattr(self.policy, "model", None)
        suffix_runner = getattr(model, "_npu_action_suffix_graph_runner", None)
        return {
            "capture_unhealthy_reason": self._npu_capture_unhealthy_reason,
            "npu_graph_cache": suffix_runner.stats() if suffix_runner is not None else None,
        }

    def _get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Fail the worker closed after capture/replay corrupts NPU state."""
        if self._npu_capture_unhealthy_reason is not None:
            raise NPUCaptureProcessUnhealthyError(
                f"NPU capture/replay process is unhealthy; restart the server worker: "
                f"{self._npu_capture_unhealthy_reason}"
            )
        try:
            return self.policy.get_action(observations)
        except NPUCaptureProcessUnhealthyError as error:
            setattr(self, "_npu_capture_unhealthy_reason", str(error))
            setattr(self, "running", False)
            raise


class BeingHInferenceClient(BaseInferenceClient):
    """Client for the production Being-H inference endpoints."""

    def __init__(self, host: str = "localhost", port: int = 5555, api_token: Optional[str] = None):
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)

    def get_inference_stats(self) -> Dict[str, Any]:
        return self.call_endpoint("get_inference_stats", requires_input=False)
