# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instance-local identity for V2 draft forward contexts."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_DRAFT_FORWARD_CONFIG_KEY = "_ascend_is_draft_model"


def create_draft_forward_config(vllm_config: VllmConfig) -> VllmConfig:
    """Tag only the speculator's config view; preserve shared model resources.

    V2 speculators pass their stored VllmConfig to set_forward_context, even
    when its model_config describes the target. Shallow copying avoids rerunning
    VllmConfig post-init and preserves attention/compilation registrations.
    """
    draft_config = copy(vllm_config)
    draft_config.additional_config = dict(getattr(vllm_config, "additional_config", None) or {})
    draft_config.additional_config[_DRAFT_FORWARD_CONFIG_KEY] = True
    return draft_config


def is_draft_forward_config(vllm_config: VllmConfig) -> bool:
    """Read the explicit marker instead of inferring a draft from its shape."""
    additional_config = getattr(vllm_config, "additional_config", None) or {}
    return additional_config.get(_DRAFT_FORWARD_CONFIG_KEY, False) is True
