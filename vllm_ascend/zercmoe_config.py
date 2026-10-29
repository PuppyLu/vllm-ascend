# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration shared by the A3 ZercMoE W8A8 integration."""

from contextlib import contextmanager
from contextvars import ContextVar

ZERCMOE_ALIGNMENT = 16
ZERCMOE_MIN_SHMEM_MB = 1024

_DRAFT_MODEL_LOADING = ContextVar("zercmoe_draft_model_loading", default=False)


@contextmanager
def zercmoe_draft_model_loading():
    """Mark draft construction and post-loading without leaking across contexts."""
    token = _DRAFT_MODEL_LOADING.set(True)
    try:
        yield
    finally:
        _DRAFT_MODEL_LOADING.reset(token)


def is_zercmoe_draft_model_loading() -> bool:
    return _DRAFT_MODEL_LOADING.get()


def load_zercmoe_draft_model(drafter, target) -> None:
    """Keep the draft role active for construction and weight post-loading."""
    with zercmoe_draft_model_loading():
        drafter.load_model(target)


def allow_zercmoe_shared_expert_overlap(*, enabled: int, single_weight: bool, is_a3: bool) -> bool:
    """Only a Zerc-only A3 target can bypass legacy fused-MC2 overlap restrictions."""
    return enabled == 1 and single_weight is True and is_a3


def use_shared_expert_overlap(*, configured: bool, has_shared_experts: bool) -> bool:
    """Keep a single-weight MTP draft on its legacy shared-expert path."""
    return bool(configured and has_shared_experts and not is_zercmoe_draft_model_loading())
