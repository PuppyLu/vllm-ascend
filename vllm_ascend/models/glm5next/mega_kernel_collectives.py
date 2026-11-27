# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small fixed-size coordination primitives for the MegaKernel decode path."""

import torch
import torch.distributed as dist

_DIGEST_WORDS = 16


class MegaKernelCollectiveError(RuntimeError):
    """All participants must terminate the job after a failed MC2 phase."""


def enable_diagnostic_route_capture(session):
    """Prepare a local snapshot before the benchmark's all-rank preflight."""
    if session is None:
        raise ValueError("MC2 session does not support diagnostic route capture")
    if not hasattr(session, "capture_routes"):
        # The 1e591b193d runtime always snapshots these actual kernel outputs.
        if all(hasattr(session, name) for name in ("last_route_ids", "last_route_weights")):
            return None
        raise ValueError("MC2 session does not support diagnostic route capture")
    previous = session.capture_routes
    session.capture_routes = True
    return previous


def agree_or_abort(local_error, device, group, phase):
    """Complete a rank-consistent phase decision before downstream HCCL work."""
    failed = torch.tensor([int(local_error is not None)], dtype=torch.int32, device=device)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=group)
    if bool(failed.item()):
        raise MegaKernelCollectiveError(
            f"{phase} failed on at least one rank; abort job and restore state"
        ) from local_error


def agree_decode_step(accepted, digest, device, group, *, local_error=None):
    """Agree eligibility and the complete ordered-step digest with one reduction.

    Hash words are 16-bit values stored in int32, so both signs are safe for
    MAX. Every rank must call, even when locally ineligible or its digest is
    malformed. The one host read is necessary to make a common fallback
    decision before any layer can submit its HCCL work.
    """
    valid = (
        accepted is True
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    words = [int(digest[i : i + 4], 16) for i in range(0, 64, 4)] if valid else [0] * _DIGEST_WORDS
    # An unexpected local preparation failure must abort the whole group,
    # rather than turn into an ordinary unsupported-batch fallback on peers.
    status = 2 if local_error is not None else int(not valid)
    values = torch.tensor([status, *words, *(-word for word in words)], dtype=torch.int32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    reduced = values.cpu().tolist()
    if reduced[0] == 2:
        raise MegaKernelCollectiveError(
            "model step preparation failed on at least one rank; abort job"
        ) from local_error
    return reduced[0] == 0 and all(reduced[1 + i] == -reduced[1 + _DIGEST_WORDS + i] for i in range(_DIGEST_WORDS))
