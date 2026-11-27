# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small fixed-size coordination primitives and cold HCCL helpers for MegaKernel."""

import ctypes as C
import hashlib
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
        # Older 1e591b193d-era runtimes always capture routes when both attributes exist.
        if all(hasattr(session, name) for name in ("last_route_ids", "last_route_weights")):
            return None
        raise ValueError("MC2 session does not support diagnostic route capture")
    previous = session.capture_routes
    session.capture_routes = True
    return previous


def agree_or_abort(local_error, device, group, phase, *, scratch=None):
    """Complete a rank-consistent phase decision before downstream HCCL work."""
    flag = scratch if scratch is not None else torch.empty(1, dtype=torch.int32, device=device)
    flag[0] = int(local_error is not None)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if bool(flag.item()):
        raise MegaKernelCollectiveError(
            f"{phase} failed on at least one rank; abort job and restore state"
        ) from local_error


def agree_preflight(error, pure_decode, device, group, *, scratch=None, unsupported_exc=None):
    """Agree on accept (True), group fallback (False), or abort in one reduction."""
    is_unsupported = unsupported_exc is not None and isinstance(error, unsupported_exc)
    status = 2 if (error is not None and not is_unsupported) else int(error is not None or not pure_decode)
    decision = scratch if scratch is not None else torch.empty(1, dtype=torch.int32, device=device)
    decision[0] = status
    dist.all_reduce(decision, op=dist.ReduceOp.MAX, group=group)
    result = int(decision.item())
    if result == 2:
        cause = error if not is_unsupported else None
        raise MegaKernelCollectiveError("model adapter preflight failed on at least one rank; abort job") from cause
    return result == 0


def _to_int_list(tensor):
    return tensor.tolist() if tensor.device.type == "cpu" else tensor.cpu().tolist()


def agree_decode_step(accepted, digest, device, group, *, scratch=None, local_error=None):
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
    words = [int(digest[i : i + 4], 16) for i in range(0, 64, 4)] if (local_error is None and valid) else [0] * _DIGEST_WORDS
    # An unexpected local preparation failure must abort the whole group
    # instead of turning into an ordinary fallback on peers.
    status = 2 if local_error is not None else int(not valid)
    payload = [status, *words, *(-word for word in words)]
    if scratch is not None:
        values = scratch
        values.copy_(torch.tensor(payload, dtype=torch.int32))
    else:
        values = torch.tensor(payload, dtype=torch.int32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    reduced = _to_int_list(values)
    if reduced[0] >= 2:
        raise MegaKernelCollectiveError(
            "model step preparation failed on at least one rank; abort job"
        ) from local_error
    return reduced[0] == 0 and all(reduced[1 + i] == -reduced[1 + _DIGEST_WORDS + i] for i in range(_DIGEST_WORDS))


def agree_decode_step_contract(
    accepted,
    rows,
    digest,
    max_rows_per_source,
    max_total_rows,
    world,
    device,
    group,
    *,
    scratch=None,
    local_error=None,
):
    """Agree step eligibility, ordered SHA-256 digest, and equal row count in one int32 all_reduce."""
    valid_digest = (
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    valid_rows = (
        not isinstance(rows, bool)
        and isinstance(rows, int)
        and 1 <= rows <= max_rows_per_source
        and rows * world <= max_total_rows
        and rows < 2**31
    )
    if local_error is not None:
        status = 2
        row_val = 0
        words = [0] * _DIGEST_WORDS
    elif accepted is True and valid_digest and valid_rows:
        status = 0
        row_val = rows
        words = [int(digest[i : i + 4], 16) for i in range(0, 64, 4)]
    else:
        status = 1
        row_val = 0
        words = [0] * _DIGEST_WORDS
    payload = [status, row_val, -row_val, *words, *(-word for word in words)]
    if scratch is not None:
        values = scratch
        values.copy_(torch.tensor(payload, dtype=torch.int32))
    else:
        values = torch.tensor(payload, dtype=torch.int32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    reduced = _to_int_list(values)
    if reduced[0] >= 2:
        raise MegaKernelCollectiveError("step contract coordination failed on at least one rank; abort job") from local_error
    if reduced[0] != 0:
        return False, 0
    max_rows, neg_min_rows = reduced[1], reduced[2]
    digest_ok = all(reduced[3 + i] == -reduced[3 + _DIGEST_WORDS + i] for i in range(_DIGEST_WORDS))
    rows_ok = (
        max_rows == -neg_min_rows
        and 1 <= max_rows <= max_rows_per_source
        and max_rows * world <= max_total_rows
    )
    if not (digest_ok and rows_ok):
        return False, 0
    return True, int(max_rows)


def digest_agreement(raw, device, group, phase):
    """Startup agreement verifying byte-identical arbitrary config/manifest payloads."""
    digest = torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.int32, device=device)
    bounds = torch.cat((digest, -digest))
    dist.all_reduce(bounds, op=dist.ReduceOp.MAX, group=group)
    hi, neg_lo = bounds.chunk(2)
    if not torch.equal(hi, -neg_lo):
        raise MegaKernelCollectiveError(f"{phase} differs across ranks; abort job")


def admission_control(rounds, tp_size, framework):
    """Encode the agreed session mode in the existing MC2 control attribute."""
    if type(rounds) is not int or not 1 <= rounds <= 27:
        raise ValueError("invalid topology propagation rounds")
    if type(tp_size) is not int or not 1 <= tp_size <= 256 or type(framework) is not bool:
        raise ValueError("invalid framework admission control")
    return (1 << 30) | (tp_size << 8) | rounds if framework else rounds


class HcclDomainInspector:
    """Resolve and verify distinct HCCL communicator names and handles at startup."""

    def __init__(self):
        fwk = C.CDLL("libhccl_fwk.so", mode=C.RTLD_GLOBAL)
        lib = C.CDLL("libhccl.so", mode=C.RTLD_GLOBAL)
        self._get = fwk.HcclCommGetHandleWithName
        self._get.argtypes, self._get.restype = ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int)
        self._rank = lib.HcclGetRankId
        self._rank.argtypes, self._rank.restype = ([C.c_void_p, C.POINTER(C.c_uint32)], C.c_int)
        self._size = lib.HcclGetRankSize
        self._size.argtypes, self._size.restype = ([C.c_void_p, C.POINTER(C.c_uint32)], C.c_int)
        self._libs = (fwk, lib)

    def inspect(self, groups, device, rank, world):
        if len(groups) < 2:
            raise ValueError("HCCL domain inspection requires at least TP and EP groups")
        names, handles = [], []
        if len(groups) == 3:
            expected_pairs = (
                (rank, world),
                (dist.get_rank(groups[1]), dist.get_world_size(groups[1])),
                (dist.get_rank(groups[2]), dist.get_world_size(groups[2])),
            )
        else:
            expected_pairs = tuple((dist.get_rank(g), dist.get_world_size(g)) for g in groups)
        for group, expected in zip(groups, expected_pairs):
            name = group._get_backend(device).get_hccl_comm_name(dist.get_rank(group))
            if not isinstance(name, str) or not name or "\0" in name:
                raise ValueError("bad actual HCCL communicator name")
            handle = C.c_void_p()
            if self._get(name.encode(), C.byref(handle)) or not handle.value:
                raise ValueError("HCCL communicator cannot be resolved")
            actual_rank, actual_size = C.c_uint32(), C.c_uint32()
            if (
                self._rank(handle, C.byref(actual_rank))
                or self._size(handle, C.byref(actual_size))
                or (actual_rank.value, actual_size.value) != expected
            ):
                raise ValueError("HCCL communicator rank/size mismatch")
            names.append(name)
            handles.append(handle.value)
        if len(set(names)) != len(groups) or len(set(handles)) != len(groups):
            raise ValueError("actual HCCL communicator domains must be distinct")
        return tuple(names)
