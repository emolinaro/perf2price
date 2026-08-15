#!/usr/bin/env python3
"""Persist, validate, and classify perf2price benchmark run state."""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import errno
import io
import json
import math
import os
import pathlib
import re
import socket
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Literal

import perf2price_fit


class RunStateError(ValueError):
    """A saved run cannot be read or safely resumed."""


@dataclass(frozen=True)
class PlanRow:
    name: str
    isl: int
    osl: int
    prefix_tokens: int


@dataclass(frozen=True)
class PointStatus:
    state: Literal["complete", "retryable", "not_started"]
    reason: str


@dataclass(frozen=True)
class ExpectedPoint:
    row: PlanRow
    concurrency: int
    duration: float
    http_connection_limit: int


PROFILE_NAME_RE = re.compile(r"[A-Za-z0-9._-]+\Z")
CONFIG_SCHEMA_VERSION = 2

STRING_FIELDS = (
    "url",
    "endpoint",
    "models_endpoint",
    "model",
    "tokenizer",
    "concurrency_list",
    "grace_period",
)
OPTIONAL_STRING_FIELDS = (
    "tokenizer_revision",
    "plan_file",
    "aiperf_version",
)
BOOL_FIELDS = (
    "tokenizer_trust_remote_code",
    "apply_chat_template",
    "legacy_max_tokens",
    "run_cache_tests",
    "skip_probe",
)
NUMBER_FIELDS = (
    "duration_seconds",
    "ttft_p99_ms",
    "itl_p99_ms",
)
INT_FIELDS = (
    "max_context",
    "random_seed",
    "num_dataset_entries",
)
REQUIRED_CONFIG_FIELDS = (
    *STRING_FIELDS,
    *OPTIONAL_STRING_FIELDS,
    *BOOL_FIELDS,
    *NUMBER_FIELDS,
    *INT_FIELDS,
    "resource_hour_cost_usd",
    "extra_aiperf_args",
)
EMITTED_FIELDS = (
    "config_schema_version",
    "url",
    "endpoint",
    "models_endpoint",
    "model",
    "tokenizer",
    "tokenizer_revision",
    "concurrency_list",
    "duration_seconds",
    "grace_period",
    "ttft_p99_ms",
    "itl_p99_ms",
    "resource_hour_cost_usd",
    "random_seed",
    "num_dataset_entries",
    "tokenizer_trust_remote_code",
    "apply_chat_template",
    "legacy_max_tokens",
    "run_cache_tests",
    "skip_probe",
    "max_context",
    "aiperf_version",
    "api_key_required",
    "http_connection_limit",
)


def validate_profile_name(name: str) -> None:
    if name in {".", ".."} or not PROFILE_NAME_RE.fullmatch(name):
        raise RunStateError(
            f"invalid profile name {name!r}; use only A-Z, a-z, 0-9, '.', '_', and '-'"
        )


def _is_number(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_concurrencies(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise RunStateError("saved concurrency list is empty")
    values = []
    seen = set()
    for item in items:
        if not re.fullmatch(r"[0-9]+", item) or int(item) <= 0:
            raise RunStateError(f"invalid saved concurrency: {item!r}")
        parsed = int(item)
        if parsed not in seen:
            values.append(parsed)
            seen.add(parsed)
    return values


def _validated_config(config) -> dict:
    if not isinstance(config, dict):
        raise RunStateError("run_config.json must contain a JSON object")

    normalized = dict(config)
    normalized.setdefault("config_schema_version", 0)
    if not _is_int(normalized["config_schema_version"]):
        raise RunStateError(
            "run_config.json field 'config_schema_version' must be an integer"
        )
    if normalized["config_schema_version"] not in {0, 1, CONFIG_SCHEMA_VERSION}:
        raise RunStateError(
            "unsupported run configuration schema version: "
            f"{normalized['config_schema_version']}"
        )

    missing = sorted(set(REQUIRED_CONFIG_FIELDS) - set(normalized))
    if (
        normalized["config_schema_version"] >= 1
        and "api_key_required" not in normalized
    ):
        missing.append("api_key_required")
    if (
        normalized["config_schema_version"] >= CONFIG_SCHEMA_VERSION
        and "http_connection_limit" not in normalized
    ):
        missing.append("http_connection_limit")
    if missing:
        raise RunStateError(
            "run_config.json is missing required field(s): " + ", ".join(missing)
        )

    for field in STRING_FIELDS:
        value = normalized[field]
        if not isinstance(value, str) or not value:
            raise RunStateError(f"run_config.json field {field!r} must be a string")
    for field in OPTIONAL_STRING_FIELDS:
        value = normalized[field]
        if value is not None and not isinstance(value, str):
            raise RunStateError(
                f"run_config.json field {field!r} must be a string or null"
            )
    for field in BOOL_FIELDS:
        if not isinstance(normalized[field], bool):
            raise RunStateError(f"run_config.json field {field!r} must be Boolean")
    for field in NUMBER_FIELDS:
        if not _is_number(normalized[field]):
            raise RunStateError(f"run_config.json field {field!r} must be numeric")
    for field in INT_FIELDS:
        if not _is_int(normalized[field]):
            raise RunStateError(f"run_config.json field {field!r} must be an integer")

    if normalized["duration_seconds"] <= 0:
        raise RunStateError("run_config.json field 'duration_seconds' must be > 0")
    if normalized["ttft_p99_ms"] < 0 or normalized["itl_p99_ms"] < 0:
        raise RunStateError("saved SLO values must be >= 0")
    if normalized["max_context"] < 0 or normalized["random_seed"] < 0:
        raise RunStateError("saved max context and random seed must be >= 0")
    if normalized["num_dataset_entries"] <= 0:
        raise RunStateError("saved dataset entry count must be > 0")
    try:
        if (
            normalized["grace_period"] != "inf"
            and float(normalized["grace_period"]) < 0
        ):
            raise ValueError
    except ValueError as exc:
        raise RunStateError(
            "run_config.json field 'grace_period' must be >= 0 or 'inf'"
        ) from exc

    hour_cost = normalized["resource_hour_cost_usd"]
    if hour_cost is not None and (not _is_number(hour_cost) or hour_cost <= 0):
        raise RunStateError(
            "run_config.json field 'resource_hour_cost_usd' must be positive or null"
        )

    extra_args = normalized["extra_aiperf_args"]
    if not isinstance(extra_args, list) or not all(
        isinstance(arg, str) for arg in extra_args
    ):
        raise RunStateError(
            "run_config.json field 'extra_aiperf_args' must be a string list"
        )

    normalized.setdefault("api_key_required", None)
    api_key_required = normalized["api_key_required"]
    if api_key_required is not None and not isinstance(api_key_required, bool):
        raise RunStateError(
            "run_config.json field 'api_key_required' must be Boolean or null"
        )
    concurrencies = parse_concurrencies(normalized["concurrency_list"])
    if normalized["config_schema_version"] < CONFIG_SCHEMA_VERSION:
        normalized["http_connection_limit"] = max(concurrencies) + 64
    if (
        not _is_int(normalized["http_connection_limit"])
        or normalized["http_connection_limit"] <= 0
    ):
        raise RunStateError(
            "run_config.json field 'http_connection_limit' must be a positive integer"
        )
    return normalized


def load_run_config(root: pathlib.Path) -> dict:
    path = pathlib.Path(root) / "run_config.json"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RunStateError(f"missing saved run configuration: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStateError(f"cannot read saved run configuration {path}: {exc}") from exc
    return _validated_config(config)


def load_plan(path: pathlib.Path) -> list[PlanRow]:
    path = pathlib.Path(path)
    try:
        stream = path.open(newline="", encoding="utf-8")
    except OSError as exc:
        raise RunStateError(f"cannot read benchmark plan {path}: {exc}") from exc

    rows = []
    seen = set()
    with stream:
        for line_number, raw in enumerate(csv.reader(stream), start=1):
            values = [value.strip() for value in raw]
            if not values or not any(values):
                continue
            name = values[0]
            if name == "name" or name.startswith("#"):
                continue
            if len(values) != 4:
                raise RunStateError(
                    f"benchmark plan line {line_number} must contain exactly 4 columns"
                )
            validate_profile_name(name)
            if name in seen:
                raise RunStateError(f"duplicate profile name in benchmark plan: {name!r}")
            parsed = []
            for field, value in zip(("isl", "osl", "prefix_tokens"), values[1:]):
                if not re.fullmatch(r"[0-9]+", value):
                    raise RunStateError(
                        f"invalid {field} for {name!r} on line {line_number}: {value!r}"
                    )
                parsed.append(int(value))
            rows.append(PlanRow(name, *parsed))
            seen.add(name)
    if not rows:
        raise RunStateError(f"benchmark plan contains no data rows: {path}")
    return rows


def expected_points(root: pathlib.Path) -> list[ExpectedPoint]:
    root = pathlib.Path(root)
    config = load_run_config(root)
    rows = load_plan(root / "benchmark_plan.csv")
    points = []
    for row in rows:
        if row.prefix_tokens > 0 and not config["run_cache_tests"]:
            continue
        if (
            config["max_context"] > 0
            and row.isl + row.prefix_tokens + row.osl > config["max_context"]
        ):
            continue
        for concurrency in parse_concurrencies(config["concurrency_list"]):
            points.append(
                ExpectedPoint(
                    row,
                    concurrency,
                    float(config["duration_seconds"]),
                    config["http_connection_limit"],
                )
            )
    return points


def atomic_write_json(path: pathlib.Path, payload: dict) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_run_config(path: pathlib.Path, config: dict) -> None:
    atomic_write_json(path, _validated_config(config))


def write_run_context(
    path: pathlib.Path,
    row: PlanRow,
    concurrency: int,
    duration: float,
    http_connection_limit: int,
) -> None:
    validate_profile_name(row.name)
    if concurrency <= 0:
        raise RunStateError("concurrency must be positive")
    if not math.isfinite(duration) or duration <= 0:
        raise RunStateError("requested duration must be positive")
    if min(row.isl, row.osl, row.prefix_tokens) < 0:
        raise RunStateError("plan token counts must be non-negative")
    if not _is_int(http_connection_limit) or http_connection_limit <= 0:
        raise RunStateError("HTTP connection limit must be a positive integer")
    atomic_write_json(
        path,
        {
            "profile": row.name,
            "isl": row.isl,
            "osl": row.osl,
            "prefix_tokens": row.prefix_tokens,
            "concurrency": concurrency,
            "requested_duration_seconds": duration,
            "http_connection_limit": http_connection_limit,
        },
    )


def _load_context(path: pathlib.Path) -> dict:
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RunStateError(f"cannot read run context {path}: {exc}") from exc
    if not isinstance(context, dict):
        raise RunStateError(f"run context must contain a JSON object: {path}")
    return context


def finish_run_context(path: pathlib.Path, exit_code: int) -> None:
    context = _load_context(pathlib.Path(path))
    context["aiperf_exit_code"] = exit_code
    context["aiperf_ok"] = exit_code == 0
    atomic_write_json(path, context)


def classify_point(
    root: pathlib.Path,
    row: PlanRow,
    concurrency: int,
    duration: float,
    http_connection_limit: int,
) -> PointStatus:
    point = pathlib.Path(root) / "runs" / row.name / f"c{concurrency}"
    if not os.path.lexists(point):
        return PointStatus("not_started", "directory_missing")
    if point.is_symlink() or not point.is_dir():
        return PointStatus("retryable", "directory_invalid")
    context_path = point / "run_context.json"
    if not context_path.exists():
        return PointStatus("retryable", "context_missing")
    try:
        context = _load_context(context_path)
    except RunStateError:
        return PointStatus("retryable", "context_invalid")

    expected = {
        "profile": row.name,
        "isl": row.isl,
        "osl": row.osl,
        "prefix_tokens": row.prefix_tokens,
        "concurrency": concurrency,
        "http_connection_limit": http_connection_limit,
    }
    for field, value in expected.items():
        saved = context.get(field)
        matches = (
            isinstance(saved, str) and saved == value
            if field == "profile"
            else _is_int(saved) and saved == value
        )
        if not matches:
            return PointStatus("retryable", f"metadata_mismatch_{field}")
    saved_duration = context.get("requested_duration_seconds")
    if not _is_number(saved_duration) or not math.isclose(
        float(saved_duration), float(duration), rel_tol=0, abs_tol=1e-9
    ):
        return PointStatus(
            "retryable", "metadata_mismatch_requested_duration_seconds"
        )
    saved_exit_code = context.get("aiperf_exit_code")
    if (
        context.get("aiperf_ok") is not True
        or not _is_int(saved_exit_code)
        or saved_exit_code != 0
    ):
        return PointStatus("retryable", "aiperf_not_successful")

    with contextlib.redirect_stderr(io.StringIO()):
        _, summary = perf2price_fit.load_summary(point)
    if summary is None:
        return PointStatus("retryable", "summary_missing_or_invalid")
    return PointStatus("complete", "success")


def _unique_destination(path: pathlib.Path) -> pathlib.Path:
    if not os.path.lexists(path):
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.name}-{counter}")
        if not os.path.lexists(candidate):
            return candidate
        counter += 1


def _backup_directory(root: pathlib.Path, *parts: str) -> pathlib.Path:
    root_path = pathlib.Path(root)
    try:
        resolved_root = root_path.resolve(strict=True)
        root_mode = resolved_root.stat().st_mode
    except OSError as exc:
        raise RunStateError(f"cannot resolve run directory {root}: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise RunStateError(f"run directory is not a real directory: {root}")

    current = root_path
    for part in ("resume_backups", *parts):
        if part in {"", ".", ".."} or pathlib.PurePath(part).name != part:
            raise RunStateError(f"invalid backup path component: {part!r}")
        current = current / part
        try:
            current.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise RunStateError(
                f"cannot create backup directory {current}: {exc}"
            ) from exc
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise RunStateError(
                f"cannot inspect backup directory {current}: {exc}"
            ) from exc
        if not stat.S_ISDIR(mode):
            raise RunStateError(
                f"backup path component is not a real directory: {current}"
            )
        try:
            current.resolve(strict=True).relative_to(resolved_root)
        except (OSError, ValueError) as exc:
            raise RunStateError(
                f"backup directory escapes run directory: {current}"
            ) from exc
    return current


def archive_point(
    root: pathlib.Path,
    profile: str,
    concurrency: int,
    reason: str,
    *,
    now_ns: int | None = None,
    pid: int | None = None,
) -> pathlib.Path | None:
    validate_profile_name(profile)
    if not _is_int(concurrency) or concurrency <= 0:
        raise RunStateError("concurrency must be a positive integer")
    root = pathlib.Path(root)
    source = root / "runs" / profile / f"c{concurrency}"
    if not os.path.lexists(source):
        return None
    stamp = time.time_ns() if now_ns is None else now_ns
    owner = os.getpid() if pid is None else pid
    safe_reason = re.sub(r"[^A-Za-z0-9._-]", "_", reason).strip("_")
    safe_reason = safe_reason or "unknown"
    target = _backup_directory(root, profile, f"c{concurrency}") / (
        f"attempt-{stamp}-{owner}-{safe_reason}"
    )
    target = _unique_destination(target)
    try:
        os.replace(source, target)
    except OSError as exc:
        raise RunStateError(
            f"cannot preserve retryable point {source} at {target}: {exc}"
        ) from exc
    return target


def _archive_unexpected_entry(
    root: pathlib.Path,
    source: pathlib.Path,
    label: str,
    *,
    now_ns: int,
    pid: int,
) -> pathlib.Path:
    safe_label = re.sub(r"[^A-Za-z0-9._-]", "_", label).strip("_")
    safe_label = safe_label or "entry"
    target = _backup_directory(root, "unexpected") / (
        f"attempt-{now_ns}-{pid}-{safe_label}"
    )
    target = _unique_destination(target)
    try:
        os.replace(source, target)
    except OSError as exc:
        raise RunStateError(
            f"cannot preserve unexpected run entry {source} at {target}: {exc}"
        ) from exc
    return target


def archive_unexpected_points(
    root: pathlib.Path,
    *,
    now_ns: int | None = None,
    pid: int | None = None,
) -> list[pathlib.Path]:
    root = pathlib.Path(root)
    runs = root / "runs"
    if not os.path.lexists(runs):
        return []
    stamp = time.time_ns() if now_ns is None else now_ns
    owner = os.getpid() if pid is None else pid
    points = expected_points(root)
    expected = {}
    for point in points:
        expected.setdefault(point.row.name, set()).add(f"c{point.concurrency}")
    if runs.is_symlink() or not runs.is_dir():
        return [
            _archive_unexpected_entry(
                root, runs, "runs", now_ns=stamp, pid=owner
            )
        ]

    archived = []
    for profile_entry in sorted(runs.iterdir(), key=lambda path: path.name):
        expected_children = expected.get(profile_entry.name)
        if (
            expected_children is None
            or profile_entry.is_symlink()
            or not profile_entry.is_dir()
        ):
            archived.append(
                _archive_unexpected_entry(
                    root,
                    profile_entry,
                    profile_entry.name,
                    now_ns=stamp,
                    pid=owner,
                )
            )
            continue
        for child in sorted(profile_entry.iterdir(), key=lambda path: path.name):
            if (
                child.name not in expected_children
                or child.is_symlink()
                or not child.is_dir()
            ):
                archived.append(
                    _archive_unexpected_entry(
                        root,
                        child,
                        f"{profile_entry.name}-{child.name}",
                        now_ns=stamp,
                        pid=owner,
                    )
                )
    return archived


def process_is_alive(pid: int) -> bool:
    if not _is_int(pid) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_owner(lock: pathlib.Path) -> dict | None:
    try:
        owner = json.loads((lock / "owner.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(owner, dict):
        return None
    if not _is_int(owner.get("pid")) or owner["pid"] <= 0:
        return None
    if not isinstance(owner.get("hostname"), str) or not owner["hostname"]:
        return None
    if not isinstance(owner.get("acquired_at"), str):
        return None
    active_child_pid = owner.get("active_child_pid")
    if active_child_pid is not None and (
        not _is_int(active_child_pid) or active_child_pid <= 0
    ):
        return None
    return owner


def _discard_candidate(candidate: pathlib.Path) -> None:
    if not candidate.exists():
        return
    owner = candidate / "owner.json"
    if owner.exists():
        owner.unlink()
    candidate.rmdir()


def _install_lock(lock: pathlib.Path, candidate: pathlib.Path) -> bool:
    if os.path.lexists(lock):
        return False
    try:
        os.rename(candidate, lock)
    except OSError as exc:
        if exc.errno in {errno.EEXIST, errno.ENOTEMPTY}:
            return False
        raise
    return True


def acquire_lock(
    root: pathlib.Path,
    owner_pid: int,
    *,
    hostname: str | None = None,
    is_alive: Callable[[int], bool] = process_is_alive,
    now_ns: int | None = None,
) -> pathlib.Path:
    if not _is_int(owner_pid) or owner_pid <= 0:
        raise RunStateError("lock owner PID must be a positive integer")
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    host = hostname or socket.gethostname()
    stamp = time.time_ns() if now_ns is None else now_ns
    lock = root / ".perf2price.lock"

    for attempt in range(5):
        candidate = root / f".perf2price.lock.candidate-{owner_pid}-{stamp}-{attempt}"
        try:
            candidate.mkdir()
            atomic_write_json(
                candidate / "owner.json",
                {
                    "pid": owner_pid,
                    "hostname": host,
                    "acquired_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
            )
            if _install_lock(lock, candidate):
                return lock
        except OSError as exc:
            raise RunStateError(f"cannot acquire run lock {lock}: {exc}") from exc
        finally:
            try:
                _discard_candidate(candidate)
            except OSError:
                pass

        existing = _lock_owner(lock)
        if existing is not None:
            if existing["hostname"] != host:
                raise RunStateError(
                    f"run is locked by PID {existing['pid']} on host "
                    f"{existing['hostname']}"
                )
            active_child_pid = existing.get("active_child_pid")
            if is_alive(existing["pid"]):
                raise RunStateError(
                    f"run is locked by live PID {existing['pid']} on host {host}"
                )
            if active_child_pid is not None and is_alive(active_child_pid):
                raise RunStateError(
                    f"run is locked by active AIPerf PID {active_child_pid} on host "
                    f"{host}"
                )
            stale_owner = existing["pid"]
        else:
            stale_owner = "unknown"

        backup = _unique_destination(
            _backup_directory(root, "locks")
            / f"stale-{stamp}-{stale_owner}"
        )
        try:
            os.rename(lock, backup)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RunStateError(
                f"cannot preserve stale run lock {lock} at {backup}: {exc}"
            ) from exc
    raise RunStateError(f"could not acquire run lock after repeated races: {lock}")


def register_lock_child(
    root: pathlib.Path,
    owner_pid: int,
    child_pid: int,
    *,
    hostname: str | None = None,
) -> None:
    if not _is_int(child_pid) or child_pid <= 0:
        raise RunStateError("lock child PID must be a positive integer")
    root = pathlib.Path(root)
    lock = root / ".perf2price.lock"
    host = hostname or socket.gethostname()
    owner = _lock_owner(lock)
    if owner is None:
        raise RunStateError(f"cannot register child on malformed run lock: {lock}")
    if owner["pid"] != owner_pid or owner["hostname"] != host:
        raise RunStateError(
            f"run lock is owned by PID {owner['pid']} on host {owner['hostname']}"
        )
    owner["active_child_pid"] = child_pid
    atomic_write_json(lock / "owner.json", owner)


def run_locked(root: pathlib.Path, owner_pid: int, command: list[str]) -> None:
    if not command:
        raise RunStateError("locked command must not be empty")
    register_lock_child(root, owner_pid, os.getpid())
    try:
        os.execvp(command[0], command)
    except OSError as exc:
        raise RunStateError(
            f"cannot execute AIPerf command {command[0]!r}: {exc}"
        ) from exc


def release_lock(
    root: pathlib.Path,
    owner_pid: int,
    *,
    hostname: str | None = None,
    is_alive: Callable[[int], bool] = process_is_alive,
) -> None:
    root = pathlib.Path(root)
    lock = root / ".perf2price.lock"
    if not lock.exists():
        return
    host = hostname or socket.gethostname()
    owner = _lock_owner(lock)
    if owner is None:
        raise RunStateError(f"cannot release malformed run lock: {lock}")
    if owner["pid"] != owner_pid or owner["hostname"] != host:
        raise RunStateError(
            f"run lock is owned by PID {owner['pid']} on host {owner['hostname']}"
        )
    active_child_pid = owner.get("active_child_pid")
    if active_child_pid is not None and is_alive(active_child_pid):
        raise RunStateError(
            f"run lock still protects active AIPerf PID {active_child_pid}"
        )
    try:
        (lock / "owner.json").unlink()
        lock.rmdir()
    except OSError as exc:
        raise RunStateError(f"cannot release run lock {lock}: {exc}") from exc


def _flag(value: str, name: str) -> bool:
    if value not in {"0", "1"}:
        raise RunStateError(f"{name} must be 0 or 1")
    return value == "1"


def _number(value: str, name: str):
    try:
        number = float(value)
    except ValueError as exc:
        raise RunStateError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise RunStateError(f"{name} must be finite")
    return int(number) if number == int(number) else number


def _integer(value: str, name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise RunStateError(f"{name} must be an integer") from exc


def _emit_config(root: pathlib.Path) -> None:
    config = load_run_config(root)
    output = sys.stdout.buffer
    for key in EMITTED_FIELDS:
        value = config[key]
        if isinstance(value, bool):
            text = "1" if value else "0"
        elif value is None:
            text = ""
        else:
            text = str(value)
        output.write(key.encode() + b"\0" + text.encode() + b"\0")
    for arg in config["extra_aiperf_args"]:
        output.write(b"extra_aiperf_arg\0" + arg.encode() + b"\0")


def _emit_points(root: pathlib.Path) -> None:
    for point in expected_points(root):
        print(
            "\t".join(
                (
                    point.row.name,
                    str(point.row.isl),
                    str(point.row.osl),
                    str(point.row.prefix_tokens),
                    str(point.concurrency),
                    str(point.duration),
                    str(point.http_connection_limit),
                )
            )
        )


def _write_config_from_args(values) -> None:
    try:
        extra_args = json.loads(values.extra_args_json)
    except json.JSONDecodeError as exc:
        raise RunStateError(f"extra AIPerf arguments are not valid JSON: {exc}") from exc
    config = {
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "url": values.url,
        "endpoint": values.endpoint,
        "models_endpoint": values.models_endpoint,
        "model": values.model,
        "tokenizer": values.tokenizer,
        "tokenizer_revision": values.tokenizer_revision or None,
        "concurrency_list": values.concurrency_list,
        "duration_seconds": _number(values.duration, "duration"),
        "grace_period": values.grace,
        "ttft_p99_ms": _number(values.ttft, "TTFT SLO"),
        "itl_p99_ms": _number(values.itl, "ITL SLO"),
        "resource_hour_cost_usd": (
            _number(values.hour_cost, "resource hour cost")
            if values.hour_cost
            else None
        ),
        "random_seed": _integer(values.random_seed, "random seed"),
        "num_dataset_entries": _integer(values.num_entries, "dataset entries"),
        "tokenizer_trust_remote_code": _flag(values.trust_remote, "trust_remote"),
        "apply_chat_template": _flag(values.apply_chat, "apply_chat"),
        "legacy_max_tokens": _flag(values.legacy, "legacy"),
        "run_cache_tests": _flag(values.cache_tests, "cache_tests"),
        "skip_probe": _flag(values.skip_probe, "skip_probe"),
        "max_context": _integer(values.max_context, "max context"),
        "plan_file": values.plan_file or None,
        "aiperf_version": values.aiperf_version or None,
        "api_key_required": _flag(values.api_key_required, "api_key_required"),
        "http_connection_limit": _integer(
            values.http_connection_limit, "HTTP connection limit"
        ),
        "extra_aiperf_args": extra_args,
    }
    write_run_config(pathlib.Path(values.path), config)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    emit = commands.add_parser("emit-config")
    emit.add_argument("root")

    emit_points = commands.add_parser("emit-points")
    emit_points.add_argument("root")

    validate = commands.add_parser("validate-plan")
    validate.add_argument("path")

    classify = commands.add_parser("classify-point")
    classify.add_argument("root")
    classify.add_argument("name")
    classify.add_argument("isl", type=int)
    classify.add_argument("osl", type=int)
    classify.add_argument("prefix", type=int)
    classify.add_argument("concurrency", type=int)
    classify.add_argument("duration", type=float)
    classify.add_argument("http_connection_limit", type=int)

    write_context = commands.add_parser("write-context")
    write_context.add_argument("path")
    write_context.add_argument("name")
    write_context.add_argument("isl", type=int)
    write_context.add_argument("osl", type=int)
    write_context.add_argument("prefix", type=int)
    write_context.add_argument("concurrency", type=int)
    write_context.add_argument("duration", type=float)
    write_context.add_argument("http_connection_limit", type=int)

    finish_context = commands.add_parser("finish-context")
    finish_context.add_argument("path")
    finish_context.add_argument("exit_code", type=int)

    archive = commands.add_parser("archive-point")
    archive.add_argument("root")
    archive.add_argument("profile")
    archive.add_argument("concurrency", type=int)
    archive.add_argument("reason")

    archive_unexpected = commands.add_parser("archive-unexpected")
    archive_unexpected.add_argument("root")

    acquire = commands.add_parser("acquire-lock")
    acquire.add_argument("root")
    acquire.add_argument("owner_pid", type=int)

    release = commands.add_parser("release-lock")
    release.add_argument("root")
    release.add_argument("owner_pid", type=int)

    locked = commands.add_parser("run-locked")
    locked.add_argument("root")
    locked.add_argument("owner_pid", type=int)
    locked.add_argument("command_args", nargs=argparse.REMAINDER)

    write_config = commands.add_parser("write-config")
    config_args = (
        "path",
        "url",
        "endpoint",
        "models_endpoint",
        "model",
        "tokenizer",
        "tokenizer_revision",
        "concurrency_list",
        "duration",
        "grace",
        "ttft",
        "itl",
        "hour_cost",
        "random_seed",
        "num_entries",
        "trust_remote",
        "apply_chat",
        "legacy",
        "cache_tests",
        "skip_probe",
        "max_context",
        "plan_file",
        "aiperf_version",
        "api_key_required",
        "http_connection_limit",
        "extra_args_json",
    )
    for name in config_args:
        write_config.add_argument(name)
    return parser


def main(argv=None) -> int:
    values = _parser().parse_args(argv)
    try:
        if values.command == "emit-config":
            _emit_config(pathlib.Path(values.root))
        elif values.command == "emit-points":
            _emit_points(pathlib.Path(values.root))
        elif values.command == "validate-plan":
            load_plan(pathlib.Path(values.path))
        elif values.command == "classify-point":
            row = PlanRow(values.name, values.isl, values.osl, values.prefix)
            status = classify_point(
                pathlib.Path(values.root),
                row,
                values.concurrency,
                values.duration,
                values.http_connection_limit,
            )
            print(f"{status.state}\t{status.reason}")
        elif values.command == "write-context":
            row = PlanRow(values.name, values.isl, values.osl, values.prefix)
            write_run_context(
                pathlib.Path(values.path),
                row,
                values.concurrency,
                values.duration,
                values.http_connection_limit,
            )
        elif values.command == "finish-context":
            finish_run_context(pathlib.Path(values.path), values.exit_code)
        elif values.command == "archive-point":
            backup = archive_point(
                pathlib.Path(values.root),
                values.profile,
                values.concurrency,
                values.reason,
            )
            print(backup or "")
        elif values.command == "archive-unexpected":
            for backup in archive_unexpected_points(pathlib.Path(values.root)):
                print(backup)
        elif values.command == "acquire-lock":
            acquire_lock(pathlib.Path(values.root), values.owner_pid)
        elif values.command == "release-lock":
            release_lock(pathlib.Path(values.root), values.owner_pid)
        elif values.command == "run-locked":
            command = values.command_args
            if command[:1] == ["--"]:
                command = command[1:]
            run_locked(pathlib.Path(values.root), values.owner_pid, command)
        elif values.command == "write-config":
            _write_config_from_args(values)
        else:
            raise AssertionError(f"unhandled command: {values.command}")
    except RunStateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
