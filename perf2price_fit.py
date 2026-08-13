#!/usr/bin/env python3
"""Parse AIPerf runs, select capacity points, and fit the token-cost model.

Resource time for the regression is the requested profiling window, plus half
of any grace-period drain:

    T = requested_duration + 0.5 * max(0, benchmark_duration - requested)

Coefficients are estimated in rate space (equal weight per selected profile):

    1 ~= a * (I / T) + b * (O / T) + c * (C / T)
"""

from __future__ import annotations

import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

PLATEAU_REL = 0.05
OSL_RATIO_MIN = 0.5
BOOTSTRAP_SAMPLES = 400
ZERO_COEF_EPS = 1e-18
MAX_CONDITION_NUMBER = 10_000.0

SUMMARY_COLUMNS = [
    "profile",
    "isl",
    "osl",
    "prefix_tokens",
    "concurrency",
    "usage_ok",
    "cache_reported",
    "prompt_tokens",
    "noncached_input_tokens",
    "cached_input_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "request_count",
    "actual_osl",
    "osl_ratio",
    "requested_duration_seconds",
    "benchmark_duration_s",
    "resource_seconds",
    "request_throughput_rps",
    "api_prompt_tps",
    "api_completion_tps",
    "api_total_tps",
    "output_token_throughput_tps",
    "total_token_throughput_tps",
    "ttft_avg_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "itl_avg_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "request_latency_p99_ms",
    "error_request_count",
    "aiperf_exit_code",
    "aiperf_ok",
    "summary_json",
]

SELECTED_COLUMNS = SUMMARY_COLUMNS + [
    "selection_note",
    "plateau_reached",
    "fit_eligible",
    "fit_exclude_reason",
]


def finite(v):
    return v is not None and math.isfinite(float(v))


def _as_float(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def scalar_metric(data, name, stat="avg"):
    """Read a distribution statistic. Never falls back to a cumulative sum."""
    value = data.get(name)
    if value is None:
        return None
    raw = _as_float(value)
    if raw is not None:
        return raw
    if isinstance(value, dict):
        keys = (stat, "value", "mean") if stat == "avg" else (stat,)
        for key in keys:
            raw = _as_float(value.get(key))
            if raw is not None:
                return raw
    return None


def total_metric(data, name, prefer_sum=True):
    """Read a counter or scalar total."""
    value = data.get(name)
    if value is None:
        return None
    raw = _as_float(value)
    if raw is not None:
        return raw
    if isinstance(value, dict):
        keys = ("sum", "total", "value", "avg") if prefer_sum else ("value", "avg", "sum", "total")
        for key in keys:
            raw = _as_float(value.get(key))
            if raw is not None:
                return raw
    return None


def extract_error_count(data):
    """Number of errored requests in a run.

    AIPerf 0.12 exports an ``error_summary`` array instead of an
    ``error_request_count`` metric, so derive the count from there and fall
    back to the metric name in case future versions add it. The metric is
    ERROR_ONLY and absent on clean runs, which this treats as zero.
    """
    v = total_metric(data, "error_request_count")
    if v is not None:
        return v
    summary = data.get("error_summary")
    if isinstance(summary, list):
        total = 0
        for entry in summary:
            if isinstance(entry, dict):
                count = entry.get("count")
                if isinstance(count, (int, float)) and not isinstance(count, bool):
                    total += count
                else:
                    total += 1
            else:
                total += 1
        return float(total)
    return None


def _read_summary(path):
    try:
        data = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"WARNING: cannot read AIPerf summary JSON {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"WARNING: AIPerf summary JSON is not an object: {path}", file=sys.stderr)
        return None
    return data


def load_summary(run_dir):
    candidates = [
        run_dir / "profile_export_aiperf.json",
        run_dir / "profile_export.json",
    ]
    for p in candidates:
        if p.exists():
            data = _read_summary(p)
            if data is not None:
                return p, data
    for p in sorted(run_dir.glob("*.json")):
        if (
            p in candidates
            or p.name == "run_context.json"
            or "timeslice" in p.name
            or "server_metrics" in p.name
        ):
            continue
        data = _read_summary(p)
        if data is None:
            continue
        if (
            "benchmark_duration" in data
            or "total_usage_prompt_tokens" in data
            or "request_throughput" in data
        ):
            return p, data
    return None, None


def effective_resource_seconds(requested, measured):
    """Allocation seconds for the fit.

    Use the requested profiling window. If AIPerf's span extends past that
    (grace-period drain), count only half of the extra time so a linearly
    declining cooldown is not treated as fully busy.
    """
    if not finite(measured) or measured <= 0:
        return None
    if not finite(requested) or requested <= 0:
        return float(measured)
    measured = float(measured)
    requested = float(requested)
    if measured <= requested:
        return measured
    return requested + 0.5 * (measured - requested)


def write_csv(path, records, columns):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def meets_slo(r, ttft_limit, itl_limit):
    if not r["usage_ok"]:
        return False
    if ttft_limit > 0:
        if not finite(r["ttft_p99_ms"]) or r["ttft_p99_ms"] > ttft_limit:
            return False
    if itl_limit > 0:
        if not finite(r["itl_p99_ms"]) or r["itl_p99_ms"] > itl_limit:
            return False
    return True


def parse_rows(root):
    rows = []
    for ctx_path in sorted(root.glob("runs/*/c*/run_context.json")):
        ctx = json.loads(ctx_path.read_text())
        summary_path, data = load_summary(ctx_path.parent)
        if data is None:
            print(f"WARNING: no AIPerf summary JSON in {ctx_path.parent}", file=sys.stderr)
            continue

        prompt = total_metric(data, "total_usage_prompt_tokens")
        completion = total_metric(data, "total_usage_completion_tokens")
        cached_raw = total_metric(data, "total_usage_prompt_cache_read_tokens")
        cache_reported = cached_raw is not None
        cached = cached_raw if cached_raw is not None else 0.0
        if finite(prompt) and finite(cached) and cache_reported and cached > prompt:
            print(
                f"WARNING: {ctx_path.parent}: cached_tokens ({cached}) > prompt_tokens "
                f"({prompt}); marking usage invalid",
                file=sys.stderr,
            )

        usage_ok = (
            finite(prompt)
            and prompt >= 0
            and finite(completion)
            and completion >= 0
            and finite(cached)
            and cached >= 0
            and cached <= prompt
        )
        noncached = prompt - cached if finite(prompt) and finite(cached) else None

        duration = total_metric(data, "benchmark_duration", prefer_sum=False)
        req_tput = total_metric(data, "request_throughput", prefer_sum=False)
        out_tput = total_metric(data, "output_token_throughput", prefer_sum=False)
        total_tput = total_metric(data, "total_token_throughput", prefer_sum=False)
        reasoning = total_metric(data, "total_usage_reasoning_tokens") or 0.0
        request_count = total_metric(data, "request_count")
        requested = ctx.get("requested_duration_seconds")
        resource_seconds = effective_resource_seconds(requested, duration)

        actual_osl = None
        osl_ratio = None
        osl = ctx.get("osl")
        if (
            finite(completion)
            and finite(request_count)
            and request_count > 0
            and finite(osl)
            and osl > 0
        ):
            actual_osl = completion / request_count
            osl_ratio = actual_osl / osl

        row = {
            **ctx,
            "summary_json": str(summary_path.relative_to(root)),
            "usage_ok": int(usage_ok),
            "cache_reported": int(cache_reported),
            "prompt_tokens": prompt,
            "noncached_input_tokens": noncached,
            "cached_input_tokens": cached,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "benchmark_duration_s": duration,
            "resource_seconds": resource_seconds,
            "request_throughput_rps": req_tput,
            "output_token_throughput_tps": out_tput,
            "total_token_throughput_tps": total_tput,
            "ttft_avg_ms": scalar_metric(data, "time_to_first_token", "avg"),
            "ttft_p95_ms": scalar_metric(data, "time_to_first_token", "p95"),
            "ttft_p99_ms": scalar_metric(data, "time_to_first_token", "p99"),
            "itl_avg_ms": scalar_metric(data, "inter_token_latency", "avg"),
            "itl_p95_ms": scalar_metric(data, "inter_token_latency", "p95"),
            "itl_p99_ms": scalar_metric(data, "inter_token_latency", "p99"),
            "request_latency_p99_ms": scalar_metric(data, "request_latency", "p99"),
            "request_count": request_count,
            "actual_osl": actual_osl,
            "osl_ratio": osl_ratio,
            "error_request_count": extract_error_count(data),
            "aiperf_exit_code": ctx.get("aiperf_exit_code"),
            "aiperf_ok": ctx.get("aiperf_ok"),
        }

        t_for_rate = resource_seconds if finite(resource_seconds) and resource_seconds > 0 else None
        if t_for_rate and usage_ok:
            row["api_prompt_tps"] = prompt / t_for_rate
            row["api_completion_tps"] = completion / t_for_rate
            row["api_total_tps"] = (prompt + completion) / t_for_rate
        else:
            row["api_prompt_tps"] = None
            row["api_completion_tps"] = None
            row["api_total_tps"] = None

        rows.append(row)
    return rows


def annotate_plateau(candidates, chosen, ttft_limit, itl_limit):
    max_tested = max(r["concurrency"] for r in candidates)
    chosen_tput = chosen["request_throughput_rps"] if finite(chosen["request_throughput_rps"]) else None
    with_tput = [r for r in candidates if finite(r["request_throughput_rps"])]
    global_best = max((r["request_throughput_rps"] for r in with_tput), default=None)

    if chosen["concurrency"] == max_tested:
        lower = [r for r in with_tput if r["concurrency"] < max_tested]
        if not lower or chosen_tput is None:
            return False, "at_sweep_edge"
        prev = max(lower, key=lambda r: r["concurrency"])
        if chosen_tput <= prev["request_throughput_rps"] * (1.0 + PLATEAU_REL):
            return True, None
        return False, "at_sweep_edge"

    peaked = (
        chosen_tput is not None
        and global_best is not None
        and chosen_tput >= (1.0 - PLATEAU_REL) * global_best
    )
    if peaked:
        return True, "peak"
    if ttft_limit > 0 or itl_limit > 0:
        return False, "slo_capped"
    return False, "below_peak"


def select_capacity_points(rows, ttft_limit, itl_limit):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["profile"]].append(r)

    selected = []
    for profile, candidates in sorted(grouped.items()):
        errored = [r for r in candidates if (r.get("error_request_count") or 0) > 0]
        if errored:
            print(
                f"WARNING: '{profile}': excluding {len(errored)} run(s) with "
                "errored requests from capacity selection",
                file=sys.stderr,
            )
        usable = [
            r
            for r in candidates
            if r.get("aiperf_ok") is not False
            and (r.get("error_request_count") or 0) == 0
            and r["usage_ok"]
        ]
        feasible = [
            r
            for r in usable
            if meets_slo(r, ttft_limit, itl_limit)
        ]
        if not feasible:
            print(
                f"WARNING: no error-free/SLO-feasible/API-usage-valid run for profile '{profile}'",
                file=sys.stderr,
            )
            continue
        feasible.sort(
            key=lambda r: (
                r["request_throughput_rps"] if finite(r["request_throughput_rps"]) else -1,
                r["concurrency"],
            )
        )
        chosen = dict(feasible[-1])
        plateau, note = annotate_plateau(usable, chosen, ttft_limit, itl_limit)
        chosen["plateau_reached"] = int(plateau)
        if note:
            chosen["selection_note"] = note
        else:
            chosen["selection_note"] = None
        if note == "at_sweep_edge":
            print(
                f"NOTE: '{profile}' best point is at the max tested concurrency "
                f"({chosen['concurrency']}); saturation may not be reached. "
                "Consider extending --concurrency upward.",
                file=sys.stderr,
            )
        elif note == "slo_capped":
            print(
                f"NOTE: '{profile}' selected concurrency {chosen['concurrency']} "
                "is below the throughput peak because of SLOs; GPU saturation "
                "may not have been reached.",
                file=sys.stderr,
            )
        selected.append(chosen)
    return selected


def fit_exclude_reason(r):
    if not r["usage_ok"]:
        return "usage_not_ok"
    if not finite(r["resource_seconds"]) or r["resource_seconds"] <= 0:
        return "no_resource_seconds"
    if not finite(r["noncached_input_tokens"]) or not finite(r["completion_tokens"]):
        return "tokens_not_finite"
    prefix = r.get("prefix_tokens") or 0
    if prefix > 0:
        if not r.get("cache_reported"):
            return "cache_tokens_not_reported"
        if not finite(r["cached_input_tokens"]) or r["cached_input_tokens"] <= 0:
            return "cache_tokens_zero"
    if finite(r.get("osl_ratio")) and r["osl_ratio"] < OSL_RATIO_MIN:
        return f"actual_osl_ratio_{r['osl_ratio']:.3f}_below_{OSL_RATIO_MIN}"
    return None


def nnls_enumerate(X, y):
    """Non-negative least squares via subset OLS, fitted in rate space.

    Minimizes ||1 - (X / y) β||² subject to β ≥ 0. Each row is weighted
    equally regardless of token volume. Returns β in seconds/token, active
    columns, and the column-scaled design-matrix condition number.
    """
    import numpy as np

    t = y.reshape(-1, 1)
    X_rate = X / t
    if not np.all(np.isfinite(X_rate)):
        return None
    column_norms = np.linalg.norm(X_rate, axis=0)
    if np.any(~np.isfinite(column_norms)) or np.any(column_norms <= 0):
        return None
    X_rate_scaled = X_rate / column_norms
    condition_number = float(np.linalg.cond(X_rate_scaled))
    y_rate = np.ones(len(y))
    best = None
    p = X_rate.shape[1]
    if (
        np.linalg.matrix_rank(X_rate_scaled) < p
        or not np.isfinite(condition_number)
        or condition_number > MAX_CONDITION_NUMBER
    ):
        return None
    for mask in range(1, 1 << p):
        cols = [i for i in range(p) if mask & (1 << i)]
        beta_sub, *_ = np.linalg.lstsq(X_rate[:, cols], y_rate, rcond=None)
        if np.any(beta_sub < -1e-15):
            continue
        beta_sub = np.maximum(beta_sub, 0.0)
        beta = np.zeros(p)
        beta[cols] = beta_sub
        residual = y_rate - X_rate @ beta
        rss = float(residual @ residual)
        if best is None or rss < best[0]:
            best = (rss, beta, cols)
    if best is None:
        return None
    return best[1], best[2], condition_number


def bootstrap_multipliers(X, y, names, n_boot, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(y)
    samples = {name: [] for name in names if name != "noncached_input"}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        fitted = nnls_enumerate(X[idx], y[idx])
        if fitted is None:
            continue
        beta, _, _ = fitted
        coeff = dict(zip(names, beta))
        a = float(coeff.get("noncached_input", 0.0))
        if a <= ZERO_COEF_EPS:
            continue
        if "output" in coeff:
            samples["output"].append(float(coeff["output"]) / a)
        if "cached_input" in coeff:
            samples["cached_input"].append(float(coeff["cached_input"]) / a)

    def ci(values):
        if len(values) < 50:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "p025": float(np.percentile(arr, 2.5)),
            "p975": float(np.percentile(arr, 97.5)),
            "n_bootstrap": int(len(values)),
        }

    out = {"outputMultiplier": ci(samples.get("output", []))}
    if "cached_input" in names:
        out["cachedMultiplier"] = ci(samples.get("cached_input", []))
    else:
        out["cachedMultiplier"] = None
    return out


def main(argv):
    root = pathlib.Path(argv[1])
    ttft_limit = float(argv[2])
    itl_limit = float(argv[3])
    hour_cost = float(argv[4]) if argv[4] else None

    rows = parse_rows(root)
    if not rows:
        raise SystemExit("No benchmark summaries could be parsed.")

    write_csv(root / "summary.csv", rows, SUMMARY_COLUMNS)

    selected = select_capacity_points(rows, ttft_limit, itl_limit)
    for r in selected:
        reason = fit_exclude_reason(r)
        r["fit_exclude_reason"] = reason
        r["fit_eligible"] = int(reason is None)
        if reason:
            print(
                f"WARNING: '{r['profile']}' excluded from coefficient fit: {reason}",
                file=sys.stderr,
            )

    write_csv(root / "selected_capacity_points.csv", selected, SELECTED_COLUMNS)

    model_name = None
    seed = 100
    cfg_path = root / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        model_name = cfg.get("model")
        if cfg.get("random_seed") is not None:
            seed = int(cfg["random_seed"])

    fit_result = {
        "model": model_name,
        "selection": {
            "ttft_p99_ms_max": ttft_limit or None,
            "itl_p99_ms_max": itl_limit or None,
            "rule": (
                "highest request throughput per workload among error-free, "
                "API-usage-valid runs satisfying enabled SLOs"
            ),
            "plateau_relative_threshold": PLATEAU_REL,
            "osl_ratio_min": OSL_RATIO_MIN,
            "selected_profiles": [
                {
                    "profile": r["profile"],
                    "concurrency": r["concurrency"],
                    "selection_note": r.get("selection_note"),
                    "plateau_reached": bool(r.get("plateau_reached")),
                    "fit_eligible": bool(r.get("fit_eligible")),
                    "fit_exclude_reason": r.get("fit_exclude_reason"),
                }
                for r in selected
            ],
        },
        "resource_seconds_definition": (
            "requested_duration + 0.5 * max(0, benchmark_duration - requested_duration)"
        ),
        "notes": [
            "completion_tokens already includes provider-billed completion/reasoning tokens in OpenAI-style usage; reasoning_tokens is recorded only as a diagnostic and is not added again.",
            "cached_input_tokens is only measurable when the endpoint reports usage.prompt_tokens_details.cached_tokens. Cache-plan rows without reported cache hits are excluded from the fit so they cannot contaminate the input coefficient.",
            "The regression uses drain-adjusted resource seconds and equal-weight rate-space NNLS. Inspect residual error before using it for billing.",
        ],
    }

    try:
        import numpy as np
    except Exception as exc:
        fit_result["fit_error"] = f"numpy unavailable: {exc}"
        (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
        _print_report(root, fit_result)
        return 0

    fit_rows = [r for r in selected if r.get("fit_eligible")]
    have_cache = any(
        r["cache_reported"] and r["cached_input_tokens"] > 0 for r in fit_rows
    )
    names = ["noncached_input", "output"]
    if have_cache:
        names.append("cached_input")

    if len(fit_rows) < len(names):
        fit_result["fit_error"] = (
            f"not enough selected workload points: {len(fit_rows)} eligible rows for "
            f"{len(names)} coefficients"
        )
        (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
        _print_report(root, fit_result)
        return 0

    X_rows = []
    y_list = []
    for r in fit_rows:
        x = [r["noncached_input_tokens"], r["completion_tokens"]]
        if have_cache:
            x.append(r["cached_input_tokens"] if r["cache_reported"] else 0.0)
        X_rows.append(x)
        y_list.append(r["resource_seconds"])

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_list, dtype=float)

    fitted = nnls_enumerate(X, y)
    if fitted is None:
        fit_result["fit_error"] = "no stable non-negative coefficient solution found"
        (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
        _print_report(root, fit_result)
        return 0

    beta, _, cond = fitted
    pred = X @ beta
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mean_y = float(np.mean(y))
    rel_rmse = rmse / mean_y if mean_y else None
    rate_resid = 1.0 - (pred / y)
    rate_rmse = float(np.sqrt(np.mean(rate_resid ** 2)))
    coeff = dict(zip(names, [float(v) for v in beta]))
    a = coeff.get("noncached_input", 0.0)
    b = coeff.get("output", 0.0)
    c = coeff.get("cached_input") if have_cache else None
    cache_identified = have_cache and c is not None and c > ZERO_COEF_EPS

    if a <= ZERO_COEF_EPS:
        fit_result["fit_error"] = (
            "fitted noncached-input coefficient is zero; workload matrix "
            "does not identify an input baseline well enough"
        )
        (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
        _print_report(root, fit_result)
        return 0
    if b <= ZERO_COEF_EPS:
        fit_result["fit_error"] = (
            "fitted output coefficient is zero; workload matrix does not "
            "identify output cost well enough"
        )
        (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
        _print_report(root, fit_result)
        return 0

    equations = []
    for r, xrow, t_obs, t_hat in zip(fit_rows, X_rows, y, pred):
        parts = [
            f"{xrow[0]}*a_noncached_input",
            f"{xrow[1]}*b_output",
        ]
        if have_cache:
            parts.append(f"{xrow[2]}*c_cached_input")
        equations.append(
            {
                "profile": r["profile"],
                "concurrency": r["concurrency"],
                "resource_seconds": float(t_obs),
                "benchmark_duration_s": r["benchmark_duration_s"],
                "requested_duration_seconds": r.get("requested_duration_seconds"),
                "noncached_input_tokens": xrow[0],
                "output_tokens": xrow[1],
                "cached_input_tokens": xrow[2] if have_cache else 0.0,
                "predicted_seconds": float(t_hat),
                "residual_seconds": float(t_obs - t_hat),
                "equation": " + ".join(parts) + f" ~= {float(t_obs)}",
            }
        )

    seconds_per_mtok = {
        "noncached_input": a * 1_000_000,
        "output": b * 1_000_000,
        "cached_input": (c * 1_000_000) if cache_identified else None,
    }
    weighted_capacity = (3600.0 / a / 1_000_000.0) if a > ZERO_COEF_EPS else None

    fit_result["fit"] = {
        "equation": (
            "resource_seconds ~= "
            "a*noncached_input_tokens + b*output_tokens"
            + (" + c*cached_input_tokens" if have_cache else "")
        ),
        "objective": "equal-weight NNLS on 1 ~= a*(I/T) + b*(O/T) + c*(C/T)",
        "coefficients_resource_seconds_per_token": {
            "a_noncached_input": a,
            "b_output": b,
            "c_cached_input": c if have_cache else None,
        },
        "relative_rmse": rel_rmse,
        "rmse_seconds": rmse,
        "rate_rmse": rate_rmse,
        "condition_number": cond,
        "condition_number_max": MAX_CONDITION_NUMBER,
        "rows_used": len(fit_rows),
        "cache_coefficient_fitted": have_cache,
        "cache_coefficient_identified": cache_identified,
    }
    fit_result["equations"] = equations
    fit_result["time_model"] = {
        "seconds_per_token": {
            "noncached_input": a,
            "output": b,
            "cached_input": c if cache_identified else None,
        },
        "seconds_per_million_tokens": seconds_per_mtok,
        "weighted_capacity_mtok_per_hour": weighted_capacity,
    }

    multipliers = {
        "inputMultiplier": 1.0,
        "outputMultiplier": b / a,
        "cachedMultiplier": (c / a) if cache_identified else None,
    }
    fit_result["multipliers"] = multipliers
    fit_result["multiplier_confidence_intervals"] = bootstrap_multipliers(
        X, y, names, BOOTSTRAP_SAMPLES, seed
    )

    if hour_cost is not None:
        usd_per_resource_second = hour_cost / 3600.0
        input_per_m = a * usd_per_resource_second * 1_000_000
        output_per_m = b * usd_per_resource_second * 1_000_000
        cache_per_m = (
            c * usd_per_resource_second * 1_000_000 if cache_identified else None
        )
        fit_result["resource_hour_cost_usd"] = hour_cost
        fit_result["optional_cost_model"] = {
            "resource_hour_cost": hour_cost,
            "cost_recovery_per_million_tokens": {
                "noncached_input": input_per_m,
                "output": output_per_m,
                "cached_input": cache_per_m,
            },
        }
        fit_result["cost_recovery_usd_per_million_tokens"] = {
            "noncached_input": input_per_m,
            "output": output_per_m,
            "cached_input": cache_per_m,
        }
        fit_result["cost_formula"] = (
            "cost_usd = "
            "noncached_input_tokens*input_usd_per_token + "
            "output_tokens*output_usd_per_token"
            + (
                " + cached_input_tokens*cached_input_usd_per_token"
                if cache_identified
                else ""
            )
        )

    (root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")
    _print_report(root, fit_result)
    return 0


def _print_report(root, fit_result):
    print()
    print("======================================================================")
    print("Benchmark complete")
    print("======================================================================")
    print(f"All runs:          {root / 'summary.csv'}")
    print(f"Capacity points:   {root / 'selected_capacity_points.csv'}")
    print(f"Pricing fit:       {root / 'pricing_fit.json'}")
    print()

    if "multipliers" in fit_result:
        m = fit_result["multipliers"]
        ci = fit_result.get("multiplier_confidence_intervals") or {}
        print("Fitted multipliers:")
        print(f"  inputMultiplier  = {m['inputMultiplier']:.6g}")
        out_ci = ci.get("outputMultiplier")
        if out_ci:
            print(
                f"  outputMultiplier = {m['outputMultiplier']:.6g} "
                f"(95% CI [{out_ci['p025']:.6g}, {out_ci['p975']:.6g}])"
            )
        else:
            print(f"  outputMultiplier = {m['outputMultiplier']:.6g}")
        if m["cachedMultiplier"] is None:
            print("  cachedMultiplier = not available (cache tokens not reported/identified)")
        else:
            cache_ci = ci.get("cachedMultiplier")
            if cache_ci:
                print(
                    f"  cachedMultiplier = {m['cachedMultiplier']:.6g} "
                    f"(95% CI [{cache_ci['p025']:.6g}, {cache_ci['p975']:.6g}])"
                )
            else:
                print(f"  cachedMultiplier = {m['cachedMultiplier']:.6g}")

    tm = fit_result.get("time_model") or {}
    cap = tm.get("weighted_capacity_mtok_per_hour")
    if cap is not None:
        print()
        print(f"Weighted serving capacity: {cap:.6g} input-equivalent MTok / hour")

    if "cost_recovery_usd_per_million_tokens" in fit_result:
        p = fit_result["cost_recovery_usd_per_million_tokens"]
        print()
        print("Approximate cost-recovery prices:")
        print(f"  non-cached input = ${p['noncached_input']:.6g} / MTok")
        print(f"  output           = ${p['output']:.6g} / MTok")
        if p["cached_input"] is None:
            print("  cached input     = not available")
        else:
            print(f"  cached input     = ${p['cached_input']:.6g} / MTok")

    fit = fit_result.get("fit") or {}
    if fit.get("relative_rmse") is not None:
        print()
        print(f"Fit relative RMSE: {100 * fit['relative_rmse']:.2f}%")
        if fit.get("condition_number") is not None:
            print(f"Design-matrix condition number: {fit['condition_number']:.4g}")
        if fit["relative_rmse"] > 0.15:
            print("WARNING: fit error > 15%; the additive token-cost model may be too simple")
            print("         for these workload points. Inspect summary.csv before billing.")
    if "fit_error" in fit_result:
        print()
        print("Fit warning:", fit_result["fit_error"])

    print()
    print("Note: these are deployment-capacity coefficients, not external provider prices.")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
