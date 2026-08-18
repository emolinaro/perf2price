#!/usr/bin/env python3
"""Plot throughput–latency Pareto curves from one or more perf2price runs.

Each workload profile becomes one figure. Models from the selected runs are
overlaid on the same axes. Markers are the tested concurrencies; the legend
sits below the axes and uses the model name from run metadata, without the
Hugging Face provider prefix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

PROFILE_ORDER = (
    "prefill_1k",
    "prefill_4k",
    "prefill_16k",
    "decode_512",
    "decode_2k",
    "mixed_4k",
    "mixed_8k",
    "cache_4k",
    "cache_8k",
    "cache_16k",
    "cache_32k",
)

METRIC_LABELS = {
    "output_token_throughput_tps": ("Output token throughput", "tokens/s"),
    "total_token_throughput_tps": ("Total token throughput", "tokens/s"),
    "api_completion_tps": ("API completion throughput", "tokens/s"),
    "api_total_tps": ("API total throughput", "tokens/s"),
    "request_throughput_rps": ("Request throughput", "requests/s"),
    "ttft_p99_ms": ("TTFT p99", "ms"),
    "ttft_p95_ms": ("TTFT p95", "ms"),
    "ttft_avg_ms": ("TTFT mean", "ms"),
    "itl_p99_ms": ("ITL p99", "ms"),
    "itl_p95_ms": ("ITL p95", "ms"),
    "itl_avg_ms": ("ITL mean", "ms"),
    "request_latency_p99_ms": ("Request latency p99", "ms"),
}

DEFAULT_X = "output_token_throughput_tps"
DEFAULT_Y = ("ttft_p99_ms", "itl_p99_ms")

# Okabe–Ito, yellow omitted (poor contrast on white).
COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#000000",
)
MARKERS = ("o", "s", "D", "^", "v", "P", "X")
ANNOTATION_OFFSETS = (
    (5.5, 5.5),
    (5.5, -11.0),
    (-18.0, 5.5),
    (5.5, 12.0),
    (-18.0, -11.0),
)
LOG_SPAN = 20.0


@dataclass(frozen=True)
class Point:
    concurrency: int
    values: dict[str, float]


@dataclass(frozen=True)
class Series:
    label: str
    model: str
    run_dir: Path
    points: tuple[Point, ...]
    isl: int | None
    osl: int | None
    prefix_tokens: int | None


def short_model_name(model: str | None) -> str:
    """Drop the org/provider prefix: moonshotai/Kimi-K3 -> Kimi-K3."""
    if model is None:
        return "unknown"
    name = str(model).strip().rstrip("/")
    if not name:
        return "unknown"
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return name or "unknown"


def metric_axis_label(metric: str) -> str:
    if metric in METRIC_LABELS:
        name, unit = METRIC_LABELS[metric]
        return f"{name} ({unit})"
    return metric.replace("_", " ")


def _finite_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _int_or_none(value) -> int | None:
    number = _finite_float(value)
    if number is None:
        return None
    return int(number)


def discover_run_dirs(paths: list[Path]) -> list[Path]:
    """Accept run directories or a parent that contains them."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> bool:
        resolved = path.resolve()
        if resolved in seen:
            return True
        if (resolved / "run_config.json").is_file():
            seen.add(resolved)
            found.append(resolved)
            return True
        return False

    for raw in paths:
        path = Path(raw)
        if not path.is_dir():
            raise SystemExit(f"not a directory: {path}")
        if add(path):
            continue
        for child in sorted(path.iterdir()):
            if child.is_dir():
                add(child)
    if not found:
        raise SystemExit("no perf2price run directories found (need run_config.json)")
    return found


def _load_config(run_dir: Path) -> dict:
    path = run_dir / "run_config.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"run config is not an object: {path}")
    return payload


def _row_is_plottable(row: dict, metrics: tuple[str, ...]) -> bool:
    if not _truthy(row.get("aiperf_ok")):
        return False
    if not _truthy(row.get("usage_ok")):
        return False
    errors = _finite_float(row.get("error_request_count"))
    if errors is not None and errors > 0:
        return False
    if _int_or_none(row.get("concurrency")) is None:
        return False
    return all(_finite_float(row.get(metric)) is not None for metric in metrics)


def _load_summary_rows(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.csv"
    if summary_path.is_file():
        with summary_path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    import perf2price_fit as fit

    parsed = fit.parse_rows(run_dir)
    return [{key: row.get(key) for key in row} for row in parsed]


def load_run_series(
    run_dir: Path, x_metric: str, y_metrics: tuple[str, ...]
) -> tuple[str, dict[str, list[Point]], dict[str, tuple[int | None, int | None, int | None]]]:
    config = _load_config(run_dir)
    model = config.get("model")
    if not isinstance(model, str) or not model.strip():
        model = run_dir.name
    required = (x_metric, *y_metrics)
    grouped: dict[str, list[Point]] = defaultdict(list)
    specs: dict[str, tuple[int | None, int | None, int | None]] = {}
    skipped = 0
    for row in _load_summary_rows(run_dir):
        profile = row.get("profile")
        if not profile:
            continue
        if not _row_is_plottable(row, required):
            skipped += 1
            continue
        values = {metric: float(_finite_float(row[metric])) for metric in required}
        grouped[profile].append(
            Point(concurrency=int(_int_or_none(row["concurrency"])), values=values)
        )
        specs[profile] = (
            _int_or_none(row.get("isl")),
            _int_or_none(row.get("osl")),
            _int_or_none(row.get("prefix_tokens")),
        )
    for profile, points in grouped.items():
        points.sort(key=lambda point: point.concurrency)
    if skipped:
        print(
            f"NOTE: {run_dir.name}: omitted {skipped} row(s) that were failed, "
            "errored, or missing the requested metrics",
            file=sys.stderr,
        )
    return model, dict(grouped), specs


def _disambiguate_labels(models: list[str], run_dirs: list[Path]) -> list[str]:
    shorts = [short_model_name(model) for model in models]
    counts = Counter(shorts)
    labels = []
    for short, run_dir in zip(shorts, run_dirs):
        if counts[short] == 1:
            labels.append(short)
            continue
        stamp = run_dir.name.removeprefix("perf2price-run-")
        labels.append(f"{short} ({stamp})")
    return labels


def collect_series(
    run_dirs: list[Path], x_metric: str, y_metrics: tuple[str, ...]
) -> dict[str, list[Series]]:
    loaded = []
    for run_dir in run_dirs:
        model, grouped, specs = load_run_series(run_dir, x_metric, y_metrics)
        loaded.append((run_dir, model, grouped, specs))
    labels = _disambiguate_labels([item[1] for item in loaded], [item[0] for item in loaded])
    by_profile: dict[str, list[Series]] = defaultdict(list)
    for label, (run_dir, model, grouped, specs) in zip(labels, loaded):
        for profile, points in grouped.items():
            if not points:
                continue
            isl, osl, prefix = specs.get(profile, (None, None, None))
            by_profile[profile].append(
                Series(
                    label=label,
                    model=model,
                    run_dir=run_dir,
                    points=tuple(points),
                    isl=isl,
                    osl=osl,
                    prefix_tokens=prefix,
                )
            )
    order_index = {name: i for i, name in enumerate(PROFILE_ORDER)}
    return dict(
        sorted(
            by_profile.items(),
            key=lambda item: (order_index.get(item[0], len(PROFILE_ORDER)), item[0]),
        )
    )


def _profile_subtitle(series_list: list[Series]) -> str:
    specs = {(item.isl, item.osl, item.prefix_tokens) for item in series_list}
    if len(specs) != 1:
        return "Workload lengths differ across models; see run metadata."
    isl, osl, prefix = next(iter(specs))
    parts = []
    if isl is not None:
        parts.append(f"ISL {isl:,}")
    if osl is not None:
        parts.append(f"OSL {osl:,}")
    if prefix:
        parts.append(f"prefix {prefix:,}")
    return "  ·  ".join(parts)


def _should_log(values: list[float]) -> bool:
    positive = [value for value in values if value > 0]
    if len(positive) < 2:
        return False
    return max(positive) / min(positive) >= LOG_SPAN


def _apply_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.formatter.useoffset": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _legend_columns(labels: list[str]) -> int:
    if not labels:
        return 1
    return min(len(labels), 3)


def _padded_limits(values: list[float], log_scale: bool) -> tuple[float, float]:
    vmin = min(values)
    vmax = max(values)
    if log_scale:
        vmin = min(value for value in values if value > 0)
        span = math.log10(vmax) - math.log10(vmin)
        span = span if span > 0 else 0.3
        return 10 ** (math.log10(vmin) - 0.08 * span), 10 ** (math.log10(vmax) + 0.18 * span)
    span = vmax - vmin if vmax > vmin else max(abs(vmax), 1.0)
    return 0.0, vmax + 0.16 * span


def _annotate_series(ax, series: Series, x_metric: str, y_metric: str, series_index: int) -> None:
    from matplotlib import patheffects

    color = COLORS[series_index % len(COLORS)]
    offset = ANNOTATION_OFFSETS[series_index % len(ANNOTATION_OFFSETS)]
    halo = [patheffects.withStroke(linewidth=2.6, foreground="white")]
    last_display = None
    for point in series.points:
        x_value = point.values[x_metric]
        y_value = point.values[y_metric]
        display = ax.transData.transform((x_value, y_value))
        if last_display is not None:
            dx = display[0] - last_display[0]
            dy = display[1] - last_display[1]
            if dx * dx + dy * dy < 12.0 * 12.0:
                continue
        last_display = display
        text = ax.annotate(
            f"c{point.concurrency}",
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.5,
            color=color,
            ha="left" if offset[0] >= 0 else "right",
            va="center",
            zorder=4 + series_index,
        )
        text.set_path_effects(halo)


def plot_profile(
    profile: str,
    series_list: list[Series],
    x_metric: str,
    y_metrics: tuple[str, ...],
    out_dir: Path,
    formats: tuple[str, ...],
    annotate: bool,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator

    series_list = sorted(series_list, key=lambda item: item.label.lower())
    n_panels = len(y_metrics)
    height = 3.05 + 2.45 * n_panels
    fig, axes = plt.subplots(
        n_panels,
        1,
        sharex=True,
        figsize=(7.16, height),
        layout="constrained",
        squeeze=False,
    )
    axes = axes[:, 0]
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, hspace=0.06, wspace=0.04)
    panel_tags = "abcdefghijklmnopqrstuvwxyz"

    for panel_index, (ax, y_metric) in enumerate(zip(axes, y_metrics)):
        x_all: list[float] = []
        y_all: list[float] = []
        for series_index, series in enumerate(series_list):
            xs = [point.values[x_metric] for point in series.points]
            ys = [point.values[y_metric] for point in series.points]
            x_all.extend(xs)
            y_all.extend(ys)
            color = COLORS[series_index % len(COLORS)]
            marker = MARKERS[series_index % len(MARKERS)]
            ax.plot(
                xs,
                ys,
                color=color,
                marker=marker,
                markersize=7.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.75,
                linewidth=1.7,
                linestyle="-",
                label=series.label if panel_index == 0 else None,
                zorder=3 + series_index,
            )

        log_y = _should_log(y_all)
        log_x = bool(x_all) and min(x_all) > 0 and _should_log(x_all)
        if log_y:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        if log_x:
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto"))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        if x_all:
            ax.set_xlim(*_padded_limits(x_all, log_x))
        if y_all:
            ax.set_ylim(*_padded_limits(y_all, log_y))

        ax.set_ylabel(metric_axis_label(y_metric))
        ax.set_title(f"({panel_tags[panel_index]})", loc="left", pad=3, fontweight="bold")
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linestyle=":", linewidth=0.55, color="#6e6e6e", alpha=0.55)

    if annotate:
        fig.canvas.draw()
        for ax, y_metric in zip(axes, y_metrics):
            for series_index, series in enumerate(series_list):
                _annotate_series(ax, series, x_metric, y_metric, series_index)

    axes[-1].set_xlabel(metric_axis_label(x_metric))
    subtitle = _profile_subtitle(series_list)
    title = profile if not subtitle else f"{profile}\n{subtitle}"
    fig.suptitle(title, fontsize=11, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=_legend_columns(labels),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.4,
        handletextpad=0.55,
        borderaxespad=0.15,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    stem = f"pareto-{profile}"
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, facecolor="white")
        written.append(path)
    plt.close(fig)
    return written


def parse_metrics(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return default
    metrics: list[str] = []
    for value in values:
        for item in value.split(","):
            metric = item.strip()
            if not metric:
                continue
            if metric not in METRIC_LABELS:
                known = ", ".join(METRIC_LABELS)
                raise SystemExit(f"unknown metric '{metric}'. Known: {known}")
            if metric not in metrics:
                metrics.append(metric)
    if not metrics:
        return default
    return tuple(metrics)


def parse_formats(value: str) -> tuple[str, ...]:
    allowed = {"png", "pdf", "svg"}
    formats = []
    for item in value.split(","):
        fmt = item.strip().lower()
        if not fmt:
            continue
        if fmt not in allowed:
            raise SystemExit(f"unsupported format '{fmt}'. Use png, pdf, or svg")
        if fmt not in formats:
            formats.append(fmt)
    if not formats:
        raise SystemExit("no output formats given")
    return tuple(formats)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a throughput–latency Pareto figure for each benchmark "
            "profile, overlaying models from the given perf2price run directories."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories, or a parent such as perf2price-run-selected",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("pareto-figures"),
        help="Directory for figures (default: pareto-figures)",
    )
    parser.add_argument(
        "--x-metric",
        default=DEFAULT_X,
        help=f"Throughput metric on the x-axis (default: {DEFAULT_X})",
    )
    parser.add_argument(
        "--y-metric",
        action="append",
        dest="y_metrics",
        help=(
            "Latency metric(s) on the y-axis. Repeat or comma-separate. "
            f"Default: {','.join(DEFAULT_Y)}"
        ),
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated figure formats: pdf, png, svg (default: pdf,png)",
    )
    parser.add_argument(
        "--no-annotate-concurrency",
        action="store_true",
        help="Do not label markers with c<concurrency>",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    x_metric = parse_metrics([args.x_metric], (DEFAULT_X,))[0]
    y_metrics = parse_metrics(args.y_metrics, DEFAULT_Y)
    if x_metric in y_metrics:
        raise SystemExit("x-metric and y-metric must differ")
    formats = parse_formats(args.formats)
    run_dirs = discover_run_dirs(list(args.paths))
    by_profile = collect_series(run_dirs, x_metric, y_metrics)
    if not by_profile:
        raise SystemExit("no plottable benchmark points found")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is required for plotting. Install with:\n"
            "  pip install -r requirements-plot.txt",
            file=sys.stderr,
        )
        return 1

    _apply_style(plt)
    written: list[Path] = []
    for profile, series_list in by_profile.items():
        written.extend(
            plot_profile(
                profile,
                series_list,
                x_metric,
                y_metrics,
                args.out_dir,
                formats,
                annotate=not args.no_annotate_concurrency,
            )
        )
        labels = ", ".join(item.label for item in series_list)
        print(f"{profile}: {len(series_list)} model(s) ({labels})")

    print(f"Wrote {len(written)} figure(s) to {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
