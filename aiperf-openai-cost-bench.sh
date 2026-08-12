#!/usr/bin/env bash
set -euo pipefail

# Backend-agnostic AIPerf harness for OpenAI-compatible chat/completions endpoints.
#
# It:
#   1. Probes the OpenAI-compatible endpoint.
#   2. Runs a matrix of prefill/decode/mixed/cache workloads.
#   3. Uses API-reported usage token counts (--use-server-token-count).
#   4. Sweeps concurrency at each workload.
#   5. Parses AIPerf summary JSON into summary.csv.
#   6. Selects the highest-throughput error-free, usage-valid concurrency per
#      workload, optionally under SLOs.
#   7. Fits:
#
#        resource_seconds ~= a * non_cached_input_tokens
#                         + b * output_tokens
#                         + c * cached_input_tokens
#
#      and reports:
#        inputMultiplier  = 1
#        outputMultiplier = b / a
#        cachedMultiplier = c / a
#
#   8. If --resource-hour-cost is supplied, converts the fitted coefficients to
#      $/M tokens for the whole serving allocation.
#
# No backend-specific metrics endpoint is required. AIPerf /metrics scraping is
# explicitly disabled. Cache accounting requires the server to expose
# usage.prompt_tokens_details.cached_tokens.

AIPERF_BIN="${AIPERF_BIN:-aiperf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

URL=""
MODEL=""
TOKENIZER=""
TOKENIZER_REVISION=""
TOKENIZER_TRUST_REMOTE_CODE=0
APPLY_CHAT_TEMPLATE=0
ENDPOINT="/v1/chat/completions"
MODELS_ENDPOINT="/v1/models"
API_KEY="${OPENAI_API_KEY:-}"

CONCURRENCY_LIST="4,8,16,32,64"
DURATION="120"
GRACE_PERIOD="120"
RANDOM_SEED="100"
NUM_DATASET_ENTRIES="12800"
OUT_DIR="./aiperf-cost-benchmark-$(date +%Y%m%d-%H%M%S)"
PLAN_FILE=""
MAX_CONTEXT="0"

# Optional SLOs. Zero means disabled.
TTFT_P99_MS="0"
ITL_P99_MS="0"

# Optional economic conversion: cost/hour of the ENTIRE serving allocation
# (e.g. 4 H100s, 8 B200s, or 2 nodes/16 B200s).
RESOURCE_HOUR_COST=""

RUN_CACHE_TESTS=1
LEGACY_MAX_TOKENS=0
SKIP_PROBE=0

EXTRA_AIPERF_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  aiperf-openai-cost-bench.sh --url URL --model MODEL [options]

Required:
  --url URL                  OpenAI-compatible base URL, e.g. http://localhost:8000
  --model MODEL              Served model name sent in requests

Tokenizer (used locally by AIPerf, not by the inference server):
  --tokenizer TOKENIZER      Model-specific HF tokenizer ID, local tokenizer
                             directory, or 'builtin'. Defaults to --model.
                             Example with a served alias:
                               --model gpt-oss-prod \
                               --tokenizer openai/gpt-oss-120b
  --tokenizer-revision REV   HF branch, tag, or commit to pin.
  --tokenizer-trust-remote-code
                             Allow trusted custom tokenizer code.
  --apply-chat-template      Ask AIPerf to account for the HF chat template
                             when targeting synthetic ISL. Useful for chat APIs.

Benchmark:
  --concurrency LIST         Comma-separated concurrency sweep
                             Default: 4,8,16,32,64
  --duration SEC             Profiling duration per point
                             Default: 120
  --grace-period SEC         Wait for in-flight responses after duration
                             Default: 120
  --out-dir DIR              Artifact directory
  --plan FILE                Custom CSV plan: name,isl,osl,prefix_tokens
  --max-context TOKENS       Skip plan rows where ISL + prefix + OSL exceeds this
                             Default: 0 (do not enforce)
  --no-cache-tests           Skip rows with prefix_tokens > 0

OpenAI API:
  --endpoint PATH            Default: /v1/chat/completions
  --models-endpoint PATH     Default: /v1/models
  --api-key KEY              Bearer token; defaults to OPENAI_API_KEY
  --legacy-max-tokens        Ask AIPerf to use max_tokens instead of
                             max_completion_tokens
  --skip-probe               Skip the small API capability probe

SLO selection:
  --ttft-p99-ms MS           Only use runs with p99 TTFT <= this value
  --itl-p99-ms MS            Only use runs with p99 ITL <= this value
                             Zero disables a threshold.

Economic conversion:
  --resource-hour-cost USD   Hourly cost of the ENTIRE serving allocation.
                             Examples:
                               4x H100 node: pass total node $/h
                               8x B200 node: pass total node $/h
                               16x B200:     pass total 2-node $/h
                             If omitted, multipliers are still fitted.

Other:
  --random-seed N            Default: 100
  --num-dataset-entries N    Default: 12800
  -h, --help

Everything after "--" is appended verbatim to every "aiperf profile" call.

Examples:

  # Local model
  ./aiperf-openai-cost-bench.sh \
    --url http://localhost:8000 \
    --model Qwen/Qwen3-30B-A3B \
    --concurrency 1,2,4,8,16

  # Remote/Kubernetes endpoint reachable from this machine
  ./aiperf-openai-cost-bench.sh \
    --url https://llm.example.org \
    --model openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --concurrency 4,8,16,32,64 \
    --ttft-p99-ms 2000 \
    --itl-p99-ms 50 \
    --resource-hour-cost 15.96

  # Custom AIPerf switches
  ./aiperf-openai-cost-bench.sh \
    --url http://localhost:8000 \
    --model my-model \
    -- \
    --network-latency-automatic
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
  --url)
    URL="${2:?missing value for --url}"
    shift 2
    ;;
  --model)
    MODEL="${2:?missing value for --model}"
    shift 2
    ;;
  --tokenizer)
    TOKENIZER="${2:?missing value for --tokenizer}"
    shift 2
    ;;
  --tokenizer-revision)
    TOKENIZER_REVISION="${2:?missing value for --tokenizer-revision}"
    shift 2
    ;;
  --tokenizer-trust-remote-code)
    TOKENIZER_TRUST_REMOTE_CODE=1
    shift
    ;;
  --apply-chat-template)
    APPLY_CHAT_TEMPLATE=1
    shift
    ;;
  --endpoint)
    ENDPOINT="${2:?missing value for --endpoint}"
    shift 2
    ;;
  --models-endpoint)
    MODELS_ENDPOINT="${2:?missing value for --models-endpoint}"
    shift 2
    ;;
  --api-key)
    API_KEY="${2:?missing value for --api-key}"
    shift 2
    ;;
  --api-key=*)
    API_KEY="${1#--api-key=}"
    [[ -n "$API_KEY" ]] || die "missing value for --api-key"
    shift
    ;;
  --concurrency)
    CONCURRENCY_LIST="${2:?missing value for --concurrency}"
    shift 2
    ;;
  --duration)
    DURATION="${2:?missing value for --duration}"
    shift 2
    ;;
  --grace-period)
    GRACE_PERIOD="${2:?missing value for --grace-period}"
    shift 2
    ;;
  --out-dir)
    OUT_DIR="${2:?missing value for --out-dir}"
    shift 2
    ;;
  --plan)
    PLAN_FILE="${2:?missing value for --plan}"
    shift 2
    ;;
  --max-context)
    MAX_CONTEXT="${2:?missing value for --max-context}"
    shift 2
    ;;
  --ttft-p99-ms)
    TTFT_P99_MS="${2:?missing value for --ttft-p99-ms}"
    shift 2
    ;;
  --itl-p99-ms)
    ITL_P99_MS="${2:?missing value for --itl-p99-ms}"
    shift 2
    ;;
  --resource-hour-cost)
    RESOURCE_HOUR_COST="${2:?missing value for --resource-hour-cost}"
    shift 2
    ;;
  --random-seed)
    RANDOM_SEED="${2:?missing value for --random-seed}"
    shift 2
    ;;
  --num-dataset-entries)
    NUM_DATASET_ENTRIES="${2:?missing value for --num-dataset-entries}"
    shift 2
    ;;
  --no-cache-tests)
    RUN_CACHE_TESTS=0
    shift
    ;;
  --legacy-max-tokens)
    LEGACY_MAX_TOKENS=1
    shift
    ;;
  --skip-probe)
    SKIP_PROBE=1
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  --)
    shift
    EXTRA_AIPERF_ARGS=("$@")
    break
    ;;
  *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$URL" ]] || die "--url is required"
[[ -n "$MODEL" ]] || die "--model is required"

URL="${URL%/}"
TOKENIZER="${TOKENIZER:-$MODEL}"

command -v "$AIPERF_BIN" >/dev/null 2>&1 || die "AIPerf not found: $AIPERF_BIN"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"

mkdir -p "$OUT_DIR"

echo "Tokenizer used by AIPerf: $TOKENIZER"
if [[ "$TOKENIZER" = "builtin" ]]; then
  echo "WARNING: using AIPerf's generic builtin tokenizer; prefer the model-specific tokenizer for precise ISL calibration." >&2
fi

# Validate numeric-ish arguments early.
"$PYTHON_BIN" - "$DURATION" "$GRACE_PERIOD" "$MAX_CONTEXT" \
  "$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}" \
  "$RANDOM_SEED" "$NUM_DATASET_ENTRIES" <<'PY'
import sys
duration, grace, max_context, ttft, itl, hourly, seed, entries = sys.argv[1:]
if float(duration) <= 0:
    raise SystemExit("--duration must be > 0")
if grace != "inf" and float(grace) < 0:
    raise SystemExit("--grace-period must be >= 0 or 'inf'")
if int(max_context) < 0:
    raise SystemExit("--max-context must be >= 0")
if float(ttft) < 0 or float(itl) < 0:
    raise SystemExit("SLO values must be >= 0")
if hourly and float(hourly) <= 0:
    raise SystemExit("--resource-hour-cost must be > 0")
if int(seed) < 0:
    raise SystemExit("--random-seed must be a non-negative integer")
if int(entries) <= 0:
    raise SystemExit("--num-dataset-entries must be a positive integer")
PY

# Default workload plan.
# isl means the unique synthetic input portion.
# For cache rows, total intended prompt size is roughly prefix_tokens + isl.
DEFAULT_PLAN="${OUT_DIR}/benchmark_plan.csv"
if [[ -n "$PLAN_FILE" ]]; then
  cp "$PLAN_FILE" "$DEFAULT_PLAN"
else
  cat >"$DEFAULT_PLAN" <<'EOF'
name,isl,osl,prefix_tokens
prefill_1k,1024,64,0
prefill_4k,4096,64,0
prefill_16k,16384,64,0
decode_512,512,512,0
decode_2k,512,2048,0
mixed_4k,4096,512,0
mixed_8k,8192,1024,0
cache_4k,512,256,4096
cache_8k,512,256,8192
EOF
fi

# Validate concurrency list and compute max.
MAX_CONCURRENCY="$(
  "$PYTHON_BIN" - "$CONCURRENCY_LIST" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
if not items:
    raise SystemExit("empty concurrency list")
vals = []
for item in items:
    v = int(item)
    if v <= 0:
        raise SystemExit(f"invalid concurrency: {v}")
    vals.append(v)
print(max(vals))
PY
)"

export AIPERF_HTTP_CONNECTION_LIMIT="${AIPERF_HTTP_CONNECTION_LIMIT:-$((MAX_CONCURRENCY + 64))}"

# Write run_config.json via Python so values are properly JSON-escaped
# (MODEL/URL may contain characters that would corrupt a raw heredoc).
"$PYTHON_BIN" - "${OUT_DIR}/run_config.json" \
  "$URL" "$ENDPOINT" "$MODELS_ENDPOINT" "$MODEL" "$TOKENIZER" \
  "$TOKENIZER_REVISION" "$CONCURRENCY_LIST" "$DURATION" "$GRACE_PERIOD" \
  "$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}" \
  "$RANDOM_SEED" "$NUM_DATASET_ENTRIES" <<'PY'
import json
import sys

(path, url, endpoint, models_endpoint, model, tokenizer, revision,
 concurrency_list, duration, grace, ttft, itl, hourly_cost,
 seed, entries) = sys.argv[1:16]

def num(v):
    n = float(v)
    return int(n) if n == int(n) else n

config = {
    "url": url,
    "endpoint": endpoint,
    "models_endpoint": models_endpoint,
    "model": model,
    "tokenizer": tokenizer,
    "tokenizer_revision": revision or None,
    "concurrency_list": concurrency_list,
    "duration_seconds": num(duration),
    "grace_period": grace,
    "ttft_p99_ms": num(ttft),
    "itl_p99_ms": num(itl),
    "resource_hour_cost_usd": num(hourly_cost) if hourly_cost else None,
    "random_seed": int(seed),
    "num_dataset_entries": int(entries),
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
PY

probe_endpoint() {
  echo "Probing OpenAI-compatible endpoint: ${URL}${ENDPOINT}"
  "$PYTHON_BIN" - "$URL" "$MODELS_ENDPOINT" "$ENDPOINT" "$MODEL" "$API_KEY" "$LEGACY_MAX_TOKENS" <<'PY'
import json
import math
import sys
import urllib.error
import urllib.request

base, models_path, chat_path, model, api_key, legacy = sys.argv[1:7]

headers = {"Content-Type": "application/json"}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

def redact_api_key(value):
    text = str(value)
    return text.replace(api_key, "REDACTED") if api_key else text

def usage_token_counts(usage):
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    counts = (prompt, completion)
    if not all(
        isinstance(count, (int, float))
        and not isinstance(count, bool)
        and math.isfinite(count)
        and count >= 0
        for count in counts
    ):
        return None
    return counts

def request_json(method, url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

try:
    models = request_json("GET", base + models_path)
    ids = [x.get("id") for x in models.get("data", []) if isinstance(x, dict)]
    if ids:
        if model in ids:
            print(f"  /models: found '{redact_api_key(model)}'")
        else:
            print(f"  WARNING: '{redact_api_key(model)}' not listed by /models")
            print("  Available model ids:", redact_api_key(", ".join(str(x) for x in ids[:20])))
    else:
        print("  WARNING: /models returned no model ids")
except Exception as exc:
    print(f"  WARNING: /models probe failed: {redact_api_key(exc)}")

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "stream": False,
}
if legacy == "1":
    payload["max_tokens"] = 8
else:
    payload["max_completion_tokens"] = 8

try:
    response = request_json("POST", base + chat_path, payload, timeout=60)
except urllib.error.HTTPError as exc:
    body = redact_api_key(exc.read().decode(errors="replace"))
    print(f"  ERROR: chat probe returned HTTP {exc.code}: {body[:1000]}", file=sys.stderr)
    raise SystemExit(1)
except Exception as exc:
    print(f"  ERROR: chat probe failed: {redact_api_key(exc)}", file=sys.stderr)
    raise SystemExit(1)

usage = response.get("usage")
counts = usage_token_counts(usage)
if counts is None:
    print("  WARNING: response has no usable 'usage' token counts.")
    print("           --use-server-token-count will not provide cost-accounting totals.")
else:
    print("  usage:", redact_api_key(json.dumps(usage, sort_keys=True)))
    prompt, completion = counts
    cached = None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    print(f"  prompt/input tokens: {redact_api_key(prompt)}")
    print(f"  completion/output tokens: {redact_api_key(completion)}")
    if cached is None:
        print("  cached tokens: not reported (cache multiplier cannot be fitted unless cache runs report them)")
    else:
        print(f"  cached tokens: {redact_api_key(cached)}")

# Streaming probe: --streaming + --use-server-token-count needs usage on
# streamed responses, which only works if the server honors
# stream_options: {"include_usage": true}. The non-streaming probe above
# cannot detect a missing include_usage implementation.
stream_payload = dict(payload)
stream_payload["stream"] = True
stream_payload["stream_options"] = {"include_usage": True}
saw_usage = False
try:
    data = json.dumps(stream_payload).encode()
    req = urllib.request.Request(base + chat_path, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and usage_token_counts(obj.get("usage")) is not None:
                saw_usage = True
except urllib.error.HTTPError as exc:
    body = redact_api_key(exc.read().decode(errors="replace"))
    print(f"  WARNING: streamed probe returned HTTP {exc.code}: {body[:400]}")
    print("           The server may reject stream_options/include_usage.")
except Exception as exc:
    print(f"  WARNING: streamed probe failed: {redact_api_key(exc)}")
else:
    if saw_usage:
        print("  streamed usage: present (stream_options.include_usage honored)")
    else:
        print("  WARNING: streamed response completed without usable token counts.")
        print("           --use-server-token-count will not provide cost-accounting totals.")
PY
}

if [[ "$SKIP_PROBE" -eq 0 ]]; then
  probe_endpoint
fi

# Split and normalize the concurrency list (trims spaces, drops empty
# entries so "4, 8,,16" does not produce garbage levels or empty dirs).
IFS=',' read -r -a CONCURRENCIES_RAW <<<"$CONCURRENCY_LIST"
CONCURRENCIES=()
for c in "${CONCURRENCIES_RAW[@]}"; do
  c="${c//[[:space:]]/}"
  if [[ -n "$c" ]]; then
    CONCURRENCIES+=("$c")
  fi
done

run_one() {
  local name="$1"
  local isl="$2"
  local osl="$3"
  local prefix="$4"
  local concurrency="$5"

  local run_dir="${OUT_DIR}/runs/${name}/c${concurrency}"
  mkdir -p "$run_dir"

  cat >"${run_dir}/run_context.json" <<EOF
{
  "profile": "$name",
  "isl": $isl,
  "osl": $osl,
  "prefix_tokens": $prefix,
  "concurrency": $concurrency,
  "requested_duration_seconds": $DURATION
}
EOF

  local warmup="$concurrency"
  if ((warmup < 4)); then
    warmup=4
  fi

  local -a cmd=(
    "$AIPERF_BIN" profile
    --artifact-dir "$run_dir"
    --model "$MODEL"
    --tokenizer "$TOKENIZER"
    --endpoint-type chat
    --endpoint "$ENDPOINT"
    --url "$URL"
    --streaming
    --use-server-token-count
    --synthetic-input-tokens-mean "$isl"
    --synthetic-input-tokens-stddev 0
    --output-tokens-mean "$osl"
    --output-tokens-stddev 0
    --concurrency "$concurrency"
    --benchmark-duration "$DURATION"
    --benchmark-grace-period "$GRACE_PERIOD"
    --warmup-request-count "$warmup"
    --num-dataset-entries "$NUM_DATASET_ENTRIES"
    --random-seed "$RANDOM_SEED"
    --no-server-metrics
    --export-level summary
    --ui simple
  )

  if [[ -n "$TOKENIZER_REVISION" ]]; then
    cmd+=(--tokenizer-revision "$TOKENIZER_REVISION")
  fi

  if [[ "$TOKENIZER_TRUST_REMOTE_CODE" -eq 1 ]]; then
    cmd+=(--tokenizer-trust-remote-code)
  fi

  if [[ "$APPLY_CHAT_TEMPLATE" -eq 1 ]]; then
    cmd+=(--apply-chat-template)
  fi

  if [[ -n "$API_KEY" ]]; then
    cmd+=(--api-key "$API_KEY")
  fi

  if [[ "$LEGACY_MAX_TOKENS" -eq 1 ]]; then
    cmd+=(--use-legacy-max-tokens)
  fi

  if ((prefix > 0)); then
    # Pool size 1 is intentional: after warmup, this gives a deterministic
    # repeatedly-used prefix and maximizes the chance of observable cache hits.
    cmd+=(
      --prefix-prompt-length "$prefix"
      --prefix-prompt-pool-size 1
    )
  fi

  if [[ ${#EXTRA_AIPERF_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_AIPERF_ARGS[@]}")
  fi

  echo
  echo "======================================================================"
  echo "Profile=$name ISL=$isl OSL=$osl prefix=$prefix concurrency=$concurrency"
  # Print the command for reproducibility, but redact the API key value so
  # secrets do not end up in terminal scrollback or captured logs.
  printf 'Command:'
  local prev=""
  local arg
  for arg in "${cmd[@]}"; do
    if [[ "$prev" == "--api-key" ]]; then
      printf ' %q' "REDACTED"
    elif [[ "$arg" == --api-key=* ]]; then
      printf ' %q' "--api-key=REDACTED"
    else
      printf ' %q' "$arg"
    fi
    prev="$arg"
  done
  printf '\n'
  echo "======================================================================"

  "${cmd[@]}"
}

echo
echo "Benchmark plan: $DEFAULT_PLAN"
cat "$DEFAULT_PLAN"
echo

# The final '|| [[ -n "$name" ]]' keeps a last row without a trailing
# newline from being silently dropped.
while IFS=',' read -r name isl osl prefix || [[ -n "$name" ]]; do
  # Strip whitespace first so " name, ..." or indented rows still match
  # the header/comment guards below.
  name="${name//[[:space:]]/}"
  isl="${isl//[[:space:]]/}"
  osl="${osl//[[:space:]]/}"
  prefix="${prefix//[[:space:]]/}"

  # Skip header and empty/comment lines.
  [[ "$name" == "name" ]] && continue
  [[ -z "$name" ]] && continue
  [[ "${name:0:1}" == "#" ]] && continue

  [[ "$isl" =~ ^[0-9]+$ ]] || die "invalid ISL for '$name': $isl"
  [[ "$osl" =~ ^[0-9]+$ ]] || die "invalid OSL for '$name': $osl"
  [[ "$prefix" =~ ^[0-9]+$ ]] || die "invalid prefix_tokens for '$name': $prefix"

  if ((prefix > 0 && RUN_CACHE_TESTS == 0)); then
    echo "Skipping cache profile '$name' (--no-cache-tests)"
    continue
  fi

  if ((MAX_CONTEXT > 0 && isl + prefix + osl > MAX_CONTEXT)); then
    echo "Skipping '$name': intended ISL+prefix+OSL=$((isl + prefix + osl)) > max context $MAX_CONTEXT"
    continue
  fi

  for concurrency in "${CONCURRENCIES[@]}"; do
    run_one "$name" "$isl" "$osl" "$prefix" "$concurrency"
  done
done <"$DEFAULT_PLAN"

# Parse results, select one capacity point per workload, and fit coefficients.
"$PYTHON_BIN" - "$OUT_DIR" "$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}" <<'PY'
import csv
import json
import math
import pathlib
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])
ttft_limit = float(sys.argv[2])
itl_limit = float(sys.argv[3])
hour_cost = float(sys.argv[4]) if sys.argv[4] else None

def scalar_metric(data, name, stat="avg"):
    value = data.get(name)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        v = value.get(stat)
        if v is None and stat == "avg":
            # Defensive fallbacks for schema variations.
            for k in ("value", "sum"):
                if isinstance(value.get(k), (int, float)):
                    return float(value[k])
            return None
        if isinstance(v, (int, float)):
            return float(v)
    return None

def extract_error_count(data):
    """Number of errored requests in a run.

    AIPerf 0.12 exports an ``error_summary`` array instead of an
    ``error_request_count`` metric, so derive the count from there and fall
    back to the metric name in case future versions add it.
    """
    v = scalar_metric(data, "error_request_count")
    if v is not None:
        return v
    summary = data.get("error_summary")
    if isinstance(summary, list):
        total = 0
        for entry in summary:
            if isinstance(entry, dict) and isinstance(entry.get("count"), (int, float)):
                total += entry["count"]
            else:
                total += 1
        return float(total)
    return None

def load_summary(run_dir):
    candidates = [
        run_dir / "profile_export_aiperf.json",
        run_dir / "profile_export.json",
    ]
    for p in candidates:
        if p.exists():
            return p, json.loads(p.read_text())
    # Support --profile-export-prefix supplied through extra AIPerf args.
    for p in sorted(run_dir.glob("*.json")):
        if p.name in {"run_context.json"} or "timeslice" in p.name or "server_metrics" in p.name:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and (
            "benchmark_duration" in data
            or "total_usage_prompt_tokens" in data
            or "request_throughput" in data
        ):
            return p, data
    return None, None

rows = []
for ctx_path in sorted(root.glob("runs/*/c*/run_context.json")):
    ctx = json.loads(ctx_path.read_text())
    summary_path, data = load_summary(ctx_path.parent)
    if data is None:
        print(f"WARNING: no AIPerf summary JSON in {ctx_path.parent}", file=sys.stderr)
        continue

    prompt = scalar_metric(data, "total_usage_prompt_tokens")
    completion = scalar_metric(data, "total_usage_completion_tokens")
    cached_raw = scalar_metric(data, "total_usage_prompt_cache_read_tokens")
    cache_reported = cached_raw is not None
    cached = cached_raw if cached_raw is not None else 0.0

    # If server usage is missing, this run cannot support API-based accounting.
    usage_ok = prompt is not None and completion is not None
    noncached = max(0.0, prompt - cached) if prompt is not None else None

    duration = scalar_metric(data, "benchmark_duration")
    req_tput = scalar_metric(data, "request_throughput")
    out_tput = scalar_metric(data, "output_token_throughput")
    total_tput = scalar_metric(data, "total_token_throughput")
    reasoning = scalar_metric(data, "total_usage_reasoning_tokens") or 0.0

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
        "request_count": scalar_metric(data, "request_count"),
        "error_request_count": extract_error_count(data),
    }

    if duration and usage_ok and duration > 0:
        row["api_prompt_tps"] = prompt / duration
        row["api_completion_tps"] = completion / duration
        row["api_total_tps"] = (prompt + completion) / duration
    else:
        row["api_prompt_tps"] = None
        row["api_completion_tps"] = None
        row["api_total_tps"] = None

    rows.append(row)

if not rows:
    raise SystemExit("No benchmark summaries could be parsed.")

columns = [
    "profile", "isl", "osl", "prefix_tokens", "concurrency",
    "usage_ok", "cache_reported",
    "prompt_tokens", "noncached_input_tokens", "cached_input_tokens",
    "completion_tokens", "reasoning_tokens",
    "benchmark_duration_s",
    "request_throughput_rps",
    "api_prompt_tps", "api_completion_tps", "api_total_tps",
    "output_token_throughput_tps", "total_token_throughput_tps",
    "ttft_avg_ms", "ttft_p95_ms", "ttft_p99_ms",
    "itl_avg_ms", "itl_p95_ms", "itl_p99_ms",
    "request_latency_p99_ms", "request_count", "error_request_count",
    "summary_json",
]

def write_csv(path, records):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

write_csv(root / "summary.csv", rows)

def finite(v):
    return v is not None and math.isfinite(float(v))

def meets_slo(r):
    if not r["usage_ok"]:
        return False
    if ttft_limit > 0:
        if not finite(r["ttft_p99_ms"]) or r["ttft_p99_ms"] > ttft_limit:
            return False
    if itl_limit > 0:
        if not finite(r["itl_p99_ms"]) or r["itl_p99_ms"] > itl_limit:
            return False
    return True

# Pick the highest request-throughput concurrency within each workload profile
# among error-free runs with valid usage that satisfy the requested SLOs.
# Because request size is fixed within a profile, request throughput is a clean
# capacity comparator inside that group.
grouped = defaultdict(list)
for r in rows:
    grouped[r["profile"]].append(r)

selected = []
for profile, candidates in sorted(grouped.items()):
    # Runs containing errored requests have distorted usage totals: the API
    # still counts tokens up to the failure point while latency/throughput
    # reflects truncated work. Exclude them from capacity selection.
    errored = [r for r in candidates if (r.get("error_request_count") or 0) > 0]
    if errored:
        print(
            f"WARNING: '{profile}': excluding {len(errored)} run(s) with "
            "errored requests from capacity selection",
            file=sys.stderr,
        )
    feasible = [
        r for r in candidates
        if (r.get("error_request_count") or 0) == 0 and meets_slo(r)
    ]
    if not feasible:
        print(f"WARNING: no error-free/SLO-feasible/API-usage-valid run for profile '{profile}'", file=sys.stderr)
        continue
    feasible.sort(
        key=lambda r: (
            r["request_throughput_rps"] if finite(r["request_throughput_rps"]) else -1,
            r["concurrency"],
        )
    )
    chosen = feasible[-1]
    max_tested = max(r["concurrency"] for r in candidates)
    if chosen["concurrency"] == max_tested:
        print(
            f"NOTE: '{profile}' best point is at the max tested concurrency "
            f"({max_tested}); saturation may not be reached. "
            "Consider extending --concurrency upward.",
            file=sys.stderr,
        )
        chosen["selection_note"] = "at_sweep_edge"
    selected.append(chosen)

write_csv(root / "selected_capacity_points.csv", selected)

fit_result = {
    "model": json.loads((root / "run_config.json").read_text()).get("model"),
    "selection": {
        "ttft_p99_ms_max": ttft_limit or None,
        "itl_p99_ms_max": itl_limit or None,
        "rule": (
            "highest request throughput per workload among error-free, "
            "API-usage-valid runs satisfying enabled SLOs"
        ),
        "selected_profiles": [
            {"profile": r["profile"], "concurrency": r["concurrency"]}
            for r in selected
        ],
    },
    "notes": [
        "completion_tokens already includes provider-billed completion/reasoning tokens in OpenAI-style usage; reasoning_tokens is recorded only as a diagnostic and is not added again.",
        "cached_input_tokens is only measurable when the endpoint reports usage.prompt_tokens_details.cached_tokens.",
        "The fitted model is an additive approximation of a continuously-batched inference system; inspect residual error before using it for billing.",
    ],
}

# Fit a non-negative least-squares model by enumerating active variable sets.
# This avoids a scipy dependency. numpy is normally part of an AIPerf install.
try:
    import numpy as np
except Exception as exc:
    fit_result["fit_error"] = f"numpy unavailable: {exc}"
else:
    fit_rows = [
        r for r in selected
        if r["usage_ok"]
        and finite(r["benchmark_duration_s"])
        and r["benchmark_duration_s"] > 0
        and finite(r["noncached_input_tokens"])
        and finite(r["completion_tokens"])
    ]

    have_cache = any(
        r["cache_reported"] and r["cached_input_tokens"] > 0
        for r in fit_rows
    )

    names = ["noncached_input", "output"]
    if have_cache:
        names.append("cached_input")

    if len(fit_rows) < len(names):
        fit_result["fit_error"] = (
            f"not enough selected workload points: {len(fit_rows)} rows for "
            f"{len(names)} coefficients"
        )
    else:
        X_rows = []
        y = []
        for r in fit_rows:
            x = [r["noncached_input_tokens"], r["completion_tokens"]]
            if have_cache:
                x.append(r["cached_input_tokens"])
            X_rows.append(x)
            # Resource allocation seconds. Multiplying by GPU count would scale
            # every coefficient equally and would not change the multipliers.
            y.append(r["benchmark_duration_s"])

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y, dtype=float)

        best = None
        p = X.shape[1]
        for mask in range(1, 1 << p):
            cols = [i for i in range(p) if mask & (1 << i)]
            beta_sub, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
            if np.any(beta_sub < -1e-15):
                continue
            beta_sub = np.maximum(beta_sub, 0.0)
            beta = np.zeros(p)
            beta[cols] = beta_sub
            residual = y - X @ beta
            rss = float(residual @ residual)
            if best is None or rss < best[0]:
                best = (rss, beta)

        if best is None:
            fit_result["fit_error"] = "no non-negative coefficient solution found"
        else:
            rss, beta = best
            pred = X @ beta
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            mean_y = float(np.mean(y))
            rel_rmse = rmse / mean_y if mean_y else None

            coeff = dict(zip(names, [float(v) for v in beta]))
            a = coeff.get("noncached_input", 0.0)
            b = coeff.get("output", 0.0)
            c = coeff.get("cached_input")

            fit_result["fit"] = {
                "equation": (
                    "resource_seconds ~= "
                    "a*noncached_input_tokens + b*output_tokens"
                    + (" + c*cached_input_tokens" if have_cache else "")
                ),
                "coefficients_resource_seconds_per_token": {
                    "a_noncached_input": a,
                    "b_output": b,
                    "c_cached_input": c,
                },
                "relative_rmse": rel_rmse,
                "rmse_seconds": rmse,
                "rows_used": len(fit_rows),
                "cache_coefficient_fitted": have_cache,
            }

            if a > 0:
                multipliers = {
                    "inputMultiplier": 1.0,
                    "outputMultiplier": b / a,
                    "cachedMultiplier": (c / a) if c is not None else None,
                }
                fit_result["multipliers"] = multipliers

                if hour_cost is not None:
                    usd_per_resource_second = hour_cost / 3600.0
                    input_per_m = a * usd_per_resource_second * 1_000_000
                    output_per_m = b * usd_per_resource_second * 1_000_000
                    cache_per_m = (
                        c * usd_per_resource_second * 1_000_000
                        if c is not None else None
                    )
                    fit_result["resource_hour_cost_usd"] = hour_cost
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
                            if c is not None else ""
                        )
                    )
            else:
                fit_result["fit_error"] = (
                    "fitted noncached-input coefficient is zero; workload matrix "
                    "does not identify an input baseline well enough"
                )

(root / "pricing_fit.json").write_text(json.dumps(fit_result, indent=2) + "\n")

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
    print("Fitted multipliers:")
    print(f"  inputMultiplier  = {m['inputMultiplier']:.6g}")
    print(f"  outputMultiplier = {m['outputMultiplier']:.6g}")
    if m["cachedMultiplier"] is None:
        print("  cachedMultiplier = not available (cache tokens not reported/fitted)")
    else:
        print(f"  cachedMultiplier = {m['cachedMultiplier']:.6g}")

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

if fit_result.get("fit", {}).get("relative_rmse") is not None:
    print()
    print(f"Fit relative RMSE: {100*fit_result['fit']['relative_rmse']:.2f}%")
    if fit_result["fit"]["relative_rmse"] > 0.15:
        print("WARNING: fit error > 15%; the additive token-cost model may be too simple")
        print("         for these workload points. Inspect summary.csv before billing.")
if "fit_error" in fit_result:
    print()
    print("Fit warning:", fit_result["fit_error"])

print()
print("Note: these are deployment-capacity coefficients, not external provider prices.")
PY

echo
echo "Artifacts written to: $OUT_DIR"
