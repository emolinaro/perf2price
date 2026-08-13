#!/usr/bin/env bash
set -euo pipefail

# perf2price - Backend-agnostic AIPerf harness for OpenAI-compatible
# chat/completions endpoints. Sweeps load, fits a per-token resource-cost
# model, and converts it to $/M-token prices for the serving allocation.
#
# It:
#   1. Probes the OpenAI-compatible endpoint.
#   2. Runs a matrix of prefill/decode/mixed/cache workloads.
#   3. Uses API-reported usage token counts (--use-server-token-count).
#   4. Sweeps concurrency at each workload.
#   5. Parses AIPerf summary JSON into summary.csv.
#   6. Selects the highest-throughput error-free, usage-valid concurrency per
#      workload, optionally under SLOs. A single AIPerf failure does not abort
#      the matrix.
#   7. Fits drain-adjusted resource seconds in rate space:
#
#        T = requested_duration + 0.5 * max(0, measured_duration - requested)
#        1 ~= a * (I / T) + b * (O / T) + c * (C / T)
#
#      Cache-plan rows without reported cache hits are excluded from the fit.
#      Reports:
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIPERF_BIN="${AIPERF_BIN:-aiperf}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FIT_PY="${SCRIPT_DIR}/perf2price_fit.py"

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
OUT_DIR="./perf2price-run-$(date +%Y%m%d-%H%M%S)"
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
  perf2price.sh --url URL --model MODEL [options]

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
  --skip-probe               Skip the API capability probe. The harness still
                             requires streamed usage for cost accounting.

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
  ./perf2price.sh \
    --url http://localhost:8000 \
    --model Qwen/Qwen3-30B-A3B \
    --concurrency 1,2,4,8,16

  # Remote/Kubernetes endpoint reachable from this machine
  ./perf2price.sh \
    --url https://llm.example.org \
    --model openai/gpt-oss-120b \
    --tokenizer openai/gpt-oss-120b \
    --concurrency 4,8,16,32,64 \
    --ttft-p99-ms 2000 \
    --itl-p99-ms 50 \
    --resource-hour-cost 15.96

  # Custom AIPerf switches
  ./perf2price.sh \
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
[[ -f "$FIT_PY" ]] || die "fit module not found: $FIT_PY"

AIPERF_VERSION="$("$AIPERF_BIN" --version 2>/dev/null | head -n 1 || true)"

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
EXTRA_JSON='[]'
if [[ ${#EXTRA_AIPERF_ARGS[@]} -gt 0 ]]; then
  EXTRA_JSON="$("$PYTHON_BIN" -c '
import json
import sys

redacted = []
redact_next = False
for arg in sys.argv[1:]:
    if redact_next:
        redacted.append("REDACTED")
        redact_next = False
    elif arg == "--api-key":
        redacted.append(arg)
        redact_next = True
    elif arg.startswith("--api-key="):
        redacted.append("--api-key=REDACTED")
    else:
        redacted.append(arg)
print(json.dumps(redacted))
' "${EXTRA_AIPERF_ARGS[@]}")"
fi
"$PYTHON_BIN" - "${OUT_DIR}/run_config.json" \
  "$URL" "$ENDPOINT" "$MODELS_ENDPOINT" "$MODEL" "$TOKENIZER" \
  "$TOKENIZER_REVISION" "$CONCURRENCY_LIST" "$DURATION" "$GRACE_PERIOD" \
  "$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}" \
  "$RANDOM_SEED" "$NUM_DATASET_ENTRIES" \
  "$TOKENIZER_TRUST_REMOTE_CODE" "$APPLY_CHAT_TEMPLATE" \
  "$LEGACY_MAX_TOKENS" "$RUN_CACHE_TESTS" "$SKIP_PROBE" "$MAX_CONTEXT" \
  "${PLAN_FILE:-}" "${AIPERF_VERSION:-}" "$EXTRA_JSON" <<'PY'
import json
import sys

(path, url, endpoint, models_endpoint, model, tokenizer, revision,
 concurrency_list, duration, grace, ttft, itl, hourly_cost,
 seed, entries, trust_remote, apply_chat, legacy, cache_tests, skip_probe,
 max_context, plan_file, aiperf_version, extra_json) = sys.argv[1:]

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
    "tokenizer_trust_remote_code": trust_remote == "1",
    "apply_chat_template": apply_chat == "1",
    "legacy_max_tokens": legacy == "1",
    "run_cache_tests": cache_tests == "1",
    "skip_probe": skip_probe == "1",
    "max_context": int(max_context),
    "plan_file": plan_file or None,
    "concurrency_list": concurrency_list,
    "duration_seconds": num(duration),
    "grace_period": grace,
    "ttft_p99_ms": num(ttft),
    "itl_p99_ms": num(itl),
    "resource_hour_cost_usd": num(hourly_cost) if hourly_cost else None,
    "random_seed": int(seed),
    "num_dataset_entries": int(entries),
    "aiperf_version": aiperf_version or None,
    "extra_aiperf_args": json.loads(extra_json) if extra_json else [],
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
    print(f"  ERROR: streamed probe returned HTTP {exc.code}: {body[:400]}", file=sys.stderr)
    print("         The server may reject stream_options/include_usage.", file=sys.stderr)
    print("         Use --skip-probe only if this endpoint was already verified.", file=sys.stderr)
    raise SystemExit(1)
except Exception as exc:
    print(f"  ERROR: streamed probe failed: {redact_api_key(exc)}", file=sys.stderr)
    print("         Use --skip-probe only if this endpoint was already verified.", file=sys.stderr)
    raise SystemExit(1)
else:
    if saw_usage:
        print("  streamed usage: present (stream_options.include_usage honored)")
    else:
        print("  ERROR: streamed response completed without usable token counts.", file=sys.stderr)
        print("         --streaming --use-server-token-count cannot account tokens.", file=sys.stderr)
        print("         Use --skip-probe only if this endpoint was already verified.", file=sys.stderr)
        raise SystemExit(1)
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

  "$PYTHON_BIN" - "${run_dir}/run_context.json" \
    "$name" "$isl" "$osl" "$prefix" "$concurrency" "$DURATION" <<'PY'
import json
import sys

path, name, isl, osl, prefix, concurrency, duration = sys.argv[1:8]
with open(path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "profile": name,
            "isl": int(isl),
            "osl": int(osl),
            "prefix_tokens": int(prefix),
            "concurrency": int(concurrency),
            "requested_duration_seconds": float(duration),
        },
        f,
        indent=2,
    )
    f.write("\n")
PY

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

  local rc
  set +e
  "${cmd[@]}"
  rc=$?
  set -e

  "$PYTHON_BIN" - "${run_dir}/run_context.json" "$rc" <<'PY'
import json
import sys

path, rc = sys.argv[1], int(sys.argv[2])
with open(path, encoding="utf-8") as f:
    ctx = json.load(f)
ctx["aiperf_exit_code"] = rc
ctx["aiperf_ok"] = rc == 0
with open(path, "w", encoding="utf-8") as f:
    json.dump(ctx, f, indent=2)
    f.write("\n")
PY

  if [[ "$rc" -ne 0 ]]; then
    echo "WARNING: AIPerf failed for profile=$name concurrency=$concurrency (exit $rc); continuing." >&2
  fi
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
# Failed AIPerf points are skipped; remaining summaries are still fitted.
"$PYTHON_BIN" "$FIT_PY" "$OUT_DIR" "$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}"

echo
echo "Artifacts written to: $OUT_DIR"
