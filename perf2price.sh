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
#        T = measured_duration - 0.5 * max(0, measured_duration - requested)
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
RUN_PY="${SCRIPT_DIR}/perf2price_run.py"

URL=""
MODEL=""
TOKENIZER=""
TOKENIZER_REVISION=""
TOKENIZER_TRUST_REMOTE_CODE=0
APPLY_CHAT_TEMPLATE=0
ENDPOINT="/v1/chat/completions"
MODELS_ENDPOINT="/v1/models"
API_KEY="${OPENAI_API_KEY:-}"
USE_PRIMARY_API_KEY=1

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
HTTP_CONNECTION_LIMIT=""
LOCK_HELD=0
RESUME_DIR=""
RESUME_MODE=0
RESUME_NO_API_KEY=0
RESUME_STREAM=""
POINTS_STREAM=""
INITIAL_OPTIONS_SEEN=()

usage() {
	cat <<'EOF'
Usage:
  perf2price.sh --url URL --model MODEL [options]
  perf2price.sh --resume RUN_DIR [--api-key KEY | --no-api-key]

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
  --resume DIR               Resume an interrupted run from its artifact directory
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
  --no-api-key               Confirm that a legacy saved run needs no credentials
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

release_run_lock() {
	if [[ "$LOCK_HELD" -eq 1 ]]; then
		"$PYTHON_BIN" "$RUN_PY" release-lock "$OUT_DIR" "$$" || true
		LOCK_HELD=0
	fi
}

cleanup() {
	release_run_lock
	if [[ -n "$RESUME_STREAM" && -e "$RESUME_STREAM" ]]; then
		rm -f -- "$RESUME_STREAM"
	fi
	if [[ -n "$POINTS_STREAM" && -e "$POINTS_STREAM" ]]; then
		rm -f -- "$POINTS_STREAM"
	fi
}

trap cleanup EXIT

while [[ $# -gt 0 ]]; do
	case "$1" in
	--resume | --api-key | --api-key=* | --no-api-key | -h | --help | --) ;;
	*) INITIAL_OPTIONS_SEEN+=("$1") ;;
	esac
	case "$1" in
	--resume)
		[[ -z "$RESUME_DIR" ]] || die "--resume may only be specified once"
		RESUME_DIR="${2:?missing value for --resume}"
		shift 2
		;;
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
	--no-api-key)
		RESUME_NO_API_KEY=1
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

if [[ -n "$RESUME_DIR" ]]; then
	RESUME_MODE=1
	OUT_DIR="$RESUME_DIR"
	if [[ ${#INITIAL_OPTIONS_SEEN[@]} -gt 0 ]]; then
		die "--resume cannot be combined with: ${INITIAL_OPTIONS_SEEN[*]}"
	fi
	if [[ ${#EXTRA_AIPERF_ARGS[@]} -gt 0 ]]; then
		die "--resume restores saved AIPerf arguments; new arguments after -- are not allowed"
	fi
else
	[[ "$RESUME_NO_API_KEY" -eq 0 ]] || die "--no-api-key requires --resume"
	[[ -n "$URL" ]] || die "--url is required"
	[[ -n "$MODEL" ]] || die "--model is required"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
[[ -f "$FIT_PY" ]] || die "fit module not found: $FIT_PY"
[[ -f "$RUN_PY" ]] || die "run-state module not found: $RUN_PY"

SAVED_AIPERF_VERSION=""
SAVED_API_KEY_REQUIRED=""
if [[ "$RESUME_MODE" -eq 1 ]]; then
	RESUME_STREAM="$(mktemp "${TMPDIR:-/tmp}/perf2price-resume.XXXXXX")"
	if ! "$PYTHON_BIN" "$RUN_PY" emit-config "$OUT_DIR" >"$RESUME_STREAM"; then
		die "cannot restore saved run configuration from $OUT_DIR"
	fi
	while IFS= read -r -d '' key && IFS= read -r -d '' value; do
		case "$key" in
		config_schema_version) ;;
		url) URL="$value" ;;
		endpoint) ENDPOINT="$value" ;;
		models_endpoint) MODELS_ENDPOINT="$value" ;;
		model) MODEL="$value" ;;
		tokenizer) TOKENIZER="$value" ;;
		tokenizer_revision) TOKENIZER_REVISION="$value" ;;
		concurrency_list) CONCURRENCY_LIST="$value" ;;
		duration_seconds) DURATION="$value" ;;
		grace_period) GRACE_PERIOD="$value" ;;
		ttft_p99_ms) TTFT_P99_MS="$value" ;;
		itl_p99_ms) ITL_P99_MS="$value" ;;
		resource_hour_cost_usd) RESOURCE_HOUR_COST="$value" ;;
		random_seed) RANDOM_SEED="$value" ;;
		num_dataset_entries) NUM_DATASET_ENTRIES="$value" ;;
		tokenizer_trust_remote_code) TOKENIZER_TRUST_REMOTE_CODE="$value" ;;
		apply_chat_template) APPLY_CHAT_TEMPLATE="$value" ;;
		legacy_max_tokens) LEGACY_MAX_TOKENS="$value" ;;
		run_cache_tests) RUN_CACHE_TESTS="$value" ;;
		skip_probe) SKIP_PROBE="$value" ;;
		max_context) MAX_CONTEXT="$value" ;;
		aiperf_version) SAVED_AIPERF_VERSION="$value" ;;
		api_key_required) SAVED_API_KEY_REQUIRED="$value" ;;
		http_connection_limit) HTTP_CONNECTION_LIMIT="$value" ;;
		extra_aiperf_arg) EXTRA_AIPERF_ARGS+=("$value") ;;
		*) die "unexpected saved configuration field: $key" ;;
		esac
	done <"$RESUME_STREAM"
	rm -f -- "$RESUME_STREAM"
	RESUME_STREAM=""
	if [[ "$RESUME_NO_API_KEY" -eq 1 && -n "$API_KEY" ]]; then
		die "--no-api-key conflicts with --api-key or OPENAI_API_KEY"
	fi

	SAVED_EXTRA_API_KEY=0
	for ((i = 0; i < ${#EXTRA_AIPERF_ARGS[@]}; i++)); do
		arg="${EXTRA_AIPERF_ARGS[$i]}"
		if [[ "$arg" == "--api-key" ]]; then
			((i + 1 < ${#EXTRA_AIPERF_ARGS[@]})) ||
				die "saved AIPerf arguments end with --api-key and cannot be resumed"
			[[ -n "$API_KEY" ]] ||
				die "resume requires --api-key or OPENAI_API_KEY to restore saved AIPerf credentials"
			EXTRA_AIPERF_ARGS[i + 1]="$API_KEY"
			SAVED_EXTRA_API_KEY=1
			i=$((i + 1))
		elif [[ "$arg" == --api-key=* ]]; then
			[[ -n "$API_KEY" ]] ||
				die "resume requires --api-key or OPENAI_API_KEY to restore saved AIPerf credentials"
			EXTRA_AIPERF_ARGS[i]="--api-key=$API_KEY"
			SAVED_EXTRA_API_KEY=1
		fi
	done
	if [[ "$SAVED_API_KEY_REQUIRED" == "1" && -z "$API_KEY" ]]; then
		die "resume requires --api-key or OPENAI_API_KEY for this authenticated run"
	fi
	if [[ "$SAVED_API_KEY_REQUIRED" == "1" && "$RESUME_NO_API_KEY" -eq 1 ]]; then
		die "--no-api-key cannot resume a saved authenticated run"
	fi
	if [[ -z "$SAVED_API_KEY_REQUIRED" && -z "$API_KEY" && "$RESUME_NO_API_KEY" -eq 0 ]]; then
		die "legacy resume requires --api-key, OPENAI_API_KEY, or --no-api-key"
	fi
	if [[ "$SAVED_EXTRA_API_KEY" -eq 1 && "$SAVED_API_KEY_REQUIRED" != "1" ]]; then
		USE_PRIMARY_API_KEY=0
	fi
fi

URL="${URL%/}"
TOKENIZER="${TOKENIZER:-$MODEL}"

command -v "$AIPERF_BIN" >/dev/null 2>&1 || die "AIPerf not found: $AIPERF_BIN"
AIPERF_VERSION="$("$AIPERF_BIN" --version 2>/dev/null | head -n 1 || true)"
if [[ "$RESUME_MODE" -eq 1 && -n "$SAVED_AIPERF_VERSION" && "$AIPERF_VERSION" != "$SAVED_AIPERF_VERSION" ]]; then
	die "AIPerf version mismatch: saved '$SAVED_AIPERF_VERSION', current '$AIPERF_VERSION'"
fi

if [[ "$RESUME_MODE" -eq 0 ]]; then
	mkdir -p "$OUT_DIR"
else
	DEFAULT_PLAN="${OUT_DIR}/benchmark_plan.csv"
	"$PYTHON_BIN" "$RUN_PY" validate-plan "$DEFAULT_PLAN"
fi
"$PYTHON_BIN" "$RUN_PY" acquire-lock "$OUT_DIR" "$$"
LOCK_HELD=1

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
if [[ "$RESUME_MODE" -eq 0 ]]; then
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
	"$PYTHON_BIN" "$RUN_PY" validate-plan "$DEFAULT_PLAN"
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

if [[ "$RESUME_MODE" -eq 0 ]]; then
	HTTP_CONNECTION_LIMIT="${AIPERF_HTTP_CONNECTION_LIMIT:-$((MAX_CONCURRENCY + 64))}"
fi
[[ "$HTTP_CONNECTION_LIMIT" =~ ^[0-9]+$ ]] ||
	die "saved AIPerf HTTP connection limit must be a positive integer"
((10#$HTTP_CONNECTION_LIMIT > 0)) ||
	die "saved AIPerf HTTP connection limit must be a positive integer"
export AIPERF_HTTP_CONNECTION_LIMIT="$HTTP_CONNECTION_LIMIT"

# Write run_config.json via Python so values are properly JSON-escaped
# (MODEL/URL may contain characters that would corrupt a raw heredoc).
if [[ "$RESUME_MODE" -eq 0 ]]; then
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

	API_KEY_REQUIRED=0
	if [[ -n "$API_KEY" ]]; then
		API_KEY_REQUIRED=1
	fi

	"$PYTHON_BIN" "$RUN_PY" write-config "${OUT_DIR}/run_config.json" \
		"$URL" "$ENDPOINT" "$MODELS_ENDPOINT" "$MODEL" "$TOKENIZER" \
		"$TOKENIZER_REVISION" "$CONCURRENCY_LIST" "$DURATION" "$GRACE_PERIOD" \
		"$TTFT_P99_MS" "$ITL_P99_MS" "${RESOURCE_HOUR_COST:-}" \
		"$RANDOM_SEED" "$NUM_DATASET_ENTRIES" \
		"$TOKENIZER_TRUST_REMOTE_CODE" "$APPLY_CHAT_TEMPLATE" \
		"$LEGACY_MAX_TOKENS" "$RUN_CACHE_TESTS" "$SKIP_PROBE" "$MAX_CONTEXT" \
		"${PLAN_FILE:-}" "${AIPERF_VERSION:-}" "$API_KEY_REQUIRED" \
		"$HTTP_CONNECTION_LIMIT" "$EXTRA_JSON"
fi

POINTS_STREAM="$(mktemp "${TMPDIR:-/tmp}/perf2price-points.XXXXXX")"
"$PYTHON_BIN" "$RUN_PY" emit-points "$OUT_DIR" >"$POINTS_STREAM"

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

if [[ "$RESUME_MODE" -eq 1 ]]; then
	"$PYTHON_BIN" "$RUN_PY" archive-unexpected "$OUT_DIR"
fi

POINT_STATE=""
POINT_REASON=""
classify_point() {
	local name="$1"
	local isl="$2"
	local osl="$3"
	local prefix="$4"
	local concurrency="$5"
	local result
	result="$("$PYTHON_BIN" "$RUN_PY" classify-point \
		"$OUT_DIR" "$name" "$isl" "$osl" "$prefix" "$concurrency" \
		"$DURATION" "$HTTP_CONNECTION_LIMIT")"
	IFS=$'\t' read -r POINT_STATE POINT_REASON <<<"$result"
	case "$POINT_STATE" in
	complete | retryable | not_started) ;;
	*) die "invalid point state returned for profile=$name concurrency=$concurrency: $POINT_STATE" ;;
	esac
}

if [[ "$RESUME_MODE" -eq 1 ]]; then
	RESUME_COMPLETE=0
	RESUME_RETRYABLE=0
	RESUME_NOT_STARTED=0
	while IFS=$'\t' read -r name isl osl prefix concurrency _point_duration _point_limit; do
		classify_point "$name" "$isl" "$osl" "$prefix" "$concurrency"
		case "$POINT_STATE" in
		complete) RESUME_COMPLETE=$((RESUME_COMPLETE + 1)) ;;
		retryable) RESUME_RETRYABLE=$((RESUME_RETRYABLE + 1)) ;;
		not_started) RESUME_NOT_STARTED=$((RESUME_NOT_STARTED + 1)) ;;
		esac
	done <"$POINTS_STREAM"
	echo "Resume status: $RESUME_COMPLETE complete, $RESUME_RETRYABLE retryable, $RESUME_NOT_STARTED not started"
fi

run_one() {
	local name="$1"
	local isl="$2"
	local osl="$3"
	local prefix="$4"
	local concurrency="$5"

	local run_dir="${OUT_DIR}/runs/${name}/c${concurrency}"
	mkdir -p "$run_dir"

	"$PYTHON_BIN" "$RUN_PY" write-context "${run_dir}/run_context.json" \
		"$name" "$isl" "$osl" "$prefix" "$concurrency" "$DURATION" \
		"$HTTP_CONNECTION_LIMIT"

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

	if [[ "$USE_PRIMARY_API_KEY" -eq 1 && -n "$API_KEY" ]]; then
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
	"$PYTHON_BIN" "$RUN_PY" run-locked "$OUT_DIR" "$$" -- "${cmd[@]}"
	rc=$?
	set -e

	"$PYTHON_BIN" "$RUN_PY" finish-context "${run_dir}/run_context.json" "$rc"

	if [[ "$rc" -ne 0 ]]; then
		echo "WARNING: AIPerf failed for profile=$name concurrency=$concurrency (exit $rc); continuing." >&2
	fi
}

echo
echo "Benchmark plan: $DEFAULT_PLAN"
cat "$DEFAULT_PLAN"
echo

while IFS=$'\t' read -r name isl osl prefix concurrency _point_duration _point_limit; do
	if [[ "$RESUME_MODE" -eq 1 ]]; then
		classify_point "$name" "$isl" "$osl" "$prefix" "$concurrency"
		case "$POINT_STATE" in
		complete)
			echo "Skipping completed point: profile=$name concurrency=$concurrency"
			continue
			;;
		retryable)
			backup="$("$PYTHON_BIN" "$RUN_PY" archive-point \
				"$OUT_DIR" "$name" "$concurrency" "$POINT_REASON")"
			echo "Retrying point: profile=$name concurrency=$concurrency reason=$POINT_REASON"
			if [[ -n "$backup" ]]; then
				echo "Preserved prior attempt: $backup"
			fi
			;;
		not_started) ;;
		esac
	fi
	run_one "$name" "$isl" "$osl" "$prefix" "$concurrency"
done <"$POINTS_STREAM"

# Parse results, select one capacity point per workload, and fit coefficients.
# Failed AIPerf points are skipped; remaining summaries are still fitted.
"$PYTHON_BIN" "$FIT_PY" "$OUT_DIR" "$TTFT_P99_MS" "$ITL_P99_MS" \
	"${RESOURCE_HOUR_COST:-}" "$POINTS_STREAM"

echo
echo "Artifacts written to: $OUT_DIR"
