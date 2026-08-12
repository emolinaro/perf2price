import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import traceback
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "perf2price.sh"


class PythonBridge(BaseHTTPRequestHandler):
    lock = threading.Lock()

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        params = parse_qs(
            self.rfile.read(length).decode(),
            keep_blank_values=True,
        )
        argv = params.get("arg", [])
        code = params["code"][0]
        stdout = io.StringIO()
        stderr = io.StringIO()
        exit_code = 0
        with self.lock:
            original_argv = sys.argv
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.argv = argv
            sys.stdout = stdout
            sys.stderr = stderr
            sys.modules.setdefault("numpy", types.ModuleType("numpy"))
            try:
                exec(compile(code, "<stdin>", "exec"), {"__name__": "__main__"})
            except SystemExit as exc:
                if exc.code is None:
                    exit_code = 0
                elif isinstance(exc.code, int):
                    exit_code = exc.code
                else:
                    print(exc.code, file=stderr)
                    exit_code = 1
            except BaseException:
                traceback.print_exc(file=stderr)
                exit_code = 1
            finally:
                sys.argv = original_argv
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        body = (
            f"{exit_code}\n{stdout.getvalue()}{stderr.getvalue()}__PYTHON_BRIDGE_END__"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class OpenAIStub(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "test-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        secret = self.headers.get("Authorization", "").removeprefix("Bearer ")
        scenario = self.server.scenario
        if scenario == "chat_error" and not payload.get("stream"):
            self._send_json(400, {"error": f"rejected key {secret}"})
            return
        if payload.get("stream"):
            if scenario == "stream_error":
                self._send_json(400, {"error": f"rejected key {secret}"})
                return
            usage = {
                "malformed_usage": {"total_tokens": 5},
                "zero_alias_usage": {"input_tokens": 0, "output_tokens": 0},
            }.get(scenario)
            if usage is None:
                body = b'data: {"choices": [{"delta": {"content": "OK"}}]}\n\n'
            else:
                body = (
                    f"data: {json.dumps({'usage': usage})}\n\ndata: [DONE]\n\n"
                ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        body = json.dumps(
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            }
        ).encode()
        self._send_body(200, "application/json", body)

    def _send_json(self, status, payload):
        self._send_body(status, "application/json", json.dumps(payload).encode())

    def _send_body(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReviewFindingRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.python_bridge = ThreadingHTTPServer(("127.0.0.1", 0), PythonBridge)
        cls.python_bridge_thread = threading.Thread(
            target=cls.python_bridge.serve_forever,
            daemon=True,
        )
        cls.python_bridge_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.python_bridge.shutdown()
        cls.python_bridge.server_close()

    def run_benchmark(self, scenario="no_usage", api_key_args=(), extra_args=()):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIStub)
        server.scenario = scenario
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf = temp / "fake-aiperf"
            fake_aiperf.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
artifact_dir=""
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--artifact-dir" ]]; then
    artifact_dir="$2"
    break
  fi
  shift
done
printf '%s\n' '{"benchmark_duration":1,"total_usage_prompt_tokens":10,"total_usage_completion_tokens":5,"request_throughput":1,"error_summary":[]}' >"$artifact_dir/profile_export_aiperf.json"
"""
            )
            fake_aiperf.chmod(0o755)
            fake_python = temp / "fake-python"
            fake_python.write_text(
                f"""#!/usr/bin/env bash
set -euo pipefail
curl_args=()
for arg in "$@"; do
  curl_args+=(--data-urlencode "arg=$arg")
done
response="$(curl -sS -X POST "${{curl_args[@]}}" --data-urlencode code@- "http://127.0.0.1:{self.python_bridge.server_port}")"
response="${{response%__PYTHON_BRIDGE_END__}}"
exit_code="${{response%%$'\\n'*}}"
output="${{response#*$'\\n'}}"
printf '%s' "$output"
exit "$exit_code"
"""
            )
            fake_python.chmod(0o755)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\nsingle,8,4,0\n")
            env = {
                **os.environ,
                "AIPERF_BIN": str(fake_aiperf),
                "PYTHON_BIN": str(fake_python),
            }
            command = [
                str(SCRIPT),
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--model",
                "test-model",
                "--tokenizer",
                "builtin",
                "--concurrency",
                "1",
                "--duration",
                "1",
                "--grace-period",
                "0",
                "--num-dataset-entries",
                "1",
                "--plan",
                str(plan),
                "--out-dir",
                str(temp / "output"),
                *api_key_args,
            ]
            if extra_args:
                command.extend(["--", *extra_args])
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result

    def test_command_and_probe_warnings_cover_edge_forms(self):
        secret = "sk-review-secret"
        result = self.run_benchmark(extra_args=(f"--api-key={secret}",))
        output = result.stdout + result.stderr
        failures = []
        if result.returncode != 0:
            failures.append(f"script exited {result.returncode}:\n{output}")
        if secret in output:
            failures.append("equals-form API key reached command output")
        if "--api-key=REDACTED" not in output:
            failures.append("equals-form API key was not visibly redacted")
        if "streamed response completed without usable token counts" not in output:
            failures.append("EOF stream without usage emitted no warning")
        if "best point is at the max tested concurrency (1)" not in output:
            failures.append("single-point sweep emitted no edge warning")
        self.assertEqual([], failures)

    def test_wrapper_equals_api_key_is_accepted_without_disclosure(self):
        secret = "sk-wrapper-secret"
        result = subprocess.run(
            [str(SCRIPT), f"--api-key={secret}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        self.assertEqual(1, result.returncode, output)
        self.assertIn("--url is required", output)
        self.assertNotIn(secret, output)

    def test_probe_http_errors_redact_api_key(self):
        for scenario in ("chat_error", "stream_error"):
            with self.subTest(scenario=scenario):
                secret = f"sk-{scenario}-secret"
                result = self.run_benchmark(
                    scenario=scenario,
                    api_key_args=("--api-key", secret),
                )
                output = result.stdout + result.stderr
                self.assertNotIn(secret, output)
                self.assertIn("REDACTED", output)

    def test_streamed_usage_requires_input_and_output_counts(self):
        result = self.run_benchmark(scenario="malformed_usage")
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertNotIn("streamed usage: present", output)
        self.assertIn("without usable token counts", output)

    def test_streamed_usage_accepts_zero_alias_counts(self):
        result = self.run_benchmark(scenario="zero_alias_usage")
        output = result.stdout + result.stderr
        self.assertEqual(0, result.returncode, output)
        self.assertIn("streamed usage: present", output)


if __name__ == "__main__":
    unittest.main()
