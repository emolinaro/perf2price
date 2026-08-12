import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aiperf-openai-cost-bench.sh"


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
        if payload.get("stream"):
            body = b'data: {"choices": [{"delta": {"content": "OK"}}]}\n\n'
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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ReviewFindingRegressionTest(unittest.TestCase):
    def test_command_and_probe_warnings_cover_edge_forms(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIStub)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf = temp / "fake-aiperf"
            fake_aiperf.write_text(
                """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
artifact_dir = pathlib.Path(args[args.index("--artifact-dir") + 1])
summary = {
    "benchmark_duration": 1,
    "total_usage_prompt_tokens": 10,
    "total_usage_completion_tokens": 5,
    "request_throughput": 1,
    "error_summary": [],
}
(artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
"""
            )
            fake_aiperf.chmod(0o755)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\nsingle,8,4,0\n")
            secret = "sk-review-secret"
            env = {
                **os.environ,
                "AIPERF_BIN": str(fake_aiperf),
                "PYTHON_BIN": sys.executable,
            }
            result = subprocess.run(
                [
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
                    "--",
                    f"--api-key={secret}",
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout + result.stderr
            failures = []
            if result.returncode != 0:
                failures.append(f"script exited {result.returncode}:\n{output}")
            if secret in output:
                failures.append("equals-form API key reached command output")
            if "--api-key=REDACTED" not in output:
                failures.append("equals-form API key was not visibly redacted")
            if "streamed response completed without any 'usage' chunk" not in output:
                failures.append("EOF stream without usage emitted no warning")
            if "best point is at the max tested concurrency (1)" not in output:
                failures.append("single-point sweep emitted no edge warning")
            self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main()
