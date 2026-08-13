import contextlib
import http.server
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

import perf2price_fit as fit


class Perf2PriceRegressionTests(unittest.TestCase):
    def write_run(
        self,
        root,
        profile,
        concurrency,
        summary,
        *,
        isl=128,
        osl=16,
        prefix_tokens=0,
        aiperf_ok=True,
    ):
        run_dir = root / "runs" / profile / f"c{concurrency}"
        run_dir.mkdir(parents=True)
        duration = summary.get("benchmark_duration", 1)
        context = {
            "profile": profile,
            "isl": isl,
            "osl": osl,
            "prefix_tokens": prefix_tokens,
            "concurrency": concurrency,
            "requested_duration_seconds": duration,
            "aiperf_exit_code": 0 if aiperf_ok else 1,
            "aiperf_ok": aiperf_ok,
        }
        (run_dir / "run_context.json").write_text(json.dumps(context))
        (run_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
        return run_dir

    def summary(self, prompt, completion, throughput, duration=1, cached=None):
        data = {
            "benchmark_duration": duration,
            "total_usage_prompt_tokens": prompt,
            "total_usage_completion_tokens": completion,
            "request_count": 1,
            "request_throughput": throughput,
            "error_summary": [],
        }
        if cached is not None:
            data["total_usage_prompt_cache_read_tokens"] = cached
        return data

    def test_run_config_redacts_extra_api_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf = temp / "aiperf"
            fake_aiperf.write_text(
                """#!/usr/bin/env python3
import json
import pathlib
import sys

if "--version" in sys.argv:
    print("aiperf 0.12.0")
    raise SystemExit(0)

artifact_dir = pathlib.Path(sys.argv[sys.argv.index("--artifact-dir") + 1])
summary = {
    "benchmark_duration": 1,
    "total_usage_prompt_tokens": 128,
    "total_usage_completion_tokens": 16,
    "request_count": 1,
    "request_throughput": 1,
    "error_summary": [],
}
(artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
"""
            )
            fake_aiperf.chmod(0o755)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\nsingle,128,16,0\n")
            out_dir = temp / "output"
            first_secret = "first-secret-value"
            second_secret = "second-secret-value"
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env["AIPERF_BIN"] = str(fake_aiperf)
            env["PYTHON_BIN"] = sys.executable
            result = subprocess.run(
                [
                    "bash",
                    str(pathlib.Path(__file__).with_name("perf2price.sh")),
                    "--url",
                    "http://localhost:8000",
                    "--model",
                    "test-model",
                    "--concurrency",
                    "1",
                    "--duration",
                    "1",
                    "--grace-period",
                    "0",
                    "--plan",
                    str(plan),
                    "--out-dir",
                    str(out_dir),
                    "--skip-probe",
                    "--",
                    "--api-key",
                    first_secret,
                    f"--api-key={second_secret}",
                ],
                cwd=pathlib.Path(__file__).parent,
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = result.stdout + result.stderr
            self.assertNotIn(first_secret, output)
            self.assertNotIn(second_secret, output)
            config = json.loads((out_dir / "run_config.json").read_text())
            self.assertEqual(
                config["extra_aiperf_args"],
                ["--api-key", "REDACTED", "--api-key=REDACTED"],
            )

    def test_malformed_primary_summary_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            bad_dir = root / "runs" / "bad" / "c1"
            bad_dir.mkdir(parents=True)
            (bad_dir / "run_context.json").write_text(
                json.dumps(
                    {
                        "profile": "bad",
                        "isl": 128,
                        "osl": 16,
                        "prefix_tokens": 0,
                        "concurrency": 1,
                        "requested_duration_seconds": 1,
                        "aiperf_ok": False,
                    }
                )
            )
            (bad_dir / "profile_export_aiperf.json").write_text('{"truncated":')
            self.write_run(root, "good", 1, self.summary(128, 16, 1))

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rows = fit.parse_rows(root)

            self.assertEqual([row["profile"] for row in rows], ["good"])
            self.assertIn("profile_export_aiperf.json", stderr.getvalue())

    def test_streaming_probe_http_error_stops_before_benchmark(self):
        class ProbeHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def write_json(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.write_json(200, {"data": [{"id": "test-model"}]})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if payload.get("stream"):
                    self.write_json(400, {"error": "include_usage unsupported"})
                else:
                    self.write_json(
                        200,
                        {
                            "usage": {
                                "prompt_tokens": 8,
                                "completion_tokens": 2,
                            }
                        },
                    )

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ProbeHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp = pathlib.Path(temp_dir)
                marker = temp / "benchmark-started"
                fake_aiperf = temp / "aiperf"
                fake_aiperf.write_text(
                    f"""#!/usr/bin/env python3
import json
import pathlib
import sys

if "--version" in sys.argv:
    print("aiperf 0.12.0")
    raise SystemExit(0)

pathlib.Path({str(marker)!r}).write_text("started")
artifact_dir = pathlib.Path(sys.argv[sys.argv.index("--artifact-dir") + 1])
summary = {{
    "benchmark_duration": 1,
    "total_usage_prompt_tokens": 128,
    "total_usage_completion_tokens": 16,
    "request_count": 1,
    "request_throughput": 1,
    "error_summary": [],
}}
(artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
"""
                )
                fake_aiperf.chmod(0o755)
                plan = temp / "plan.csv"
                plan.write_text("name,isl,osl,prefix_tokens\nsingle,128,16,0\n")
                env = os.environ.copy()
                env.pop("OPENAI_API_KEY", None)
                env["AIPERF_BIN"] = str(fake_aiperf)
                env["PYTHON_BIN"] = sys.executable
                result = subprocess.run(
                    [
                        "bash",
                        str(pathlib.Path(__file__).with_name("perf2price.sh")),
                        "--url",
                        f"http://127.0.0.1:{server.server_port}",
                        "--model",
                        "test-model",
                        "--concurrency",
                        "1",
                        "--duration",
                        "1",
                        "--grace-period",
                        "0",
                        "--plan",
                        str(plan),
                        "--out-dir",
                        str(temp / "output"),
                    ],
                    cwd=pathlib.Path(__file__).parent,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("streamed probe returned HTTP 400", result.stderr)
                self.assertFalse(marker.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_cached_tokens_above_prompt_are_ineligible_and_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            self.write_run(root, "profile", 4, self.summary(100, 16, 10))
            self.write_run(
                root,
                "profile",
                8,
                self.summary(100, 16, 20, cached=120),
            )

            with contextlib.redirect_stderr(io.StringIO()):
                rows = fit.parse_rows(root)
                selected = fit.select_capacity_points(rows, 0, 0)

            invalid = next(row for row in rows if row["concurrency"] == 8)
            self.assertEqual(invalid["usage_ok"], 0)
            self.assertEqual(invalid["cached_input_tokens"], 120)
            self.assertEqual(invalid["noncached_input_tokens"], -20)
            self.assertEqual(selected[0]["concurrency"], 4)

    def test_plateau_ignores_failed_and_errored_candidates(self):
        clean = {
            "profile": "profile",
            "concurrency": 4,
            "usage_ok": 1,
            "aiperf_ok": True,
            "error_request_count": 0,
            "request_throughput_rps": 10,
            "ttft_p99_ms": None,
            "itl_p99_ms": None,
        }
        failed = {
            **clean,
            "concurrency": 8,
            "aiperf_ok": False,
            "request_throughput_rps": 10.1,
        }
        errored = {
            **clean,
            "concurrency": 16,
            "error_request_count": 1,
            "request_throughput_rps": 10.2,
        }

        with contextlib.redirect_stderr(io.StringIO()):
            selected = fit.select_capacity_points([clean, failed, errored], 0, 0)

        self.assertEqual(selected[0]["concurrency"], 4)
        self.assertEqual(selected[0]["selection_note"], "at_sweep_edge")

    def test_rank_deficient_workloads_do_not_emit_pricing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            (root / "run_config.json").write_text(
                json.dumps({"model": "test-model", "random_seed": 1})
            )
            self.write_run(
                root,
                "first",
                1,
                self.summary(100, 50, 1, duration=10),
                osl=50,
            )
            self.write_run(
                root,
                "second",
                1,
                self.summary(200, 100, 1, duration=20),
                osl=100,
            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                result = fit.main(["perf2price_fit.py", str(root), "0", "0", ""])

            self.assertEqual(result, 0)
            pricing = json.loads((root / "pricing_fit.json").read_text())
            self.assertIn("fit_error", pricing)
            self.assertNotIn("multipliers", pricing)

    def test_zero_output_coefficient_does_not_emit_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            (root / "run_config.json").write_text(
                json.dumps({"model": "test-model", "random_seed": 1})
            )
            self.write_run(
                root,
                "input-heavy",
                1,
                self.summary(100, 10, 1, duration=10),
                osl=10,
            )
            self.write_run(
                root,
                "output-heavy",
                1,
                self.summary(50, 100, 1, duration=5),
                osl=100,
            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                result = fit.main(
                    ["perf2price_fit.py", str(root), "0", "0", "100"]
                )

            self.assertEqual(result, 0)
            pricing = json.loads((root / "pricing_fit.json").read_text())
            self.assertIn("fit_error", pricing)
            self.assertIn("output coefficient is zero", pricing["fit_error"])
            self.assertNotIn("time_model", pricing)
            self.assertNotIn("multipliers", pricing)
            self.assertNotIn("optional_cost_model", pricing)

    def test_nnls_requires_full_column_rank(self):
        X = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y = np.asarray([1.0, 2.0, 3.0])

        self.assertIsNone(fit.nnls_enumerate(X, y))

    def test_nnls_rejects_ill_conditioned_workload_ratios(self):
        X = np.asarray(
            [
                [100.0, 100.0],
                [200.0, 200.01],
                [300.0, 300.03],
                [400.0, 400.06],
            ]
        )
        y = X @ np.asarray([0.04, 0.06])

        self.assertIsNone(fit.nnls_enumerate(X, y))


if __name__ == "__main__":
    unittest.main()
