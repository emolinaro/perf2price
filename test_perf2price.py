import contextlib
import http.server
import io
import json
import os
import pathlib
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

import perf2price_fit as fit


class Perf2PriceRegressionTests(unittest.TestCase):
    def wait_for_path(self, path, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists():
                return
            time.sleep(0.02)
        self.fail(f"timed out waiting for {path}")

    def write_successful_fake_aiperf(self, root, version="aiperf 0.12.0"):
        invocation_log = root / "invocations.txt"
        fake_aiperf = root / "aiperf"
        fake_aiperf.write_text(
            f'''#!/usr/bin/env python3
import json
import pathlib
import sys

if "--version" in sys.argv:
    print({version!r})
    raise SystemExit(0)

artifact_dir = pathlib.Path(sys.argv[sys.argv.index("--artifact-dir") + 1])
profile = artifact_dir.parent.name
with pathlib.Path({str(invocation_log)!r}).open("a") as stream:
    stream.write(profile + "\\n")
isl = int(sys.argv[sys.argv.index("--synthetic-input-tokens-mean") + 1])
osl = int(sys.argv[sys.argv.index("--output-tokens-mean") + 1])
summary = {{
    "benchmark_duration": 1,
    "total_usage_prompt_tokens": isl,
    "total_usage_completion_tokens": osl,
    "request_count": 1,
    "request_throughput": 1,
    "error_summary": [],
}}
(artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
'''
        )
        fake_aiperf.chmod(0o755)
        return fake_aiperf, invocation_log

    def benchmark_env(self, fake_aiperf, api_key=None):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        if api_key is not None:
            env["OPENAI_API_KEY"] = api_key
        env["AIPERF_BIN"] = str(fake_aiperf)
        env["PYTHON_BIN"] = sys.executable
        return env

    def run_single_concurrency_benchmark(
        self, plan, out_dir, fake_aiperf, *, api_key=None
    ):
        script = pathlib.Path(__file__).with_name("perf2price.sh")
        return subprocess.run(
            [
                "bash",
                str(script),
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
            ],
            cwd=script.parent,
            env=self.benchmark_env(fake_aiperf, api_key),
            capture_output=True,
            text=True,
            timeout=30,
        )

    def snapshot_tree(self, root):
        return {
            str(path.relative_to(root)): (
                "directory" if path.is_dir() else path.read_bytes()
            )
            for path in sorted(root.rglob("*"))
        }

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

    def test_initial_run_writes_resumable_metadata(self):
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
            secret = "resume-metadata-secret"
            env = os.environ.copy()
            env["AIPERF_BIN"] = str(fake_aiperf)
            env["PYTHON_BIN"] = sys.executable
            env["OPENAI_API_KEY"] = secret

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
                ],
                cwd=pathlib.Path(__file__).parent,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            config_text = (out_dir / "run_config.json").read_text()
            config = json.loads(config_text)
            self.assertEqual(config["config_schema_version"], 1)
            self.assertTrue(config["api_key_required"])
            self.assertNotIn(secret, config_text)
            self.assertNotIn(secret, result.stdout + result.stderr)
            self.assertFalse((out_dir / ".perf2price.lock").exists())

    def test_initial_run_rejects_unsafe_profile_before_aiperf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            marker = temp / "aiperf-started"
            fake_aiperf = temp / "aiperf"
            fake_aiperf.write_text(
                f"""#!/usr/bin/env python3
import pathlib
import sys

if "--version" in sys.argv:
    print("aiperf 0.12.0")
    raise SystemExit(0)

pathlib.Path({str(marker)!r}).write_text("started")
"""
            )
            fake_aiperf.chmod(0o755)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\n../escape,128,16,0\n")
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
                    str(temp / "output"),
                    "--skip-probe",
                ],
                cwd=pathlib.Path(__file__).parent,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("invalid profile name", result.stderr)
            self.assertFalse(marker.exists())

    def test_resume_after_interrupted_point(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            invocation_log = temp / "invocations.txt"
            block_once = temp / "block-once"
            second_started = temp / "second-started"
            block_once.write_text("block")
            fake_aiperf = temp / "aiperf"
            fake_aiperf.write_text(
                f"""#!/usr/bin/env python3
import json
import pathlib
import sys
import time

if "--version" in sys.argv:
    print("aiperf 0.12.0")
    raise SystemExit(0)

artifact_dir = pathlib.Path(sys.argv[sys.argv.index("--artifact-dir") + 1])
profile = artifact_dir.parent.name
log = pathlib.Path({str(invocation_log)!r})
with log.open("a") as stream:
    stream.write(profile + "\\n")

block_once = pathlib.Path({str(block_once)!r})
if profile == "second" and block_once.exists():
    block_once.unlink()
    pathlib.Path({str(second_started)!r}).write_text("started")
    time.sleep(60)

isl = int(sys.argv[sys.argv.index("--synthetic-input-tokens-mean") + 1])
osl = int(sys.argv[sys.argv.index("--output-tokens-mean") + 1])
summary = {{
    "benchmark_duration": 1,
    "total_usage_prompt_tokens": isl,
    "total_usage_completion_tokens": osl,
    "request_count": 1,
    "request_throughput": 1,
    "error_summary": [],
}}
(artifact_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))
"""
            )
            fake_aiperf.chmod(0o755)
            plan = temp / "plan.csv"
            plan.write_text(
                "name,isl,osl,prefix_tokens\n"
                "first,128,16,0\n"
                "second,64,64,0\n"
            )
            out_dir = temp / "output"
            script = pathlib.Path(__file__).with_name("perf2price.sh")
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env["AIPERF_BIN"] = str(fake_aiperf)
            env["PYTHON_BIN"] = sys.executable
            initial_command = [
                "bash",
                str(script),
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
            ]

            process = subprocess.Popen(
                initial_command,
                cwd=script.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                self.wait_for_path(second_started, timeout=10)
                os.killpg(process.pid, signal.SIGINT)
                initial_stdout, initial_stderr = process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)

            self.assertNotEqual(
                process.returncode, 0, initial_stdout + initial_stderr
            )
            self.assertFalse((out_dir / ".perf2price.lock").exists())

            result = subprocess.run(
                ["bash", str(script), "--resume", str(out_dir)],
                cwd=script.parent,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(
                invocation_log.read_text().splitlines(),
                ["first", "second", "second"],
            )
            self.assertIn(
                "Skipping completed point: profile=first concurrency=1",
                result.stdout,
            )
            backups = list(
                (out_dir / "resume_backups" / "second" / "c1").glob("attempt-*")
            )
            self.assertEqual(len(backups), 1)
            self.assertTrue((backups[0] / "run_context.json").exists())
            for profile in ("first", "second"):
                summary = out_dir / "runs" / profile / "c1" / "profile_export_aiperf.json"
                self.assertIsInstance(json.loads(summary.read_text()), dict)
            for name in (
                "summary.csv",
                "selected_capacity_points.csv",
                "pricing_fit.json",
            ):
                self.assertTrue((out_dir / name).exists(), name)
            self.assertFalse((out_dir / ".perf2price.lock").exists())

    def test_resume_preflight_failures_do_not_mutate_saved_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf, _ = self.write_successful_fake_aiperf(temp)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\nsingle,128,16,0\n")
            baseline = temp / "baseline"
            initial = self.run_single_concurrency_benchmark(
                plan, baseline, fake_aiperf
            )
            self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
            script = pathlib.Path(__file__).with_name("perf2price.sh")

            mismatch_root = temp / "version-mismatch"
            mismatch_root.mkdir()
            mismatched_aiperf, _ = self.write_successful_fake_aiperf(
                mismatch_root, version="aiperf 0.13.0"
            )
            cases = (
                (
                    "conflicting option",
                    lambda _: None,
                    ["--duration", "2"],
                    self.benchmark_env(fake_aiperf),
                    "--resume cannot be combined",
                ),
                (
                    "missing configuration",
                    lambda root: (root / "run_config.json").unlink(),
                    [],
                    self.benchmark_env(fake_aiperf),
                    "missing saved run configuration",
                ),
                (
                    "malformed configuration",
                    lambda root: (root / "run_config.json").write_text("{"),
                    [],
                    self.benchmark_env(fake_aiperf),
                    "cannot read saved run configuration",
                ),
                (
                    "missing plan",
                    lambda root: (root / "benchmark_plan.csv").unlink(),
                    [],
                    self.benchmark_env(fake_aiperf),
                    "cannot read benchmark plan",
                ),
                (
                    "invalid plan",
                    lambda root: (root / "benchmark_plan.csv").write_text(
                        "name,isl,osl,prefix_tokens\n../escape,128,16,0\n"
                    ),
                    [],
                    self.benchmark_env(fake_aiperf),
                    "invalid profile name",
                ),
                (
                    "version mismatch",
                    lambda _: None,
                    [],
                    self.benchmark_env(mismatched_aiperf),
                    "AIPerf version mismatch",
                ),
            )

            for index, (name, mutate, extra_args, env, expected) in enumerate(cases):
                with self.subTest(name=name):
                    run_dir = temp / f"case-{index}"
                    shutil.copytree(baseline, run_dir)
                    mutate(run_dir)
                    before = self.snapshot_tree(run_dir)

                    result = subprocess.run(
                        ["bash", str(script), "--resume", str(run_dir), *extra_args],
                        cwd=script.parent,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stdout + result.stderr)
                    self.assertEqual(self.snapshot_tree(run_dir), before)

    def test_resume_requires_current_credentials_without_exposing_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf, _ = self.write_successful_fake_aiperf(temp)
            plan = temp / "plan.csv"
            plan.write_text("name,isl,osl,prefix_tokens\nsingle,128,16,0\n")
            out_dir = temp / "output"
            initial_secret = "initial-secret-value"
            initial = self.run_single_concurrency_benchmark(
                plan, out_dir, fake_aiperf, api_key=initial_secret
            )
            self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
            script = pathlib.Path(__file__).with_name("perf2price.sh")
            before = self.snapshot_tree(out_dir)

            missing = subprocess.run(
                ["bash", str(script), "--resume", str(out_dir)],
                cwd=script.parent,
                env=self.benchmark_env(fake_aiperf),
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("resume requires --api-key", missing.stderr)
            self.assertEqual(self.snapshot_tree(out_dir), before)

            mismatch_root = temp / "mismatch"
            mismatch_root.mkdir()
            mismatched_aiperf, _ = self.write_successful_fake_aiperf(
                mismatch_root, version="aiperf 0.13.0"
            )
            replacement_secret = "replacement-secret-value"
            mismatch = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--resume",
                    str(out_dir),
                    "--api-key",
                    replacement_secret,
                ],
                cwd=script.parent,
                env=self.benchmark_env(mismatched_aiperf),
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertNotEqual(mismatch.returncode, 0)
            mismatch_output = mismatch.stdout + mismatch.stderr
            self.assertIn("AIPerf version mismatch", mismatch_output)
            self.assertNotIn(replacement_secret, mismatch_output)
            self.assertEqual(self.snapshot_tree(out_dir), before)

            resumed = subprocess.run(
                [
                    "bash",
                    str(script),
                    "--resume",
                    str(out_dir),
                    "--api-key",
                    replacement_secret,
                ],
                cwd=script.parent,
                env=self.benchmark_env(fake_aiperf),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertNotIn(
                replacement_secret, resumed.stdout + resumed.stderr
            )

    def test_resume_aggregates_complete_saved_plan_without_original_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf, invocation_log = self.write_successful_fake_aiperf(temp)
            original_plan = temp / "custom-plan.csv"
            original_plan.write_text(
                "name,isl,osl,prefix_tokens\nsaved_profile,128,16,0\n"
            )
            out_dir = temp / "output"
            initial = self.run_single_concurrency_benchmark(
                original_plan, out_dir, fake_aiperf
            )
            self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
            original_plan.unlink()
            for name in (
                "summary.csv",
                "selected_capacity_points.csv",
                "pricing_fit.json",
            ):
                (out_dir / name).unlink()
            script = pathlib.Path(__file__).with_name("perf2price.sh")

            resumed = subprocess.run(
                ["bash", str(script), "--resume", str(out_dir)],
                cwd=script.parent,
                env=self.benchmark_env(fake_aiperf),
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(invocation_log.read_text().splitlines(), ["saved_profile"])
            self.assertIn("Resume status: 1 complete", resumed.stdout)
            self.assertIn("Skipping completed point", resumed.stdout)
            for name in (
                "summary.csv",
                "selected_capacity_points.csv",
                "pricing_fit.json",
            ):
                self.assertTrue((out_dir / name).exists(), name)

    def test_resume_preserves_and_retries_failed_and_malformed_points(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = pathlib.Path(temp_dir)
            fake_aiperf, invocation_log = self.write_successful_fake_aiperf(temp)
            plan = temp / "plan.csv"
            plan.write_text(
                "name,isl,osl,prefix_tokens\n"
                "failed,128,16,0\n"
                "malformed,64,64,0\n"
            )
            out_dir = temp / "output"
            initial = self.run_single_concurrency_benchmark(
                plan, out_dir, fake_aiperf
            )
            self.assertEqual(initial.returncode, 0, initial.stdout + initial.stderr)
            failed_context_path = (
                out_dir / "runs" / "failed" / "c1" / "run_context.json"
            )
            failed_context = json.loads(failed_context_path.read_text())
            failed_context["aiperf_ok"] = False
            failed_context["aiperf_exit_code"] = 1
            failed_context_path.write_text(json.dumps(failed_context))
            malformed_summary = (
                out_dir
                / "runs"
                / "malformed"
                / "c1"
                / "profile_export_aiperf.json"
            )
            malformed_summary.write_text("{")
            script = pathlib.Path(__file__).with_name("perf2price.sh")

            resumed = subprocess.run(
                ["bash", str(script), "--resume", str(out_dir)],
                cwd=script.parent,
                env=self.benchmark_env(fake_aiperf),
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertEqual(
                invocation_log.read_text().splitlines(),
                ["failed", "malformed", "failed", "malformed"],
            )
            self.assertIn("Resume status: 0 complete, 2 retryable", resumed.stdout)
            self.assertIn("reason=aiperf_not_successful", resumed.stdout)
            self.assertIn("reason=summary_missing_or_invalid", resumed.stdout)
            for profile in ("failed", "malformed"):
                backups = list(
                    (out_dir / "resume_backups" / profile / "c1").glob("attempt-*")
                )
                self.assertEqual(len(backups), 1, profile)
                summary = (
                    out_dir
                    / "runs"
                    / profile
                    / "c1"
                    / "profile_export_aiperf.json"
                )
                self.assertIsInstance(json.loads(summary.read_text()), dict)

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

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
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

            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                result = fit.main(["perf2price_fit.py", str(root), "0", "0", "100"])

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
