import json
import os
import pathlib
import tempfile
import unittest

import perf2price_run as run_state


class RunStateTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = pathlib.Path(self.temp_dir.name)

    @staticmethod
    def valid_config():
        return {
            "url": "http://localhost:8000",
            "endpoint": "/v1/chat/completions",
            "models_endpoint": "/v1/models",
            "model": "test-model",
            "tokenizer": "test-tokenizer",
            "tokenizer_revision": None,
            "tokenizer_trust_remote_code": False,
            "apply_chat_template": False,
            "legacy_max_tokens": False,
            "run_cache_tests": True,
            "skip_probe": True,
            "max_context": 0,
            "plan_file": None,
            "concurrency_list": "1,2",
            "duration_seconds": 10,
            "grace_period": "0",
            "ttft_p99_ms": 0,
            "itl_p99_ms": 0,
            "resource_hour_cost_usd": None,
            "random_seed": 100,
            "num_dataset_entries": 128,
            "aiperf_version": "0.12.0",
            "extra_aiperf_args": [],
        }

    def write_plan(self, body):
        path = self.root / "benchmark_plan.csv"
        path.write_text("name,isl,osl,prefix_tokens\n" + body)
        return path

    def write_config(self, config=None):
        path = self.root / "run_config.json"
        path.write_text(json.dumps(config or self.valid_config()))
        return path

    def point(self, row, concurrency=1):
        point = self.root / "runs" / row.name / f"c{concurrency}"
        point.mkdir(parents=True)
        return point

    def test_load_plan_accepts_comments_and_trimmed_rows(self):
        path = self.write_plan("# comment\n first , 128 , 16 , 0 \n\nsecond,0,0,4")

        self.assertEqual(
            run_state.load_plan(path),
            [
                run_state.PlanRow("first", 128, 16, 0),
                run_state.PlanRow("second", 0, 0, 4),
            ],
        )

    def test_plan_rejects_unsafe_profile_name(self):
        path = self.write_plan("../escape,128,16,0\n")

        with self.assertRaisesRegex(run_state.RunStateError, "profile name"):
            run_state.load_plan(path)

    def test_plan_rejects_duplicates_invalid_numbers_and_empty_data(self):
        cases = {
            "duplicate": "same,1,2,3\nsame,4,5,6\n",
            "negative": "single,-1,2,3\n",
            "not_integer": "single,one,2,3\n",
            "wrong_columns": "single,1,2\n",
            "empty": "# only a comment\n",
        }
        for name, body in cases.items():
            with self.subTest(name=name):
                path = self.write_plan(body)
                with self.assertRaises(run_state.RunStateError):
                    run_state.load_plan(path)

    def test_old_config_defaults_new_metadata(self):
        self.write_config()

        config = run_state.load_run_config(self.root)

        self.assertEqual(config["config_schema_version"], 0)
        self.assertIsNone(config["api_key_required"])
        self.assertEqual(config["http_connection_limit"], 66)

    def test_config_rejects_unknown_schema_version(self):
        config = self.valid_config()
        config["config_schema_version"] = 3
        config["api_key_required"] = False
        self.write_config(config)

        with self.assertRaisesRegex(run_state.RunStateError, "schema version"):
            run_state.load_run_config(self.root)

    def test_config_rejects_malformed_missing_and_wrong_typed_fields(self):
        path = self.root / "run_config.json"
        path.write_text("[]")
        with self.assertRaisesRegex(run_state.RunStateError, "JSON object"):
            run_state.load_run_config(self.root)

        config = self.valid_config()
        del config["model"]
        self.write_config(config)
        with self.assertRaisesRegex(run_state.RunStateError, "missing.*model"):
            run_state.load_run_config(self.root)

        config = self.valid_config()
        config["skip_probe"] = 1
        self.write_config(config)
        with self.assertRaisesRegex(run_state.RunStateError, "skip_probe"):
            run_state.load_run_config(self.root)

        config = self.valid_config()
        config["extra_aiperf_args"] = ["--flag", 1]
        self.write_config(config)
        with self.assertRaisesRegex(run_state.RunStateError, "extra_aiperf_args"):
            run_state.load_run_config(self.root)

    def test_write_run_config_is_atomic_and_validated(self):
        config = self.valid_config()
        config["config_schema_version"] = 2
        config["api_key_required"] = False
        config["http_connection_limit"] = 66

        run_state.write_run_config(self.root / "run_config.json", config)

        self.assertEqual(run_state.load_run_config(self.root), config)
        self.assertEqual(list(self.root.glob(".run_config.json.*")), [])

    def test_classify_point_requires_success_and_readable_summary(self):
        row = run_state.PlanRow("single", 128, 16, 0)
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66),
            run_state.PointStatus("not_started", "directory_missing"),
        )

        point = self.point(row)
        context_path = point / "run_context.json"
        run_state.write_run_context(context_path, row, 1, 10, 66)
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66),
            run_state.PointStatus("retryable", "aiperf_not_successful"),
        )

        run_state.finish_run_context(context_path, 0)
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66),
            run_state.PointStatus("retryable", "summary_missing_or_invalid"),
        )

        (point / "profile_export_aiperf.json").write_text("{}")
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66),
            run_state.PointStatus("complete", "success"),
        )

    def test_classify_point_retries_bad_context_and_failed_aiperf(self):
        row = run_state.PlanRow("single", 128, 16, 0)
        point = self.point(row)
        context_path = point / "run_context.json"

        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "context_missing",
        )
        context_path.write_text("{")
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "context_invalid",
        )
        context_path.write_text("[]")
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "context_invalid",
        )
        run_state.write_run_context(context_path, row, 1, 10, 66)
        run_state.finish_run_context(context_path, 2)
        (point / "profile_export_aiperf.json").write_text("{}")
        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "aiperf_not_successful",
        )

    def test_classify_point_treats_broken_directory_symlink_as_retryable(self):
        row = run_state.PlanRow("single", 128, 16, 0)
        point = self.root / "runs" / "single" / "c1"
        point.parent.mkdir(parents=True)
        point.symlink_to(self.root / "missing", target_is_directory=True)

        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66),
            run_state.PointStatus("retryable", "directory_invalid"),
        )

    def test_classify_point_retries_every_metadata_mismatch(self):
        row = run_state.PlanRow("single", 128, 16, 4)
        point = self.point(row, concurrency=2)
        context_path = point / "run_context.json"
        run_state.write_run_context(context_path, row, 2, 10, 66)
        run_state.finish_run_context(context_path, 0)
        (point / "profile_export_aiperf.json").write_text("{}")

        context = json.loads(context_path.read_text())
        changes = {
            "profile": "other",
            "isl": 129,
            "osl": 17,
            "prefix_tokens": 5,
            "concurrency": 3,
            "http_connection_limit": 67,
            "requested_duration_seconds": 10.1,
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                changed = dict(context)
                changed[field] = value
                context_path.write_text(json.dumps(changed))
                self.assertEqual(
                    run_state.classify_point(self.root, row, 2, 10, 66).reason,
                    f"metadata_mismatch_{field}",
                )

    def test_classify_point_rejects_boolean_integer_metadata(self):
        row = run_state.PlanRow("single", 128, 16, 0)
        point = self.point(row)
        context_path = point / "run_context.json"
        run_state.write_run_context(context_path, row, 1, 10, 66)
        run_state.finish_run_context(context_path, 0)
        context = json.loads(context_path.read_text())
        context["concurrency"] = True
        context_path.write_text(json.dumps(context))
        (point / "profile_export_aiperf.json").write_text("{}")

        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "metadata_mismatch_concurrency",
        )

    def test_classify_point_retries_malformed_summary(self):
        row = run_state.PlanRow("single", 128, 16, 0)
        point = self.point(row)
        context_path = point / "run_context.json"
        run_state.write_run_context(context_path, row, 1, 10, 66)
        run_state.finish_run_context(context_path, 0)
        (point / "profile_export_aiperf.json").write_text("{")

        self.assertEqual(
            run_state.classify_point(self.root, row, 1, 10, 66).reason,
            "summary_missing_or_invalid",
        )

    def test_archive_point_preserves_distinct_attempts_outside_runs(self):
        point = self.root / "runs" / "single" / "c1"
        point.mkdir(parents=True)
        (point / "partial.log").write_text("first")

        first = run_state.archive_point(
            self.root, "single", 1, "missing_summary", now_ns=123, pid=456
        )

        self.assertFalse(point.exists())
        self.assertEqual((first / "partial.log").read_text(), "first")
        self.assertTrue(
            str(first.relative_to(self.root)).startswith("resume_backups/single/c1/")
        )

        point.mkdir(parents=True)
        (point / "partial.log").write_text("second")
        second = run_state.archive_point(
            self.root, "single", 1, "context invalid", now_ns=124, pid=456
        )

        self.assertNotEqual(first, second)
        self.assertEqual((first / "partial.log").read_text(), "first")
        self.assertEqual((second / "partial.log").read_text(), "second")

    def test_archive_point_handles_missing_and_rejects_unsafe_targets(self):
        self.assertIsNone(
            run_state.archive_point(self.root, "single", 1, "not_started")
        )
        with self.assertRaises(run_state.RunStateError):
            run_state.archive_point(self.root, "../escape", 1, "invalid")
        with self.assertRaisesRegex(run_state.RunStateError, "concurrency"):
            run_state.archive_point(self.root, "single", 0, "invalid")

    def test_expected_points_canonicalize_quoted_plan_rows(self):
        self.write_config()
        plan = self.root / "benchmark_plan.csv"
        plan.write_text(
            'name,isl,osl,prefix_tokens\n"single","128","16","0"\n'
        )

        self.assertEqual(
            run_state.expected_points(self.root),
            [
                run_state.ExpectedPoint(
                    run_state.PlanRow("single", 128, 16, 0), 1, 10.0, 66
                ),
                run_state.ExpectedPoint(
                    run_state.PlanRow("single", 128, 16, 0), 2, 10.0, 66
                ),
            ],
        )

    def test_expected_points_deduplicate_concurrency_values(self):
        config = self.valid_config()
        config["concurrency_list"] = "1,1,2,1"
        self.write_config(config)
        self.write_plan("single,128,16,0\n")

        self.assertEqual(
            [point.concurrency for point in run_state.expected_points(self.root)],
            [1, 2],
        )

    def test_archive_point_rejects_symlinked_backup_root(self):
        external_dir = tempfile.TemporaryDirectory()
        self.addCleanup(external_dir.cleanup)
        external = pathlib.Path(external_dir.name)
        point = self.root / "runs" / "single" / "c1"
        point.mkdir(parents=True)
        (point / "partial.log").write_text("preserve in run")
        (self.root / "resume_backups").symlink_to(
            external, target_is_directory=True
        )

        with self.assertRaisesRegex(run_state.RunStateError, "real directory"):
            run_state.archive_point(self.root, "single", 1, "retryable")

        self.assertEqual((point / "partial.log").read_text(), "preserve in run")
        self.assertEqual(list(external.iterdir()), [])

    def test_archive_unexpected_points_preserves_non_authoritative_entries(self):
        self.write_config()
        self.write_plan("single,128,16,0\n")
        expected = self.root / "runs" / "single" / "c1"
        expected.mkdir(parents=True)
        (expected / "expected.txt").write_text("keep")
        extra_concurrency = self.root / "runs" / "single" / "c3"
        extra_concurrency.mkdir()
        (extra_concurrency / "extra.txt").write_text("extra concurrency")
        removed_profile = self.root / "runs" / "removed" / "c1"
        removed_profile.mkdir(parents=True)
        (removed_profile / "extra.txt").write_text("removed profile")

        archived = run_state.archive_unexpected_points(
            self.root, now_ns=123, pid=456
        )

        self.assertEqual((expected / "expected.txt").read_text(), "keep")
        self.assertFalse(extra_concurrency.exists())
        self.assertFalse(removed_profile.exists())
        self.assertEqual(len(archived), 2)
        self.assertEqual(
            sorted(
                evidence.read_text()
                for path in archived
                for evidence in path.rglob("extra.txt")
            ),
            ["extra concurrency", "removed profile"],
        )
        self.assertTrue(
            all("resume_backups/unexpected" in str(path) for path in archived)
        )

    def test_unexpected_archive_rejects_symlinked_backup_root(self):
        external_dir = tempfile.TemporaryDirectory()
        self.addCleanup(external_dir.cleanup)
        external = pathlib.Path(external_dir.name)
        self.write_config()
        self.write_plan("single,128,16,0\n")
        unexpected = self.root / "runs" / "removed" / "c1"
        unexpected.mkdir(parents=True)
        (self.root / "resume_backups").symlink_to(
            external, target_is_directory=True
        )

        with self.assertRaisesRegex(run_state.RunStateError, "real directory"):
            run_state.archive_unexpected_points(self.root)

        self.assertTrue(unexpected.is_dir())
        self.assertEqual(list(external.iterdir()), [])

    def test_lock_reclaims_only_stale_local_owner(self):
        run_state.acquire_lock(
            self.root,
            100,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=1,
        )
        with self.assertRaisesRegex(run_state.RunStateError, "PID 100"):
            run_state.acquire_lock(
                self.root,
                200,
                hostname="host",
                is_alive=lambda _: True,
                now_ns=2,
            )

        run_state.acquire_lock(
            self.root,
            200,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=3,
        )

        owner = json.loads(
            (self.root / ".perf2price.lock" / "owner.json").read_text()
        )
        self.assertEqual(owner["pid"], 200)
        stale = list((self.root / "resume_backups" / "locks").iterdir())
        self.assertEqual(len(stale), 1)
        self.assertEqual(
            json.loads((stale[0] / "owner.json").read_text())["pid"], 100
        )

        with self.assertRaisesRegex(run_state.RunStateError, "owned by"):
            run_state.release_lock(self.root, 201, hostname="host")
        run_state.release_lock(self.root, 200, hostname="host")
        self.assertFalse((self.root / ".perf2price.lock").exists())

    def test_lock_stays_held_while_registered_aiperf_child_is_alive(self):
        run_state.acquire_lock(
            self.root,
            100,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=1,
        )
        run_state.register_lock_child(self.root, 100, 300, hostname="host")

        with self.assertRaisesRegex(run_state.RunStateError, "AIPerf PID 300"):
            run_state.acquire_lock(
                self.root,
                200,
                hostname="host",
                is_alive=lambda pid: pid == 300,
                now_ns=2,
            )
        with self.assertRaisesRegex(run_state.RunStateError, "AIPerf PID 300"):
            run_state.release_lock(
                self.root,
                100,
                hostname="host",
                is_alive=lambda pid: pid == 300,
            )

    def test_stale_lock_archive_rejects_symlinked_backup_root(self):
        external_dir = tempfile.TemporaryDirectory()
        self.addCleanup(external_dir.cleanup)
        external = pathlib.Path(external_dir.name)
        run_state.acquire_lock(
            self.root,
            100,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=1,
        )
        (self.root / "resume_backups").symlink_to(
            external, target_is_directory=True
        )

        with self.assertRaisesRegex(run_state.RunStateError, "real directory"):
            run_state.acquire_lock(
                self.root,
                200,
                hostname="host",
                is_alive=lambda _: False,
                now_ns=2,
            )

        owner = json.loads(
            (self.root / ".perf2price.lock" / "owner.json").read_text()
        )
        self.assertEqual(owner["pid"], 100)
        self.assertEqual(list(external.iterdir()), [])

    def test_lock_never_reclaims_foreign_host(self):
        run_state.acquire_lock(
            self.root,
            os.getpid(),
            hostname="host-a",
            is_alive=lambda _: True,
            now_ns=1,
        )

        with self.assertRaisesRegex(run_state.RunStateError, "host-a"):
            run_state.acquire_lock(
                self.root,
                200,
                hostname="host-b",
                is_alive=lambda _: False,
                now_ns=2,
            )

    def test_lock_reclaims_malformed_owner_as_stale(self):
        lock = self.root / ".perf2price.lock"
        lock.mkdir()
        (lock / "owner.json").write_text("{")

        run_state.acquire_lock(
            self.root,
            200,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=2,
        )

        owner = json.loads((lock / "owner.json").read_text())
        self.assertEqual(owner["pid"], 200)
        self.assertTrue((self.root / "resume_backups" / "locks").is_dir())

    def test_lock_preserves_empty_existing_lock_directory(self):
        lock = self.root / ".perf2price.lock"
        lock.mkdir()

        run_state.acquire_lock(
            self.root,
            200,
            hostname="host",
            is_alive=lambda _: False,
            now_ns=2,
        )

        backups = list((self.root / "resume_backups" / "locks").iterdir())
        self.assertEqual(len(backups), 1)
        self.assertTrue(backups[0].is_dir())
        self.assertEqual(
            json.loads((lock / "owner.json").read_text())["pid"],
            200,
        )


if __name__ == "__main__":
    unittest.main()
