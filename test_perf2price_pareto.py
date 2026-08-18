import csv
import json
import pathlib
import tempfile
import unittest

import perf2price_pareto as pareto


class ShortModelNameTests(unittest.TestCase):
    def test_drops_provider_prefix(self):
        self.assertEqual(pareto.short_model_name("moonshotai/Kimi-K3"), "Kimi-K3")
        self.assertEqual(pareto.short_model_name("openai/gpt-oss-120b"), "gpt-oss-120b")
        self.assertEqual(
            pareto.short_model_name("nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"),
            "NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        )
        self.assertEqual(pareto.short_model_name("Qwen/Qwen3-32B-FP8"), "Qwen3-32B-FP8")

    def test_keeps_unprefixed_and_empty(self):
        self.assertEqual(pareto.short_model_name("Kimi-K3"), "Kimi-K3")
        self.assertEqual(pareto.short_model_name("  org/name/  "), "name")
        self.assertEqual(pareto.short_model_name(""), "unknown")
        self.assertEqual(pareto.short_model_name(None), "unknown")


class DiscoverAndLoadTests(unittest.TestCase):
    def write_run(self, root, name, model, rows):
        run_dir = root / name
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(json.dumps({"model": model}))
        fieldnames = [
            "profile",
            "isl",
            "osl",
            "prefix_tokens",
            "concurrency",
            "usage_ok",
            "output_token_throughput_tps",
            "ttft_p99_ms",
            "itl_p99_ms",
            "error_request_count",
            "aiperf_ok",
        ]
        with (run_dir / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return run_dir

    def point(
        self,
        profile="mixed_4k",
        concurrency=4,
        tput=100,
        ttft=20,
        itl=5,
        usage_ok=1,
        aiperf_ok=True,
        errors=0,
        isl=4096,
        osl=512,
        prefix=0,
    ):
        return {
            "profile": profile,
            "isl": isl,
            "osl": osl,
            "prefix_tokens": prefix,
            "concurrency": concurrency,
            "usage_ok": usage_ok,
            "output_token_throughput_tps": tput,
            "ttft_p99_ms": ttft,
            "itl_p99_ms": itl,
            "error_request_count": errors,
            "aiperf_ok": aiperf_ok,
        }

    def test_discovers_child_runs_and_skips_non_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            first = self.write_run(root, "perf2price-run-a", "org/Model-A", [self.point()])
            second = self.write_run(root, "perf2price-run-b", "org/Model-B", [self.point()])
            (root / "notes").mkdir()
            (root / "notes" / "readme.txt").write_text("ignore")
            found = pareto.discover_run_dirs([root])
            self.assertEqual(found, [first.resolve(), second.resolve()])

    def test_omits_failed_and_errored_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            run_dir = self.write_run(
                root,
                "run",
                "moonshotai/Kimi-K3",
                [
                    self.point(concurrency=4, tput=10),
                    self.point(concurrency=8, tput=20, errors=3),
                    self.point(concurrency=16, tput=30, aiperf_ok=False),
                    self.point(concurrency=32, tput=40, usage_ok=0),
                    self.point(concurrency=64, tput=50),
                ],
            )
            model, grouped, specs = pareto.load_run_series(
                run_dir, "output_token_throughput_tps", ("ttft_p99_ms",)
            )
            self.assertEqual(model, "moonshotai/Kimi-K3")
            conc = [point.concurrency for point in grouped["mixed_4k"]]
            self.assertEqual(conc, [4, 64])
            self.assertEqual(specs["mixed_4k"], (4096, 512, 0))

    def test_collects_short_labels_and_profile_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            self.write_run(
                root,
                "run-kimi",
                "moonshotai/Kimi-K3",
                [
                    self.point(profile="mixed_4k", concurrency=4),
                    self.point(profile="prefill_1k", concurrency=4, isl=1024, osl=64),
                ],
            )
            self.write_run(
                root,
                "run-qwen",
                "Qwen/Qwen3-32B-FP8",
                [self.point(profile="mixed_4k", concurrency=8, tput=80)],
            )
            by_profile = pareto.collect_series(
                pareto.discover_run_dirs([root]),
                "output_token_throughput_tps",
                ("ttft_p99_ms", "itl_p99_ms"),
            )
            self.assertEqual(list(by_profile), ["prefill_1k", "mixed_4k"])
            mixed_labels = [series.label for series in by_profile["mixed_4k"]]
            self.assertEqual(set(mixed_labels), {"Kimi-K3", "Qwen3-32B-FP8"})
            self.assertEqual(by_profile["prefill_1k"][0].label, "Kimi-K3")

    def test_disambiguates_duplicate_short_names(self):
        labels = pareto._disambiguate_labels(
            ["org/Same", "other/Same"],
            [
                pathlib.Path("perf2price-run-20260814-224701"),
                pathlib.Path("perf2price-run-20260816-123225"),
            ],
        )
        self.assertEqual(
            labels,
            ["Same (20260814-224701)", "Same (20260816-123225)"],
        )


class PlotSmokeTests(unittest.TestCase):
    def test_writes_pdf_and_png_with_legend_below(self):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            out_dir = root / "figures"
            run_a = root / "run-a"
            run_b = root / "run-b"
            for run_dir, model, tput in (
                (run_a, "moonshotai/Kimi-K3", 100.0),
                (run_b, "openai/gpt-oss-120b", 140.0),
            ):
                run_dir.mkdir()
                (run_dir / "run_config.json").write_text(json.dumps({"model": model}))
                with (run_dir / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=[
                            "profile",
                            "isl",
                            "osl",
                            "prefix_tokens",
                            "concurrency",
                            "usage_ok",
                            "output_token_throughput_tps",
                            "ttft_p99_ms",
                            "itl_p99_ms",
                            "error_request_count",
                            "aiperf_ok",
                        ],
                    )
                    writer.writeheader()
                    for concurrency, scale in ((4, 1.0), (8, 1.6), (16, 2.1)):
                        writer.writerow(
                            {
                                "profile": "decode_512",
                                "isl": 512,
                                "osl": 512,
                                "prefix_tokens": 0,
                                "concurrency": concurrency,
                                "usage_ok": 1,
                                "output_token_throughput_tps": tput * scale,
                                "ttft_p99_ms": 40 * scale,
                                "itl_p99_ms": 8 * scale,
                                "error_request_count": 0,
                                "aiperf_ok": True,
                            }
                        )

            code = pareto.main(
                [
                    str(run_a),
                    str(run_b),
                    "--out-dir",
                    str(out_dir),
                    "--formats",
                    "pdf,png",
                ]
            )
            self.assertEqual(code, 0)
            pdf = out_dir / "pareto-decode_512.pdf"
            png = out_dir / "pareto-decode_512.png"
            self.assertTrue(pdf.is_file())
            self.assertGreater(pdf.stat().st_size, 1000)
            self.assertTrue(png.is_file())
            self.assertGreater(png.stat().st_size, 1000)
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
