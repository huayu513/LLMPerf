import tempfile
import unittest
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

from web import server


class WebSearchTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "frontend verification needs Node.js")
    def test_trial_keeps_valid_badge_and_renders_escaped_warnings(self):
        script = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const elements = new Map();
const context = { console, document: { getElementById: id => {
  if (!elements.has(id)) elements.set(id, { value: "", innerHTML: "", querySelectorAll: () => [] });
  return elements.get(id);
}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync("web/static/app.js", "utf8").replace("init().catch(showError);", ""), context);
context.state.trials = [{ task_id: "task", status: "VALID", reasons: [],
  warnings: ["memory_allocation_oom_observed", "<unsafe>"], output_tokens_per_second: 20,
  successes: 256, request_count: 256 }];
context.renderTrials();
let html = context.document.getElementById("trials").innerHTML;
for (const expected of ["VALID", "失败原因 / 告警", "告警：", "不单独代表本轮失败", "&lt;unsafe&gt;", "256 / 256"])
  assert.ok(html.includes(expected), expected);
assert.ok(!html.includes("<unsafe>"));
context.document.getElementById("trial-filter").value = "memory_allocation_oom_observed";
context.renderTrials();
assert.ok(context.document.getElementById("trials").innerHTML.includes("VALID"));
context.state.trials = [{ task_id: "old-trial", status: "VALID", reasons: [] }];
context.document.getElementById("trial-filter").value = "";
context.renderTrials();
assert.ok(!context.document.getElementById("trials").innerHTML.includes("告警："));
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_run_detail_exposes_weight_reports_without_requiring_them_for_old_runs(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, "summarize_run", return_value={}), \
                patch.object(server, "code_state", return_value={}):
            root = Path(directory)
            self.assertIsNone(server.run_detail(root)["model_weights"])
            self.assertIsNone(server.run_detail(root)["weight_capacity"])
            for filename, key in (("model-weights.json", "model_weights"), ("weight-capacity.json", "weight_capacity")):
                report = {"version": 1, "status": "unknown", "reason": "fixture"}
                (root / filename).write_text(json.dumps(report))
                self.assertEqual(server.run_detail(root)[key], report)

    @unittest.skipUnless(shutil.which("node"), "frontend verification needs Node.js")
    def test_weight_panel_labels_uncertainty_escapes_errors_and_opens_reports(self):
        script = r'''
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const elements = new Map();
let click;
const button = { dataset: { capabilityArtifact: "model-weights.json" }, addEventListener: (_, handler) => { click = handler; } };
const context = { console, document: { getElementById: id => {
  if (!elements.has(id)) elements.set(id, { innerHTML: "", textContent: "", querySelectorAll: () => [button] });
  return elements.get(id);
}}};
vm.createContext(context);
vm.runInContext(fs.readFileSync("web/static/app.js", "utf8").replace("init().catch(showError);", ""), context);
const weights = { status: "complete", format: "safetensors", file_count: 16,
  parameter_count: 300e9, stored_weight_bytes: 600 * 1024 ** 3, weight_file_bytes: 600 * 1024 ** 3,
  minimum_resident_weight_bytes: 599 * 1024 ** 3, dtype_summary: { BF16: {} }, errors: ["<unsafe>"], unknowns: [] };
const capacity = { excluded_count: 2, input_candidates: 4, retained_count: 2, unknown_count: 1, candidates: [] };
context.renderImageCapabilities({ model_weights: weights, weight_capacity: capacity }, {
  engine_catalog: { engine: "vllm", parameters: {} } });
let html = context.document.getElementById("image-capabilities").innerHTML;
for (const expected of ["模型权重自动识别", "非去重参数量", "300.000B", "600.00 GiB", "BF16",
  "淘汰 2 / 4", "其中未知 1", "独立实例不合并显存", "保留不代表能启动", "&lt;unsafe&gt;",
  'data-capability-artifact="model-weights.json"', 'data-capability-artifact="weight-capacity.json"'])
  assert.ok(html.includes(expected), expected);
assert.ok(!html.includes("<unsafe>"));
let opened;
context.loadArtifact = path => { opened = path; return Promise.resolve(); };
context.setActiveTab = () => {};
context.state.selectedTrial = "trial";
click();
assert.equal(opened, "model-weights.json");
assert.equal(context.state.selectedTrial, null);
context.renderImageCapabilities({}, { engine_catalog: { engine: "sglang", parameters: {} },
  model_snapshot: { raw: { weights: { ...weights, parameter_count: null, minimum_resident_weight_bytes: null,
    quantization: "gptq", dtype_summary: { I32: {} } } } } });
html = context.document.getElementById("image-capabilities").innerHTML;
assert.ok(html.includes("未知（不猜测量化模型参数量）"));
assert.ok(html.includes("I32 / gptq"));
context.renderImageCapabilities({}, {});
assert.equal(context.document.getElementById("image-capabilities").hidden, true);
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True,
                                text=True, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_plan_candidates_exposes_failed_and_skipped_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plan.json").write_text(
                '{"candidates":[{"id":"c0","static_config":{}},'
                '{"id":"c1","static_config":{}}]}'
            )
            (run_dir / "results-index.json").write_text('{"rows":[]}')
            (run_dir / "search-state.json").write_text(
                '{"candidate_states":{'
                '"c0":{"status":"FAILED","reasons":["server_info_missing"]},'
                '"c1":{"status":"SKIPPED","reasons":["backend_blocked"]}'
                '}}'
            )
            rows = server.plan_candidates(run_dir)
        self.assertEqual(
            [(row["latest_status"], row["latest_reasons"]) for row in rows],
            [
                ("FAILED", ["server_info_missing"]),
                ("SKIPPED", ["backend_blocked"]),
            ],
        )

    def test_plan_candidates_exposes_comparison_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "plan.json").write_text(
                '{"candidates":[{"id":"c0","static_config":{'
                '"comparison_group":"8g1i-g0_1_2_3_4_5_6_7-tp8-pp1",'
                '"feature_variant":"dp_attention+dspark"}}]}'
            )
            rows = server.plan_candidates(run_dir)
        self.assertEqual(rows[0]["comparison_group"], "8g1i-g0_1_2_3_4_5_6_7-tp8-pp1")
        self.assertEqual(rows[0]["feature_variant"], "dp_attention+dspark")

    def test_preview_uses_max_thinking_when_plan_metadata_is_older(self):
        preview = server.preview_launch(
            {"served_model_name": "model", "expected_request_count": 4},
            {"id": "c0", "gpu_indexes": [0], "static_config": {}},
            1,
        )
        self.assertEqual(preview["requested_parameters"]["chat_template_kwargs"], {
            "enable_thinking": True,
            "reasoning_effort": "max",
            "thinking": True,
        })

    def test_debug_search_increases_concurrency_by_sixteen(self):
        calls = []
        metadata = {
            "expected_request_count": 64,
            "search": {"concurrency_max": 64, "start_concurrency": 16},
        }
        candidate = {"id": "c0", "gpu_indexes": (0,), "static_config": {}}

        class Control:
            def check(self):
                return None

        def fake_context(run_dir, candidate_id):
            return {}, metadata, {"c0": candidate}, candidate

        def fake_attempt(
            emit,
            control,
            run_dir,
            metadata,
            candidate,
            candidate_id,
            concurrency,
            run_class,
            mode,
            scale,
            task_prefix,
        ):
            calls.append(concurrency)
            score = {16: 10, 32: 18, 48: 25, 64: 23}[concurrency]
            return {
                "status": "VALID",
                "output_tokens_per_second": score,
                "concurrency": concurrency,
            }

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(server, "load_debug_context", side_effect=fake_context), \
                patch.object(server, "run_debug_attempt", side_effect=fake_attempt):
            summary = server.debug_search_worker(
                lambda event: None,
                Control(),
                Path(directory),
                "c0",
            )

        self.assertEqual(calls, [16, 32, 48, 64])
        self.assertEqual(summary["best_concurrency"], 48)


if __name__ == "__main__":
    unittest.main()
