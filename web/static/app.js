var state = {
  settings: null,
  resultRoot: "",
  configPath: "",
  runs: [],
  selectedRunId: null,
  runDetail: null,
  candidates: [],
  trials: [],
  candidateDetail: null,
  artifacts: [],
  currentArtifactPath: null,
  jobs: [],
  selectedJobId: null,
  selectedTrial: null
};

function el(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function pretty(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    return String(Math.round(value * 1000) / 1000);
  }
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value, null, 2);
  return String(value);
}

function badge(text) {
  var raw = String(text || "UNKNOWN");
  var upper = raw.toUpperCase();
  var cls = "";
  if (["PASS", "VALID", "OK", "PLANNED"].indexOf(upper) >= 0) cls = "ok";
  else if (["PARTIAL", "WARN", "RUNNING_OR_INTERRUPTED", "RUNNING", "INCONCLUSIVE"].indexOf(upper) >= 0) cls = "warn";
  else if (["FAILED", "FAIL", "UNSUPPORTED", "ERROR"].indexOf(upper) >= 0) cls = "fail";
  return '<span class="badge ' + cls + '">' + escapeHtml(raw) + "</span>";
}

function query(params) {
  var sp = new URLSearchParams();
  Object.keys(params || {}).forEach(function(key) {
    var value = params[key];
    if (value !== undefined && value !== null && value !== "") sp.set(key, value);
  });
  var text = sp.toString();
  return text ? "?" + text : "";
}

async function api(path, options) {
  var init = options || {};
  if (init.body && typeof init.body !== "string") {
    init.headers = Object.assign({ "Content-Type": "application/json" }, init.headers || {});
    init.body = JSON.stringify(init.body);
  }
  var response = await fetch(path, init);
  var text = await response.text();
  var data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (error) {
    data = { text: text };
  }
  if (!response.ok) throw new Error((data && data.error) || response.statusText);
  return data;
}

function runPath(suffix) {
  if (!state.selectedRunId) throw new Error("未选择 run");
  return "/api/runs/" + encodeURIComponent(state.selectedRunId) + (suffix || "") + query({ result_root: state.resultRoot });
}

function setActiveTab(name) {
  document.querySelectorAll(".tab").forEach(function(tab) {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".tab-panel").forEach(function(panel) {
    panel.classList.toggle("active", panel.id === "tab-" + name);
  });
}

function renderKV(items) {
  var html = '<div class="summary-grid">';
  items.forEach(function(item) {
    html += '<div class="kv"><div class="key">' + escapeHtml(item[0]) + '</div><div class="value">' + escapeHtml(pretty(item[1])) + "</div></div>";
  });
  html += "</div>";
  return html;
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function planOrderValue(row) {
  var value = finiteNumber(row && row.plan_order);
  return value === null ? 1000000 : value;
}

function trialOrderValue(row) {
  var value = finiteNumber(row && row.trial_order);
  return value === null ? 1000000 : value;
}

function outputValue(row, field) {
  var value = finiteNumber(row && row[field]);
  return value === null ? -Infinity : value;
}

function candidateSortMode() {
  return el("candidate-sort") ? el("candidate-sort").value : "plan";
}

function trialSortMode() {
  return el("trial-sort") ? el("trial-sort").value : "plan";
}

function candidatePlanSort(a, b) {
  var diff = planOrderValue(a) - planOrderValue(b);
  if (diff) return diff;
  return String(a.id || "").localeCompare(String(b.id || ""));
}

function sortedCandidates(rows) {
  var copy = rows.slice();
  if (candidateSortMode() === "output_desc") {
    copy.sort(function(a, b) {
      var diff = outputValue(b, "best_output_tokens_per_second") - outputValue(a, "best_output_tokens_per_second");
      return diff || candidatePlanSort(a, b);
    });
    return copy;
  }
  copy.sort(candidatePlanSort);
  return copy;
}

function runClassRank(row) {
  var runClass = String((row && row.run_class) || "");
  var ranks = {
    smoke: 0,
    concurrency: 1,
    diagnostic: 1,
    tuning: 2,
    final_repeat: 3,
    open_loop: 4,
    formal: 5
  };
  return Object.prototype.hasOwnProperty.call(ranks, runClass) ? ranks[runClass] : 9;
}

function trialPlanSort(a, b) {
  var diff = planOrderValue(a) - planOrderValue(b);
  if (diff) return diff;
  diff = trialOrderValue(a) - trialOrderValue(b);
  if (diff) return diff;
  diff = runClassRank(a) - runClassRank(b);
  if (diff) return diff;
  diff = (finiteNumber(a.concurrency) || 0) - (finiteNumber(b.concurrency) || 0);
  if (diff) return diff;
  diff = (finiteNumber(a.attempt) || 0) - (finiteNumber(b.attempt) || 0);
  if (diff) return diff;
  return String(a.task_id || a.id || "").localeCompare(String(b.task_id || b.id || ""));
}

function sortedTrials(rows) {
  var copy = rows.slice();
  if (trialSortMode() === "output_desc") {
    copy.sort(function(a, b) {
      var diff = outputValue(b, "output_tokens_per_second") - outputValue(a, "output_tokens_per_second");
      return diff || trialPlanSort(a, b);
    });
    return copy;
  }
  copy.sort(trialPlanSort);
  return copy;
}

function trialStage(row) {
  var runClass = String((row && row.run_class) || "");
  if (runClass === "final_repeat") return "二阶段";
  if (runClass === "open_loop") return "open-loop";
  if (runClass === "smoke" || runClass === "concurrency" || runClass === "tuning" || runClass === "diagnostic") return "一阶段";
  return "-";
}

function stageSummary(rows) {
  var counts = { "一阶段": 0, "二阶段": 0, "open-loop": 0 };
  (rows || []).forEach(function(row) {
    var stage = trialStage(row);
    if (Object.prototype.hasOwnProperty.call(counts, stage)) counts[stage] += 1;
  });
  return "一阶段 " + counts["一阶段"] + " / 二阶段 " + counts["二阶段"] + " / open-loop " + counts["open-loop"];
}

function currentStage(rows, bestDoc) {
  if (bestDoc && bestDoc.status) return "已写入 best.json";
  if ((rows || []).some(function(row) { return trialStage(row) === "二阶段"; })) return "二阶段";
  if ((rows || []).some(function(row) { return trialStage(row) === "一阶段"; })) return "一阶段";
  return "未开始";
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  state.resultRoot = state.settings.default_result_root;
  state.configPath = state.settings.default_config_path;
  el("result-root").value = state.resultRoot;
  el("config-path").value = state.configPath;
  el("server-state").innerHTML =
    "<div>" + escapeHtml(state.settings.llmperf_root) + '</div><div class="small">Python: ' + escapeHtml(state.settings.python) + "</div>";
}

async function loadRuns() {
  state.resultRoot = el("result-root").value.trim() || state.settings.default_result_root;
  var data = await api("/api/runs" + query({ result_root: state.resultRoot }));
  state.runs = data.runs || [];
  renderRuns();
  if (state.selectedRunId) {
    var stillExists = state.runs.some(function(run) { return run.id === state.selectedRunId; });
    if (stillExists) await selectRun(state.selectedRunId);
  }
}

function renderRuns() {
  el("run-count").textContent = String(state.runs.length);
  var box = el("runs");
  if (!state.runs.length) {
    box.className = "list empty";
    box.textContent = "暂无数据";
    return;
  }
  box.className = "list";
  var html = "";
  state.runs.forEach(function(run) {
    var best = run.best_summary && run.best_summary.best;
    var score = best && best.output_tokens_per_second ? pretty(best.output_tokens_per_second) + " tok/s" : "-";
    html += '<div class="run-item ' + (run.id === state.selectedRunId ? "active" : "") + '" data-run="' + escapeHtml(run.id) + '">';
    html += '<div class="title-line"><strong class="code">' + escapeHtml(run.id) + "</strong>" + badge(run.status) + "</div>";
    html += '<div class="small">' + escapeHtml(run.path) + "</div>";
    html += '<div class="small">candidates: ' + pretty(run.candidate_count) + " · trials: " + pretty(run.row_count) + " · best: " + escapeHtml(score) + "</div>";
    html += "</div>";
  });
  box.innerHTML = html;
  box.querySelectorAll(".run-item").forEach(function(item) {
    item.addEventListener("click", function() { selectRun(item.dataset.run).catch(showError); });
  });
}

async function selectRun(runId, options) {
  var opts = options || {};
  state.selectedRunId = runId;
  state.runDetail = await api(runPath(""));
  var candidates = await api(runPath("/candidates"));
  var trials = await api(runPath("/trials"));
  state.candidates = candidates.candidates || [];
  state.trials = trials.rows || [];
  if (!opts.preserveArtifacts) {
    state.artifacts = [];
    state.currentArtifactPath = null;
    state.selectedTrial = null;
  }
  renderRuns();
  renderSummary();
  renderCandidates();
  renderTrials();
  renderArtifacts();
  el("run-plan").disabled = false;
  el("resume-run").disabled = false;
  el("adopt-runtime").disabled = false;
  el("collect-run").disabled = false;
}

function renderSummary() {
  var detail = state.runDetail || {};
  var summary = detail.summary || {};
  var bestDoc = detail.best || {};
  var best = bestDoc.best || null;
  var code = detail.code_state || {};
  var rows = (detail.results_index && detail.results_index.rows) || [];
  var validRows = rows.filter(function(row) { return row.status === "VALID"; });
  el("summary-content").className = "";
  el("summary-content").innerHTML = renderKV([
    ["Run", summary.id],
    ["状态", summary.status],
    ["路径", summary.path],
    ["候选数", summary.candidate_count],
    ["Trial 数", summary.row_count],
    ["当前阶段", currentStage(state.trials, bestDoc)],
    ["阶段统计", stageSummary(state.trials)],
    ["Served model", summary.served_model_name],
    ["Best candidate", best && best.candidate_id],
    ["Best concurrency", best && best.concurrency],
    ["Best output tok/s", best && best.output_tokens_per_second],
    ["有效 trial", validRows.length],
    ["代码指纹一致", code.matches],
    ["当前代码 fingerprint", code.current_bundle_fingerprint]
  ]);
}

function renderCandidates() {
  var filter = el("candidate-filter").value.trim().toLowerCase();
  var rows = state.candidates.filter(function(candidate) {
    return !filter || JSON.stringify(candidate).toLowerCase().indexOf(filter) >= 0;
  });
  rows = sortedCandidates(rows);
  var box = el("candidates");
  if (!rows.length) {
    box.className = "table-wrap empty";
    box.textContent = "暂无候选。";
    return;
  }
  box.className = "table-wrap";
  var html = '<table><thead><tr>';
  ["candidate", "状态", "拓扑", "GPU", "TP/DP/PP", "DPA", "backend", "DSpark", "mem", "chunked", "当前最佳 trial"].forEach(function(name) {
    html += "<th>" + escapeHtml(name) + "</th>";
  });
  html += "</tr></thead><tbody>";
  rows.forEach(function(c) {
    html += '<tr class="clickable" title="双击查看参数详情" data-candidate="' + escapeHtml(c.id) + '">';
    html += '<td class="code">' + escapeHtml(c.id) + "</td>";
    html += "<td>" + badge(c.latest_status) + "</td>";
    html += "<td>" + escapeHtml(c.deployment_label || "-") + '<div class="small">' + pretty(c.instance_count) + " inst · " + pretty(c.gpus_per_instance) + " gpu/inst</div></td>";
    html += "<td>" + escapeHtml(pretty(c.gpu_indexes)) + "</td>";
    html += "<td>" + pretty(c.tp) + " / " + pretty(c.dp) + " / " + pretty(c.pp) + "</td>";
    html += "<td>" + pretty(c.dp_attention) + "</td>";
    html += "<td>" + escapeHtml(pretty(c.backend)) + "</td>";
    html += "<td>" + pretty(c.dspark) + "</td>";
    html += "<td>" + pretty(c.mem_fraction_static) + "</td>";
    html += "<td>" + pretty(c.chunked_prefill_size) + "</td>";
    html += "<td>" + (c.best_output_tokens_per_second ? pretty(c.best_output_tokens_per_second) + " @ c" + pretty(c.best_concurrency) : "-") + "</td>";
    html += "</tr>";
  });
  html += "</tbody></table>";
  box.innerHTML = html;
  box.querySelectorAll("tr[data-candidate]").forEach(function(row) {
    row.addEventListener("dblclick", function() { openCandidate(row.dataset.candidate).catch(showError); });
  });
}

function renderTrials() {
  var filter = el("trial-filter").value.trim().toLowerCase();
  var rows = state.trials.filter(function(row) {
    return !filter || JSON.stringify(row).toLowerCase().indexOf(filter) >= 0;
  });
  rows = sortedTrials(rows);
  var box = el("trials");
  if (!rows.length) {
    box.className = "table-wrap empty";
    box.textContent = "暂无 trial。";
    return;
  }
  box.className = "table-wrap";
  var html = '<table><thead><tr>';
  ["task", "candidate", "状态", "阶段", "class", "mode", "concurrency", "tok/s", "requests", "reasons"].forEach(function(name) {
    html += "<th>" + escapeHtml(name) + "</th>";
  });
  html += "</tr></thead><tbody>";
  rows.forEach(function(row, idx) {
    html += '<tr class="clickable" title="双击查看这次 trial 的日志" data-trial="' + idx + '">';
    html += '<td class="code">' + escapeHtml(row.task_id || row.id || "-") + (row.debug ? '<div class="small">debug</div>' : "") + "</td>";
    html += '<td class="code">' + escapeHtml(row.candidate_id || "-") + "</td>";
    html += "<td>" + badge(row.status) + "</td>";
    html += "<td>" + escapeHtml(trialStage(row)) + "</td>";
    html += "<td>" + escapeHtml(row.run_class || "-") + "</td>";
    html += "<td>" + escapeHtml(row.mode || "-") + "</td>";
    html += "<td>" + pretty(row.concurrency) + "</td>";
    html += "<td>" + pretty(row.output_tokens_per_second) + "</td>";
    html += "<td>" + pretty(row.successes) + " / " + pretty(row.request_count) + "</td>";
    html += "<td>" + escapeHtml(pretty(row.reasons || [])) + "</td>";
    html += "</tr>";
  });
  html += "</tbody></table>";
  box.innerHTML = html;
  box.querySelectorAll("tr[data-trial]").forEach(function(row) {
    row.addEventListener("dblclick", function() { showTrialArtifacts(rows[Number(row.dataset.trial)]).catch(showError); });
  });
}

async function openCandidate(candidateId) {
  state.candidateDetail = await api(runPath("/candidates/" + encodeURIComponent(candidateId)));
  var detail = state.candidateDetail;
  el("drawer-title").textContent = candidateId;
  var command = ((detail.launch_preview && detail.launch_preview.sglang_command_preview) || []).join(" ");
  var html = renderKV([
    ["默认重跑 concurrency", detail.default_debug && detail.default_debug.concurrency],
    ["默认 run_class", detail.default_debug && detail.default_debug.run_class],
    ["默认 mode", detail.default_debug && detail.default_debug.mode],
    ["served model", detail.metadata && detail.metadata.served_model_name],
    ["image", detail.metadata && detail.metadata.image],
    ["service_port", detail.metadata && detail.metadata.service_port],
    ["tool parser", detail.metadata && detail.metadata.tool_call_parser],
    ["reasoning parser", detail.metadata && detail.metadata.reasoning_parser]
  ]);
  html += '<div class="kv" style="grid-column:1/-1"><div class="key">预计 SGLang 命令</div><pre>' + escapeHtml(command) + "</pre></div>";
  html += '<div class="kv" style="grid-column:1/-1"><div class="key">规划 static_config</div><pre>' + escapeHtml(JSON.stringify(detail.static_config || {}, null, 2)) + "</pre></div>";
  html += '<div class="kv" style="grid-column:1/-1"><div class="key">预计 requested parameters</div><pre>' + escapeHtml(JSON.stringify((detail.launch_preview && detail.launch_preview.requested_parameters) || {}, null, 2)) + "</pre></div>";
  el("candidate-detail").innerHTML = html;
  el("candidate-drawer").classList.add("open");
  el("candidate-drawer").setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  el("candidate-drawer").classList.remove("open");
  el("candidate-drawer").setAttribute("aria-hidden", "true");
}

function showCandidateArtifacts() {
  if (!state.candidateDetail) return;
  state.selectedTrial = null;
  state.artifacts = state.candidateDetail.artifacts || [];
  state.currentArtifactPath = null;
  el("artifact-title").textContent = (state.candidateDetail.candidate && state.candidateDetail.candidate.id) || "Candidate artifacts";
  renderArtifacts();
  setActiveTab("artifacts");
  closeDrawer();
}

function makeRunRelative(path) {
  var runPathValue = state.runDetail && state.runDetail.summary && state.runDetail.summary.path;
  if (!path || !runPathValue) return null;
  if (path.indexOf(runPathValue + "/") === 0) return path.slice(runPathValue.length + 1);
  if (path[0] !== "/") return path;
  return null;
}

async function showTrialArtifacts(row) {
  var base = "";
  if (row.result_path) {
    base = makeRunRelative(row.result_path) || "";
  }
  if (!base && row.manifest && row.manifest.endsWith("/trial.json")) {
    base = row.manifest.slice(0, -"/trial.json".length);
  }
  var artifacts = [];
  if (base) {
    var data = await api(runPath("/artifacts") + "&base=" + encodeURIComponent(base));
    artifacts = data.artifacts || [];
  }
  state.selectedTrial = row;
  state.artifacts = artifacts;
  state.currentArtifactPath = null;
  el("artifact-title").textContent = row.task_id || row.candidate_id || "Trial artifacts";
  renderArtifacts();
  setActiveTab("artifacts");
}

function renderArtifacts() {
  var box = el("artifact-list");
  el("artifact-path").textContent = state.currentArtifactPath || "";
  el("reload-artifact").disabled = !state.currentArtifactPath;
  el("close-artifact").disabled = !state.currentArtifactPath;
  if (!state.artifacts.length) {
    box.className = "list empty";
    box.textContent = "暂无 artifact。";
    el("artifact-content").textContent = "未选择文件。";
    updateRepairButton();
    return;
  }
  box.className = "list";
  var html = "";
  state.artifacts.forEach(function(artifact) {
    html += '<div class="artifact-item ' + (artifact.path === state.currentArtifactPath ? "active" : "") + '" data-path="' + escapeHtml(artifact.path) + '">';
    html += '<div class="title-line"><strong>' + escapeHtml(artifact.name || artifact.path) + "</strong>";
    if (artifact.size) html += '<span class="small">' + pretty(artifact.size) + " B</span>";
    html += '</div><div class="small code">' + escapeHtml(artifact.path) + "</div></div>";
  });
  box.innerHTML = html;
  box.querySelectorAll(".artifact-item").forEach(function(item) {
    item.addEventListener("click", function() { loadArtifact(item.dataset.path).catch(showError); });
  });
  updateRepairButton();
}

function updateRepairButton() {
  var button = el("repair-trial");
  if (!button) return;
  var row = state.selectedTrial;
  var enabled = !!(row && !row.debug && (row.task_id || row.id || row.manifest));
  button.disabled = !enabled;
  button.title = enabled ? "追加官方 attempt 并重跑这条 trial" : "先在 Trials 里双击选择一条官方 trial";
}

function closeArtifact() {
  state.currentArtifactPath = null;
  el("artifact-path").textContent = "";
  el("artifact-content").textContent = "未选择文件。";
  renderArtifacts();
}

async function loadArtifact(path) {
  if (!path) return;
  state.currentArtifactPath = path;
  el("artifact-path").textContent = path;
  el("artifact-content").textContent = "加载中...";
  renderArtifacts();
  try {
    var data = await api(runPath("/artifact") + "&path=" + encodeURIComponent(path) + "&tail=" + String(2 * 1024 * 1024));
    var text = data.text || "";
    if (data.truncated_to_tail) text = "[只显示文件尾部]\n" + text;
    el("artifact-content").textContent = text;
  } catch (error) {
    el("artifact-content").textContent = "读取失败: " + error.message;
  }
}

async function createJob(path, body) {
  var job = await api(path, { method: "POST", body: body || {} });
  state.selectedJobId = job.id;
  await loadJobs();
  setActiveTab("jobs");
  return job;
}

async function loadJobs() {
  var data = await api("/api/jobs");
  state.jobs = data.jobs || [];
  renderJobs();
}

function renderJobs() {
  var box = el("jobs");
  if (!state.jobs.length) {
    box.className = "list empty";
    box.textContent = "暂无后台任务。";
    el("stop-job").disabled = true;
    return;
  }
  box.className = "list";
  var html = "";
  state.jobs.forEach(function(job) {
    html += '<div class="job-item ' + (job.id === state.selectedJobId ? "active" : "") + '" data-job="' + escapeHtml(job.id) + '">';
    html += '<div class="title-line"><strong>' + escapeHtml(job.name) + " · " + escapeHtml(job.id) + "</strong>" + badge(job.status) + "</div>";
    var meta = escapeHtml(job.started_at || "") + " · " + escapeHtml(job.kind || "job");
    if (job.stop_requested) meta += " · stopping";
    if (job.active_container) meta += " · " + escapeHtml(job.active_container);
    if (job.returncode !== null) meta += " · exit " + escapeHtml(job.returncode);
    html += '<div class="small">' + meta + "</div>";
    if (job.error) html += '<div class="small">' + escapeHtml(job.error) + "</div>";
    html += "</div>";
  });
  box.innerHTML = html;
  box.querySelectorAll(".job-item").forEach(function(item) {
    item.addEventListener("click", function() { selectJob(item.dataset.job).catch(showError); });
  });
  if (state.selectedJobId) selectJob(state.selectedJobId, false).catch(function() {});
}

async function selectJob(jobId, fetchFresh) {
  state.selectedJobId = jobId;
  var job = fetchFresh === false ? state.jobs.find(function(item) { return item.id === jobId; }) : await api("/api/jobs/" + encodeURIComponent(jobId));
  if (!job) return;
  el("stop-job").disabled = job.status !== "running" || !!job.stop_requested;
  var header = "$ " + ((job.argv && job.argv.join(" ")) || job.name);
  var body = (job.lines || []).join("\n");
  if (job.result) body += "\n\n[result]\n" + JSON.stringify(job.result, null, 2);
  el("job-output").textContent = header + "\n\n" + body;
}

async function stopSelectedJob() {
  if (!state.selectedJobId) return;
  await api("/api/jobs/" + encodeURIComponent(state.selectedJobId) + "/stop", { method: "POST", body: {} });
  await loadJobs();
  await selectJob(state.selectedJobId);
}

async function planRun() {
  state.configPath = el("config-path").value.trim() || state.settings.default_config_path;
  await createJob("/api/jobs/plan", { config_path: state.configPath });
}

async function autoRun() {
  state.configPath = el("config-path").value.trim() || state.settings.default_config_path;
  await createJob("/api/jobs/auto", { config_path: state.configPath });
}

async function startSavedPlan(resume) {
  await createJob(runPath(resume ? "/resume" : "/run"), {});
}

async function collectRun() {
  await api(runPath("/collect"), { method: "POST", body: {} });
  await selectRun(state.selectedRunId);
}

async function adoptRuntime() {
  if (!window.confirm("确认接受当前 LLMPerf 代码继续 resume？会备份并更新 plan.json / search-state.json 的 fingerprint。")) return;
  var note = window.prompt("备注，可留空：", "frontend adopt current runtime") || "";
  await api(runPath("/adopt-runtime"), { method: "POST", body: { note: note } });
  await selectRun(state.selectedRunId);
}

async function debugRerun() {
  var detail = state.candidateDetail;
  if (!detail) return;
  var candidateId = detail.candidate.id;
  var defaults = detail.default_debug || {};
  var rawConcurrency = window.prompt("重跑 concurrency：", defaults.concurrency || 1);
  if (rawConcurrency === null) return;
  var concurrency = Number(rawConcurrency);
  if (!Number.isFinite(concurrency) || concurrency < 1) {
    window.alert("concurrency 必须是正数");
    return;
  }
  await createJob(runPath("/debug-rerun"), {
    candidate_id: candidateId,
    concurrency: concurrency,
    run_class: defaults.run_class || "concurrency",
    mode: defaults.mode || "closed-loop"
  });
  closeDrawer();
}

async function debugSearch() {
  var detail = state.candidateDetail;
  if (!detail) return;
  await createJob(runPath("/debug-search"), {
    candidate_id: detail.candidate.id
  });
  closeDrawer();
}

async function repairSelectedTrial() {
  var row = state.selectedTrial;
  if (!row || row.debug) return;
  var payload = {
    task_id: row.task_id || row.id || undefined,
    attempt: row.attempt || undefined,
    manifest: row.manifest || undefined
  };
  await createJob(runPath("/repair-trial"), payload);
}

function bindEvents() {
  el("refresh-runs").addEventListener("click", function() { loadRuns().catch(showError); });
  el("plan-run").addEventListener("click", function() { planRun().catch(showError); });
  el("auto-run").addEventListener("click", function() { autoRun().catch(showError); });
  el("run-plan").addEventListener("click", function() { startSavedPlan(false).catch(showError); });
  el("resume-run").addEventListener("click", function() { startSavedPlan(true).catch(showError); });
  el("collect-run").addEventListener("click", function() { collectRun().catch(showError); });
  el("adopt-runtime").addEventListener("click", function() { adoptRuntime().catch(showError); });
  el("close-drawer").addEventListener("click", closeDrawer);
  el("show-candidate-artifacts").addEventListener("click", showCandidateArtifacts);
  el("debug-rerun").addEventListener("click", function() { debugRerun().catch(showError); });
  el("debug-search").addEventListener("click", function() { debugSearch().catch(showError); });
  el("candidate-filter").addEventListener("input", renderCandidates);
  el("trial-filter").addEventListener("input", renderTrials);
  el("candidate-sort").addEventListener("change", renderCandidates);
  el("trial-sort").addEventListener("change", renderTrials);
  el("repair-trial").addEventListener("click", function() { repairSelectedTrial().catch(showError); });
  el("refresh-jobs").addEventListener("click", function() { loadJobs().catch(showError); });
  el("stop-job").addEventListener("click", function() { stopSelectedJob().catch(showError); });
  el("close-artifact").addEventListener("click", closeArtifact);
  el("reload-artifact").addEventListener("click", function() { loadArtifact(state.currentArtifactPath).catch(showError); });
  document.querySelectorAll(".tab").forEach(function(tab) {
    tab.addEventListener("click", function() { setActiveTab(tab.dataset.tab); });
  });
}

function showError(error) {
  console.error(error);
  window.alert(error.message || String(error));
}

async function refreshLoop() {
  try {
    await loadJobs();
    if (state.selectedRunId) {
      var anyRunning = state.jobs.some(function(job) { return job.status === "running"; });
      if (anyRunning) await selectRun(state.selectedRunId, { preserveArtifacts: true });
    }
  } catch (error) {
    console.warn(error);
  }
  setTimeout(refreshLoop, 4000);
}

async function init() {
  bindEvents();
  await loadSettings();
  await loadRuns();
  await loadJobs();
  refreshLoop();
}

init().catch(showError);
