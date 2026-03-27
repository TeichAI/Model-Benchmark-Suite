# Model-Benchmark-Suite TODO

## Status

- [x] Map benchmark suite entry points and evaluation flow
- [x] Inspect current failure signals from `eval.log`
- [x] Review saved `lm_eval` raw result structure
- [x] Identify concrete broken paths and rough-edge issues
- [x] Implement high-priority fixes tracked in this file
- [ ] Re-test affected flows and update documentation if needed

## Confirmed Bugs

### 1. Benchmark failures are silently converted into fake success payloads

- **Priority:** High
- **Files:** `main.py`, `app.py`
- **Problem:** `main.py` catches `lm_eval` failures, writes a JSON payload anyway, and exits without a non-zero failure status.
- **Impact:** The Streamlit app can treat a failed benchmark like a valid run and display `0` or misleading results instead of surfacing the error.
- **Planned fix:** Propagate failures via exit status and make the UI detect/report invalid result payloads.
- **Status:** Completed

### 2. DeepEval task name inference breaks for task names with underscores

- **Priority:** High
- **Files:** `benchmarks/run_deepeval.py`
- **Problem:** DeepEval infers task names from filenames using a right split on `_`, which breaks names like `gpqa_diamond_zeroshot`.
- **Impact:** DeepEval cannot find matching `samples` entries for many tasks.
- **Planned fix:** Infer task names from the actual result payload instead of parsing filenames.
- **Status:** Completed

### 3. DeepEval parser does not match real `lm_eval` sample structure

- **Priority:** High
- **Files:** `benchmarks/run_deepeval.py`
- **Problem:** The current parser expects keys like `query`, `question`, `ctx`, and `choices`, but real samples often contain `Question`, `target`, `arguments`, processed choice fields, and `filtered_resps`.
- **Impact:** DeepEval skips items or produces incomplete analysis.
- **Planned fix:** Parse actual `lm_eval` sample structures robustly across MC and generation-style tasks.
- **Status:** Completed

### 4. Windows local model paths can break cache and output filenames

- **Priority:** High
- **Files:** `app.py`
- **Problem:** Filename sanitization only replaces `/`, not `\\`, `:`, or other Windows-invalid filename characters.
- **Impact:** Local model paths on Windows can break result writing and cache reuse.
- **Planned fix:** Introduce a shared filename sanitization helper for model identifiers and paths.
- **Status:** Completed

### 5. Cache reuse ignores config differences

- **Priority:** High
- **Files:** `app.py`
- **Problem:** Cached results are reused even if backend, quantization, batch size, few-shot settings, or generation kwargs have changed.
- **Impact:** Stale results can be shown as if they came from the latest requested config.
- **Planned fix:** Compare cached config to current config and invalidate or warn on mismatch.
- **Status:** Completed

### 6. CPU fallback likely uses unsafe dtype assumptions

- **Priority:** Medium
- **Files:** `benchmarks/run_lm_eval.py`
- **Problem:** CPU fallback still uses `dtype=bfloat16` in the non-quantized path.
- **Impact:** CPU execution can fail or behave inconsistently depending on backend/model support.
- **Planned fix:** Use safer dtype handling for CPU fallback.
- **Status:** Completed

### 7. `vllm` is selectable without a hard unsupported-platform guard

- **Priority:** Medium
- **Files:** `main.py`, `app.py`
- **Problem:** The app warns that `vllm` is Linux/WSL-only, but still allows unsupported selections.
- **Impact:** Users on unsupported environments can trigger avoidable failures.
- **Planned fix:** Add validation before launch and surface a clear error in the UI/CLI.
- **Status:** Completed

### 8. Benchmark defaults encourage non-comparable runs

- **Priority:** High
- **Files:** `app.py`, `main.py`, `benchmarks/run_lm_eval.py`
- **Problem:** The default quantization setting was low-bit, and the UI did not clearly flag generation overrides or sampling as benchmark-destabilizing settings.
- **Impact:** Users can produce materially worse and less reproducible scores than public leaderboard-style evaluations without realizing the run is experimental.
- **Planned fix:** Make full-precision the default path and add explicit warnings when low-bit quantization or sampling are enabled.
- **Status:** Completed

### 9. Reports described current UI state more clearly than actual run provenance

- **Priority:** High
- **Files:** `app.py`
- **Problem:** Result views and exports emphasized current sidebar settings instead of the actual per-run configuration captured in saved results and raw `lm_eval` payloads.
- **Impact:** Mixed-config runs could look directly comparable even when they were produced with different backends, quantization modes, few-shot overrides, limits, or sampling settings.
- **Planned fix:** Extract run provenance from saved configs and raw payload metadata, surface a comparability table in the UI, and use that same data in exports.
- **Status:** Completed

## Rough Edges

### 10. Mutable default argument in `run_lm_eval`

- **Priority:** Low
- **Files:** `benchmarks/run_lm_eval.py`
- **Problem:** `tasks_list` defaults to a mutable list.
- **Impact:** Low immediate risk, but poor correctness hygiene.
- **Planned fix:** Change default to `None` and normalize inside the function.
- **Status:** Completed

### 11. Result file and cache wrapper flows are inconsistent

- **Priority:** Low
- **Files:** `app.py`, `main.py`
- **Problem:** Raw results and cached wrapper payloads use different shapes and are handled differently in different paths.
- **Impact:** Increases maintenance cost and makes future bugs more likely.
- **Planned fix:** Normalize result loading logic behind one helper.
- **Status:** Completed

## Execution Plan

### Phase 1: Correctness and failure handling

- [x] Fix failure propagation from `main.py` to the Streamlit app
- [x] Prevent invalid benchmark result payloads from being treated as successful runs
- [x] Add config-aware cache validation

### Phase 2: Result-path robustness

- [x] Add Windows-safe filename sanitization helper
- [x] Use the helper consistently for raw output and cache files

### Phase 3: DeepEval repair

- [x] Replace filename-based task inference with payload-based task discovery
- [x] Support real `lm_eval` sample structures for multiple-choice tasks
- [x] Improve fallback parsing for generation-style tasks
- [x] Harden custom DeepEval model compatibility if needed

### Phase 4: Runtime guards and cleanup

- [x] Add unsupported-platform guardrails for `vllm`
- [x] Fix CPU-safe dtype selection
- [x] Remove mutable default arguments and clean up result-loading rough edges

### Phase 5: Benchmark trust and reporting

- [x] Change benchmark defaults in the UI, CLI, and runner to prefer full-precision comparable runs
- [x] Add run provenance and comparability diagnostics to the Streamlit results view and exports

## Progress Log

### Completed

- [x] Audited CLI entrypoint in `main.py`
- [x] Audited benchmark runner in `benchmarks/run_lm_eval.py`
- [x] Audited DeepEval post-processing in `benchmarks/run_deepeval.py`
- [x] Audited Streamlit orchestration and cache flow in `app.py`
- [x] Inspected `saved_results` raw payload structure against parser expectations
- [x] Updated `main.py` to exit non-zero on `lm_eval` failure instead of writing fake-success payloads
- [x] Updated `app.py` to reject invalid `lm_eval` payloads from cache or disk
- [x] Added Windows-safe reversible model filename encoding with legacy file lookup compatibility
- [x] Updated `app.py` to use shared cache/raw/DeepEval result path helpers
- [x] Rewrote `benchmarks/run_deepeval.py` to parse actual `lm_eval` MC and generation sample payloads
- [x] Updated the DeepEval custom model wrapper to support schema-based generation when required
- [x] Updated `app.py` to invalidate stale cache entries when run config does not match the current request
- [x] Updated `run_lm_eval.py` to remove the mutable default argument and use safer CPU dtype selection
- [x] Added early native-Windows `vllm` guardrails in both the CLI and Streamlit UI
- [x] Normalized raw-result and cache-wrapper loading through shared helpers in `app.py`
- [x] Changed benchmark defaults in the UI, CLI, and runner to prefer full-precision comparable runs
- [x] Added run provenance and comparability diagnostics to the Streamlit results view and exports
- [x] Forced dark foreground text for highlighted comparison rows in `app.py`
- [x] Auto-disabled chat templating for incompatible multiple-choice benchmarks in `app.py` and `benchmarks/run_lm_eval.py`

### In Progress

- [ ] Re-test the updated flows and update documentation if needed
