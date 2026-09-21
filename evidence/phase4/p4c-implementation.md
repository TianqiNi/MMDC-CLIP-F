Implemented the P4C candidate without commits, pushes, P5 execution, downloads, or patient-data access. It is ready for fresh independent review; no acceptance or G4 claim is made.

### Scope delivered

- New `view-risk-smoke` package CLI:
  - All 15 nonempty masks, including four singletons.
  - RSNA and DDSM fusion trees.
  - Clean, permitted Gaussian-noise, held-out contrast, and common-mode blur cases.
  - Signed-effect support: `-1: 16`, `0: 48`, `+1: 16`.
  - Verified frozen features, regenerated current targets, and fixed classifier predictions.
  - One-epoch partial fit/artifact/checkpoint reload for `candidate` and `same_input_mlp`, retaining the frozen 20-epoch public configuration.
  - Reloaded MSP and fitted DS-logistic artifacts.
  - P4A confidence/effect metrics and complete 14-cell clean-mask evaluation.
- New read-only `view-risk-preflight` CLI.
- Protected output handling with validated staging/final paths, no-clobber publication, and success/failure cleanup.
- CI now installs `pytest>=8,<9`, lints research tests, and runs the entire research suite including the public smoke path.
- No production dependency changes.

Examples:

```bash
mmdc-clip-f view-risk-preflight --device cpu

mmdc-clip-f view-risk-smoke --device cpu

mmdc-clip-f view-risk-smoke \
  --output-dir .cache/my-new-p4c-smoke \
  --device cpu
```

The retained evidence is [report.json](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/.cache/p4c-final-smoke-2/report.json), file SHA-256 `6c3364bb22bc30175315921202538c1430019f941703b7dcf872f754d9365487`. Its timing-independent scientific digest is `3e9eebfb72cd8e8a9386b19506aea1b8cbfe49ee530381f3a9673df2b3bb194c`.

### RED/GREEN evidence

RED:

```text
PYTHONPATH=src .../.venv/bin/python -m pytest -q \
  --basetemp=.cache/p4c-red tests/research/test_view_risk_smoke.py

ImportError: cannot import name 'smoke'
1 collection error in 0.60s
```

Final GREEN:

```text
Focused P4C tests: 7 passed in 2.76s
Full tests/research: 268 passed in 22.83s
Ruff src tests/research: All checks passed
compileall src tests/research: passed
git diff --check: passed
CLI help, smoke help, and preflight: passed
Default CPU CLI smoke: exit 0 in 2.1s
```

The sole warning was the existing unavailable-NVML warning; preflight reports CUDA as unavailable.

### Synthetic-only cost sample

Batch 1, `[1,3,224,224]`, float32 CPU, 2 warmups and 5 timed calls:

| Component | Median ms | P25–P75 ms |
|---|---:|---:|
| Encoder/feature validation | 3.511 | 3.473–3.550 |
| Cache reload | 2.317 | 2.312–2.317 |
| Candidate head | 3.189 | 3.169–3.197 |
| MSP control | 1.318 | 1.308–1.385 |
| DS control | 1.251 | 1.250–1.259 |
| Synthetic end-to-end | 13.131 | 13.093–13.157 |

Exact measured parameters: candidate `433,436`; same-input MLP `433,651`. One persisted cache occupies `34,406` bytes. CPU memory is correctly labeled process-lifetime high-water RSS: `494,587,904–496,922,624` bytes, not an operation-specific tensor peak.

These are tiny injected-model timings—not CLIP deployment measurements or evidence of efficiency improvement. Actual-backbone latency, accelerator peak memory, real cache costs, empirical benefit, pilot go/no-go, and real patient readiness remain unavailable. Locally cached pinned resources were detected but not content-audited or loaded; private data and patient mapping remain unavailable/not audited.

Changed files:

- [smoke.py](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/src/mmdc_clip_f/research/view_risk/smoke.py)
- [cli.py](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/src/mmdc_clip_f/research/view_risk/cli.py)
- [test_view_risk_smoke.py](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/tests/research/test_view_risk_smoke.py)
- [ci.yml](/home/tianqini/research/.ivr-p4c-worktrees/p4c-smoke/.github/workflows/ci.yml)

Intermediate task caches were removed; only the final git-ignored evidence bundle was retained.