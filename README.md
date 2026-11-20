# LEAF — Lipid Evolutionary Agentic Framework

In-silico, multi-agent, budget-constrained active-learning simulation for
ionizable lipid design. See [`LEAF_Research_Plan.md`](LEAF_Research_Plan.md)
(Revision 2) for the full research plan and [`LEAF_draft_paper.tex`](LEAF_draft_paper.tex)
for the paper draft. `leaf/` is the codebase; `scripts/` runs the full
experiment suite.

## 1. Environment

**Use a dedicated environment — do not run experiments in a shared/base
environment.** Package-version drift between runs is exactly the kind of
unreported confound RQ7 (reproducibility) exists to catch, and `rdkit`
in particular is easiest to get right via conda.

```bash
conda env create -f environment.yml
conda activate leaf
pip install -e .
```

No conda? A pinned `requirements.txt` is provided as a fallback (rdkit
wheels are less reliable via pip on some platforms):

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the `leaf` package in editable mode (via
`pyproject.toml`) — without it, `import leaf` and `python scripts/*.py`
fail with `ModuleNotFoundError: No module named 'leaf'` unless you
manually set `PYTHONPATH`. Editable means code edits take effect
immediately with no reinstall needed.

Verify the install:

```bash
pytest leaf/tests -v
```

All 18 tests should pass in a few seconds. If `test_vocabulary.py` is
skipped, `rdkit` did not install correctly — fix that before trusting any
downstream result, since it gates fragment validity (research plan §6.1
Step 1).

## 2. Status — what's real vs. placeholder

This is a working scaffold of the **corrected** architecture from Revision
2: a frozen ground-truth oracle separate from the per-round active-learning
surrogate, an explicit acquisition function, a corrected budget-accounting
orchestrator loop, and the statistical/leakage-probe fixes in the plan's §5.
The pipeline below runs end-to-end today and produces real plots — but
**the actual numbers are not yet reportable** because these pieces are
still placeholders:

| Placeholder | File | Must be replaced with |
|---|---|---|
| Fragment SMILES vocabulary | `leaf/fragments/vocabulary.py` | The literature-derived vocabulary + a chemically correct Ugi-3CR assembler |
| Harmonized labeled data | `leaf/data/raw/`, `leaf/data/processed/` | Real Xu et al. + AGILE data, harmonized per §5.8 (see `data/raw/README.md`) |
| $\mathcal{O}_{\text{GT}}$ | fit inline in `run_experiment.py::_fit_placeholder_oracle` | A real oracle fit on harmonized data, frozen via `GroundTruthOracle.save(...)`, validated by E1 |
| LLM Proposer/Verifier/Hypothesis calls | `agents/proposer.py`, `verifier.py`, `hypothesis.py` | Real LLM calls constrained by the JSON schemas / prompts in `agents/prompts/` (currently deterministic rule-based stand-ins, which is fine for E2's static-GA/random-search baselines but not for the "leaf" method itself) |
| LipoAgent wrapper | `baselines/lipoagent_runner.py` | Vendored public LipoAgent code (SAI-Lab-NYU, arXiv:2605.25250) |

Every script below prints a warning when it is using a placeholder, and
`scripts/run_all_experiments.py` refuses to silently pretend otherwise.

## 3. Running experiments

### 3.1 One-command full sweep

```bash
conda activate leaf
python scripts/run_all_experiments.py --experiments e1 e2 e3 e4 e6
```

This runs, for every experiment: every (method, split, seed) combination
its config specifies (`leaf/configs/e*.yaml`), saves raw results, and
generates every figure/table in the §8 reporting checklist. E5 and E8 are
excluded from the default sweep because they have preconditions (below)
and are run individually.

Add `--dry-run` first to see the full run plan (methods × splits × seeds)
without executing anything — useful for sanity-checking a config edit or
estimating wall-clock time before committing to a full ≥8-seed sweep.

Once a real frozen oracle exists (see §2 above and E1 below), pass it in
so results stop being smoke-test-only:

```bash
python scripts/run_all_experiments.py --experiments e2 e3 e4 e6 --oracle path/to/oracle_gt.pkl
```

### 3.2 Individual experiments

**E1 — Oracle validation (run this first, always).** Fits $\mathcal{O}_{\text{GT}}$
and evaluates it on random/scaffold-cold/chemotype-cold splits against the
pre-registered go/no-go threshold (§5.11). This has its own script because,
unlike E2–E8, it does not run the round loop:

```bash
python scripts/run_e1_oracle_validation.py --config leaf/configs/e1_oracle_validation.yaml
```

Do not trust any E2–E8 result on a split this script flags
`UNINTERPRETABLE`.

**E2–E4, E6 — via the sweep driver** (§3.1), or one (method, split, seed)
triple at a time for debugging:

```bash
python -m leaf.eval.run_experiment \
    --config leaf/configs/e2_agentic_value_add.yaml \
    --experiment e2 --method leaf --split random --seed 0
```

**E5 — Hypothesis rediscovery (leakage-probe-gated).** Run the pretraining-
leakage probe first and disclose the result regardless of outcome (§5.5):

```python
from leaf.agents.leakage_probe import run_leakage_probe, e5_status
results = run_leakage_probe(your_llm_call_fn)   # bare LLM, no RAG, no archive
print(e5_status(results))   # "external_validation" or "consistency_check"
```

Only proceed to run the Hypothesis Agent's rediscovery test
(`leaf/configs/e5_hypothesis_rediscovery.yaml`) after this, and report
whichever status was returned — never label it "external validation" if
the probe found leakage.

**E8 — External validation (conditional).** Only run if the LUMI-lab
supplementary dataset is actually released and licensed for reuse; track
its availability in `leaf/data/raw/README.md`.

### 3.3 Config parity checklist (before trusting an E2 result)

Every baseline **must** share LEAF's acquisition function and `q`
(research plan §3.2) — `leaf/eval/run_experiment.py` raises if a config
omits `acquisition_function`. Do not hand-edit a baseline's config to use
a different acquisition setting without also updating LEAF's.

## 4. Reproducibility

Every run writes a `manifest.json` next to its results
(`results/<experiment>/<method>_<split>_seed<seed>/manifest.json`)
containing: git commit hash + dirty-tree flag, the fully resolved config,
the seed, exact package versions, Python/platform info, and a UTC
timestamp (`leaf/eval/reproducibility.py`). If you see a "not a git
repository" or "uncommitted changes" warning while running experiments you
intend to report in the paper, commit first — an uncommitted-tree result
cannot be tied to a specific code version and is not citable.

Seed counts are fixed in the configs at **≥8** for every experiment
reporting a significance claim (E2–E4), per §5.2/§7.2 — Wilcoxon
signed-rank cannot reach $p<0.05$ at $n=3$. Do not reduce seed counts to
save compute for a number that will appear in the paper; reduce them only
for `--dry-run`-style debugging.

## 5. Where results and plots land

```
results/<experiment>/<method>_<split>_seed<seed>/
    manifest.json     # reproducibility record (§4)
    result.json       # summary metrics (final hypervolume, queries used, hypothesis precision)
    rounds.csv         # per-round log: hypervolume, coverage, inter-agent agreement, budget

figures/<experiment>/
    hypervolume_curves_<split>.{png,pdf}       # mean±std HV vs. round, one line per method
    agentic_value_add_<split>.{png,pdf}        # bootstrap-CI bar for the primary H1 claim
    coverage_vs_round_<method>_<split>_seed*.{png,pdf}
    inter_agent_agreement_<method>_<split>.{png,pdf}
    seed_sensitivity_<split>.{png,pdf}         # RQ7 variance-budget boxplot
    summary_table_<split>.csv                  # raw numbers behind every figure above

results/e1/
    split_validation_table.csv                 # RQ5 generalization gap per split
    search_space_size.json                     # |X| (RQ6)
figures/e1/
    oracle_split_validation.{png,pdf}
```

Every figure is saved as both `.png` (quick viewing) and `.pdf`
(camera-ready, vector). Nothing is plot-only-in-a-notebook — if a number
appears in the paper, its figure and the CSV behind it are both on disk
under `figures/` and `results/`.

To regenerate figures from existing results without re-running experiments
(e.g. after a plotting-style tweak):

```python
from leaf.eval import plotting
plotting.plot_hypervolume_curves("e2", ["leaf", "static_ga", "random_search"], "random", list(range(8)))
```

## 6. Repository layout

See §4 of `LEAF_Research_Plan.md` for the annotated repository structure
and §6 for the gated build order each module must satisfy before the next
one is wired in.
