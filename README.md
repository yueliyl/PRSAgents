# COMP565 Fall 2026 Course Project: PRSAgents

Read [the project description](COMP565_PRSAgents_project_description.pdf) first. All three stages are required. The editable Word version is also in this repository.

## Download and arrange the project

1. Get the scripts and documents from this repository. Choose **Code → Download ZIP** and extract it, or clone the repository:

   ```bash
   git clone https://github.com/yueliyl/PRSAgents.git
   cd PRSAgents
   ```

2. [Download PRSAgents_Data.zip](https://github.com/yueliyl/PRSAgents/releases/download/fall2026-v3/PRSAgents_Data.zip) from the [Fall 2026 data release](https://github.com/yueliyl/PRSAgents/releases/tag/fall2026-v3).
3. Extract the archive into the repository folder. Place the resulting **`data/` folder directly beside `harness.py` and `README.md`**, so an example dataset path is `data/shared_effects/`. If your unzip application creates an extra `PRSAgents_Data` folder, move its `data/` folder up one level.
4. Follow Setup below and run `python check_package.py --full` before starting the project.

The repository's source-code ZIP contains the scripts and documents. The release asset `PRSAgents_Data.zip` contains only the data. Download both; neither alone is a complete working project. Run subsequent commands from the repository root. No subscription or API key is needed for the local Ollama/Qwen workflow.

## Scripts and documentation

The repository contains the project description (`COMP565_PRSAgents_project_description.pdf` and editable `.docx`), this README, the Stage 2 guide (`STAGE2_GUIDE.md`), and the Ollama guide (`stages/STAGE3_OLLAMA.md`).

- **Load and check data:** `prsdata.py` is the low-level reader for the scenario folders and is included in this repository; `harness.py` wraps it to load a replicate; `check_package.py` checks the downloaded data against `manifest.json`; `dataset_catalog.csv` lists all 40 scenario/replicate combinations and split sizes.
- **Stage 1:** `stages/stage1_fit.py` is the source-fitting and transfer reference, `simulate/prs.py` supplies the variational-fitting helper, and `fit_stage1.py` exports source weights, target scores, and transfer matrices.
- **Stage 2:** `moeprs.py` supplies the compact model; `stage2_demo.py` gives a minimal example; `stage2_baselines.py` runs the baseline comparisons (`baselines.py` is its compatibility entry point). `models.py` supplies the shared baseline models. `stages/stage1_moeprs.py` provides compatibility adapters and optional Stage 3 variants.
- **Stage 3:** `qwen_prs_agent.py` demonstrates actual Ollama tool calls; `cv_tools.py` supplies the training-CV scaffold; `fixed_search.py` provides the non-LLM comparison; `paired_contrasts.py` compares results on common replicates.
- **Environment:** `requirements.txt` and `requirement.txt` contain the same pinned dependencies. The `__init__.py` files in `simulate/` and `stages/` make those directories importable. `.gitignore` excludes downloaded data, ZIPs, environments, and generated outputs from Git.

## Downloaded data

Only data is distributed as a ZIP. Extracting it creates four scenario folders (`shared_effects`, `ancestry_dependent`, `locus_varying`, `nonlinear_link`), each holding ten replicates stacked in one set of files, plus `data/common/` (shared ancestry LD and transfer-cohort genotypes). The data integrity manifest tracks these input files, so scripts and documentation can be updated independently.

The references are starting points, not a completed student submission. Implement and explain your Stage 1 source fitting. For Stage 2, use the provided compact MoEPRS or the paper implementation; EM derivation and implementation from scratch are not required. Explain the model, run the comparisons, and interpret the evidence. For Stage 3, extend the agent's tools, evaluation and decision policy. No simulator installation or data regeneration is needed. Do not distribute outputs from the instructor's report as your own results.

## Setup

Use **Python 3.12 or newer**. The older Python bundled with some macOS installations will not work with these pinned packages. In this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_package.py --full
```

Windows PowerShell: use `python -m venv .venv` then `.venv\Scripts\Activate.ps1`. Use the activated environment's `python` in subsequent commands. Ollama is a separate application; the Python `ollama` package alone does not install its server or model.

## Stage 1 check and export

```bash
python stages/stage1_fit.py data/shared_effects
python fit_stage1.py data/shared_effects --out outputs/stage1_demo
```

The export writes `transfer_matrix.csv`, five source weight vectors with normalization parameters, and `target_scores.csv`. Target normalization uses training individuals only. The transfer reference uses each transfer cohort's genotype normalization as in the supplied conventions; it never uses target test labels.

## Stage 2 with the provided implementation

Start with the short fit/predict example:

```bash
python stage2_demo.py data/shared_effects
```

`moeprs.py` supplies the working model, using NumPy only. You may use it unchanged. Read `STAGE2_GUIDE.md` for the model assumptions, a minimal API example, and the alternative paper-code route. Then run the full required baseline comparison:

```bash
python stage2_baselines.py data/shared_effects --out outputs/stage2_demo
```

This default uses the five supplied VIPRS score columns. To use your Stage 1 score export:

```bash
python stage2_baselines.py data/shared_effects --scores outputs/stage1_demo/target_scores.csv --out outputs/stage2_fitted_scores
```

The output contains validation R2/MSE, a gate-weight table for plotting, a training likelihood trace, the declared Stage 2 configuration, and test predictions for best single PRS, MultiPRS, genotype ridge, and Stage 2 MoEPRS. Tuning uses training CV; genotype/covariate scaling is fit inside its tuning folds. `--train-size 250` (or 500/1000/2000) selects nested seeded training subsets for the required sample-size study. Test scores cannot be computed locally.

## Stage 3 with local Qwen

Install Ollama from https://ollama.com/download and open it. Run `ollama serve` if the local server is not already running.

```bash
ollama pull qwen3:8b
python qwen_prs_agent.py --project . --data data/shared_effects --model qwen3:8b --budget 6 --out outputs/qwen_demo
```

No subscription, paid endpoint, or API key is needed for this local workflow. Model downloads require internet access and disk space; inference needs suitable hardware. See the Stage 3 guide for a first Python call and troubleshooting. The demo writes actual model/tool messages to `trace.json` and selected test predictions to `predictions.csv`. Add `--scores outputs/stage1_demo/target_scores.csv` to use your fitted scores.

For a check without an LLM server:

```bash
python qwen_prs_agent.py --project . --offline-smoke --out outputs/offline_check
```

This only checks ML fitting/export and cannot satisfy the AI-agent requirement. For the final assignment, wire your Ollama controller into the common training-CV tool rather than repeatedly ranking on the demo validation split. The comparison scaffold is:

```bash
python fixed_search.py data/shared_effects --budget 12 --out outputs/comparison_demo
```

The comparison uses a predeclared seeded random order over the same supported PC count, penalty, quadratic-gate and power-link space. If you add another family, expose it to both controllers. You must record actual LLM hypotheses and tool calls, not generate a narrative after a fixed sweep.

## Data and splits

### Scenarios and replicates

`data/` holds four scenario folders. Each is one generating rule, and each
holds **ten independent replicates** stacked along a leading array axis:

```python
from harness import Dataset, all_replicates
d = Dataset("data/locus_varying", replicate=3)      # one replicate
for d in all_replicates("data/ancestry_dependent"):  # all ten
    ...
```

The replicates are independently simulated cohorts, **not** cross-validation
folds. Differences between methods here are roughly 0.003 to 0.007 in
incremental R2, and the standard error on a single replicate is about the same
size, so compare methods paired within replicate and average over the ten.
`data/README.md` documents every file, array shape and gotcha.

Each replicate has 4,000 individuals and 1,000 typed variants, split into 2,000 train, 800 validation, and 1,200 test. Target covariates are age, sex, PC1 to PC10. The supplied panel has ten score columns: VIPRS and LASSO for five discovery sources. The core project uses the five VIPRS experts consistently.

`labels.npz` holds the phenotypes and a `split` array (0 train, 1 validation, 2 test); test phenotypes are NaN. Test IDs are available solely for prediction submission. Test labels, local ancestry truth, causal effects, founder pools, oracle numbers, regime keys, and prior solution results are not in this package. Separate Stage 1 transfer-cohort phenotypes are intentionally included for portability analysis.

Folder names state the generating scenario, and `data/README.md` describes each one. Use development replicates to debug, then freeze the prompt/tools/policy before assessing different assigned replicates. Keep development and assessment results separate and use the instructor's designated allocation when supplied.

`source_cohorts` and `heritability` are public simulation descriptors, not phenotype labels. Standardized supplied scores are a reference fallback; document their use and fit any additional preprocessing on the fitting fold only. Genotypes and phenotypes are simulated, not real 1000 Genomes individuals.

## Submission and analysis

Submit code, model/settings/seed records, source weights and score exports, transfer/baseline tables, plots, candidate-budget and runtime logs, and complete LLM/tool traces. Keep `outputs/` separate from input data. Every prediction CSV must have `iid,prediction` for the same 1,200 target test IDs. The instructor supplies final test assessment.

For a results CSV with `dataset,rep,method,score`, compare on common replicate support:

```bash
python paired_contrasts.py results.csv --baseline stage2_moe --candidate qwen_agent
```

Use either validation or instructor-returned test scores in a table, never a mixture. A positive performance gain is not guaranteed or required for full credit for the investigation.

## Troubleshooting

- Connection refused: start Ollama and check `ollama list`.
- Model missing: run `ollama pull qwen3:8b`.
- Missing Python module: activate the environment and install `requirements.txt`.
- Slow inference or insufficient memory: contact the instructor about a smaller local model and label that comparison clearly.
- The agent never calls tools: inspect `trace.json`; revise the system prompt/tool descriptions. A fallback selection is not evidence of successful autonomous refinement.

Official documentation: https://docs.ollama.com/quickstart and https://docs.ollama.com/capabilities/tool-calling.
