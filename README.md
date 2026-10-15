# COMP 565 Fall 2026 Course Project PRSAgents

Read **COMP565_PRSAgents_project_description.pdf** first. The editable Word version is included too. This folder is self-contained: run commands here, without referring to the instructor's repository. All three stages are required in the complete project description.

## What is here

| File or folder | Purpose |
| --- | --- |
| Project description `.pdf` and `.docx` | Three stages, constraints, evaluation and deliverables |
| `data/datasets/dataset_01` to `dataset_04` | Four anonymous generating settings, ten replicates each |
| `data/stage1/` | Shared ancestry LD and six transfer-cohort genotype files |
| `dataset_catalog.csv` | Paths and split sizes for all forty replicates; no answer key |
| `harness.py` | Dataset loader and validation/prediction interface |
| `stages/stage1_fit.py`, `simulate/prs.py` | Source-fitting and transfer-table references |
| `fit_stage1.py` | Export fitted source weights, target scores, transfer matrix |
| `moeprs.py`, `stage2_demo.py` | Compact provided MoEPRS and minimal fit/predict example |
| `stages/stage1_moeprs.py` | Compatibility adapters and optional Stage 3 variants |
| `STAGE2_GUIDE.md` | Provided-model workflow and optional paper implementation |
| `stage2_baselines.py` or `baselines.py` | Core Stage 2 reference checks and frozen test predictions |
| `qwen_prs_agent.py` | Local Qwen tool-calling demonstration |
| `cv_tools.py`, `fixed_search.py` | Stage 3 CV scaffold and non-LLM comparison |
| `stages/STAGE3_OLLAMA.md` | Ollama setup, example calls, final experiment protocol |
| `paired_contrasts.py` | Compare methods on common replicate support |
| `check_package.py`, `manifest.json` | Data/split checks and file integrity |
| `requirements.txt`, `requirement.txt` | Identical pinned dependencies; either filename works |

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
python stages/stage1_fit.py data/datasets/dataset_01/rep00
python fit_stage1.py data/datasets/dataset_01/rep00 --out outputs/stage1_demo
```

The export writes `transfer_matrix.csv`, five source weight vectors with normalization parameters, and `target_scores.csv`. Target normalization uses training individuals only. The transfer reference uses each transfer cohort's genotype normalization as in the supplied conventions; it never uses target test labels.

## Stage 2 with the provided implementation

Start with the short fit/predict example:

```bash
python stage2_demo.py data/datasets/dataset_01/rep00
```

`moeprs.py` supplies the working model, using NumPy only. You may use it unchanged. Read `STAGE2_GUIDE.md` for the model assumptions, a minimal API example, and the alternative paper-code route. Then run the full required baseline comparison:

```bash
python stage2_baselines.py data/datasets/dataset_01/rep00 --out outputs/stage2_demo
```

This default uses the five supplied VIPRS score columns. To use your Stage 1 score export:

```bash
python stage2_baselines.py data/datasets/dataset_01/rep00 --scores outputs/stage1_demo/target_scores.csv --out outputs/stage2_fitted_scores
```

The output contains validation R2/MSE, a gate-weight table for plotting, a training likelihood trace, the declared Stage 2 configuration, and test predictions for best single PRS, MultiPRS, genotype ridge, and Stage 2 MoEPRS. Tuning uses training CV; genotype/covariate scaling is fit inside its tuning folds. `--train-size 250` (or 500/1000/2000) selects nested seeded training subsets for the required sample-size study. Test scores cannot be computed locally.

## Stage 3 with local Qwen

Install Ollama from https://ollama.com/download and open it. Run `ollama serve` if the local server is not already running.

```bash
ollama pull qwen3:8b
python qwen_prs_agent.py --project . --data data/datasets/dataset_01/rep00 --model qwen3:8b --budget 6 --out outputs/qwen_demo
```

No subscription, paid endpoint, or API key is needed for this local workflow. Model downloads require internet access and disk space; inference needs suitable hardware. See the Stage 3 guide for a first Python call and troubleshooting. The demo writes actual model/tool messages to `trace.json` and selected test predictions to `predictions.csv`. Add `--scores outputs/stage1_demo/target_scores.csv` to use your fitted scores.

For a check without an LLM server:

```bash
python qwen_prs_agent.py --project . --offline-smoke --out outputs/offline_check
```

This only checks ML fitting/export and cannot satisfy the AI-agent requirement. For the final assignment, wire your Ollama controller into the common training-CV tool rather than repeatedly ranking on the demo validation split. The comparison scaffold is:

```bash
python fixed_search.py data/datasets/dataset_01/rep00 --budget 12 --out outputs/comparison_demo
```

The comparison uses a predeclared seeded random order over the same supported PC count, penalty, quadratic-gate and power-link space. If you add another family, expose it to both controllers. You must record actual LLM hypotheses and tool calls, not generate a narrative after a fixed sweep.

## Data and splits

Each replicate has 4,000 individuals and 1,000 typed variants, split into 2,000 train, 800 validation, and 1,200 test. Target covariates are age, sex, PC1 to PC10. The supplied panel has ten score columns: VIPRS and LASSO for five discovery sources. The core project uses the five VIPRS experts consistently.

`labels_train.csv` contains only training and validation phenotypes. Test IDs are provided solely for prediction submission. Test labels, local ancestry truth, causal effects, founder pools, oracle numbers, regime keys, and prior solution results are not in this package. Separate Stage 1 transfer-cohort phenotypes are intentionally included for portability analysis.

Dataset names conceal the generating setting. infer it from evidence if requested; do not assume the numerical alias encodes a regime. Use development replicates to debug, then freeze the prompt/tools/policy before assessing different assigned replicates. Keep development and assessment results separate and use the instructor's designated allocation when supplied.

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
