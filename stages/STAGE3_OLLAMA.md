# Stage 3 AI agent refinement with Ollama and Qwen

Build an LLM agent that autonomously proposes, evaluates, and refines MoEPRS. Use **Qwen through a local Ollama server**. The model interprets diagnostics and chooses experiments; your Python tools fit models, enforce the budget, and return evidence. Students completing Stage 3 must use an actual LLM controller. A rule-based controller or grid search is used only as a comparison.

Ollama and the local Qwen model require **no paid subscription or API key** for this workflow. Download the software, Python packages, and model once; inference then runs on your computer. You need sufficient memory, disk space, and time. Run the setup check early and contact the instructor if your hardware cannot complete it. Do not use an Ollama cloud model for the required runs.

## 1 Install and verify the local model

Install Ollama from https://ollama.com/download and open it. On Linux, run `ollama serve` if the server is not already running. Download the course model:

```bash
ollama pull qwen3:8b
ollama list
```

From the project directory, create a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, use `python -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell instead of `source`. In every command below, use the Python interpreter from this environment. The simulator dependency `msprime` is unnecessary unless you regenerate the data.

Make a first model call:

```python
from ollama import Client

client = Client(host="http://localhost:11434")
reply = client.chat(
    model="qwen3:8b",
    think=False,
    messages=[{
        "role": "user",
        "content": "Explain why a gate with too many PCs can overfit MoEPRS.",
    }],
)
print(reply.message.content)
```

Use `qwen3:8b`, `think=False`, and temperature 0 for the main comparison. Record the model digest from `ollama list`, Ollama version, Python package versions, hardware, and runtime. These settings improve comparability but do not guarantee identical generations. A smaller local model such as `qwen3:4b` may be used with instructor agreement; label its results separately.

## 2 Reproduce the starter agent

The release includes `qwen_prs_agent.py`. It selects the five supplied VIPRS columns as fixed experts, fits a linear Stage 2 baseline, and exposes two tools: `evaluate_variant` and `select_variant`. It lets Qwen vary the gate PC count and penalty, returns validation diagnostics, and saves the actual conversation and tool results.

Run this demonstration from the project directory:

```bash
python qwen_prs_agent.py --project . --data data/shared_effects --model qwen3:8b --budget 6 --out outputs/qwen_demo
```

If the released datasets have anonymous names, replace the example data path with an assigned dataset path. Do not pass a regime name to the LLM. Inspect `outputs/qwen_demo/trace.json` and `predictions.csv`. Find an actual tool call, its hypothesis, the returned result, and the agent's final selection. A final answer without tool calls does not demonstrate an agent loop.

This demonstration uses one validation split. Its six-evaluation budget includes the baseline. It is a setup exercise, not the completed Stage 3 submission or evidence of a performance improvement.

## 3 Preserve the MoE model constraints

Reuse the Stage 1 ancestry-specific PRS as fixed experts. For the tutorial, the supplied VIPRS columns provide a fallback. In the final experiment, use the same fixed expert panel for Stage 2, the AI agent, and the comparison controller; document whether the scores were fitted by your team or supplied.

Keep a softmax gate with nonnegative weights summing to one. The gate may read genetic PCs and transformations of PCs. Age and sex remain global additive covariate effects. Do not refit variant weights, feed raw genotypes into a replacement predictor, put age or sex in the gate, or replace MoEPRS with boosting or an unrestricted neural network in Stage 3.

Permitted refinements include fewer gate PCs, stronger gate regularization, a small regularized nonlinear PC basis, and monotone expert-score transformations. An aggregate monotone spline is also permitted as an explicit extension: apply it to the **genetic mixture score**, then add global covariate effects. Do not transform the full prediction including age and sex. Keep an identity-link option and penalize unsupported departures from it.

## 4 Build a genuine LLM experiment loop

Adapt the starter so the model repeatedly receives diagnostics and experiment history, states a hypothesis, calls a fitting/evaluation tool, reads the result, and decides to continue or finalize. Python executes only registered tools and validates their arguments. Multiple named agents and arbitrary generated-code execution are not required.

Expose at least the following functions, either separately or within the evaluation tool:

| Tool | Responsibility |
| --- | --- |
| Diagnose | Return residual structure and model stability from held-out predictions |
| Evaluate a variant | Fit a permitted configuration and compare it with Stage 2 |
| Read experiment history | Return prior hypotheses, outcomes, failures, and remaining budget |
| Select a variant | Finalize an already evaluated model with a reason |

Implement the gate PC/penalty choices and **at least one nonlinear refinement family**. Qwen should choose which configurations to investigate from evidence. Explain why a residual-PC correlation alone does not prove a gate defect; PRS-by-PC residual associations provide a more targeted check. Residual curvature against the genetic mixture score motivates a link correction. The model may conclude that Stage 2 should be retained.

## 5 Improve evaluation before expanding search

For the submitted controller, replace the demonstration's repeated ranking on one validation split with **three-fold cross-validation inside the training split**. Every candidate and Stage 2 must use identical folds, individuals, and preprocessing rules. Fit preprocessing on the fitting fold only. Each training individual receives a prediction from a model that did not fit that individual's label; these are out-of-fold predictions.

Return incremental R2 and mean squared error, plus the paired change in squared error relative to Stage 2. Positive values of the following difference indicate improvement:

```python
gain = (y - baseline_oof_prediction)**2 - (y - candidate_oof_prediction)**2
mean_gain = gain.mean()
```

Report uncertainty and ancestry-region diagnostics. An individual-level standard error or bootstrap conditional on frozen predictions is descriptive: it does not account for overlapping training folds, model fitting, or adaptive selection. Do not treat the three folds as independent datasets or claim statistical significance from repeated use of the same individuals. Across independent simulation replicates, report paired method differences with their standard errors on common support.

Use **at most twelve candidate evaluations per dataset per controller, including the Stage 2 baseline**. One candidate evaluation consists of all three fold fits and their pooled out-of-fold assessment. Twelve candidates therefore require up to 36 fitting operations, not twelve. Count failed fitting attempts; cache duplicate configurations and return their existing evidence. Log model calls and total fits separately. Set a finite LLM-turn limit.

Before inspecting validation results, finalize one candidate using the cross-validation evidence and a documented selection rule. Fit that candidate and Stage 2 on the full training split and assess them once on validation. Predeclare a confirmation rule; a suitable simple rule is to fall back to Stage 2 if the candidate's validation MSE is worse. Do not use confirmation to rerank all candidates or restart search. The two final train fits and two validation assessments are separate from the twelve-candidate search allowance and are available equally to both controllers. Keep the selected training fit when producing test predictions.

A calibration curve or spline must be fitted without using the labels of the individuals on which that candidate is scored. Use an inner training split or inner cross-fitting where needed, and count those additional fits in the compute log. The existing incremental-R2 helper recalibrates against evaluation labels, so also report error from the frozen prediction and discuss that distinction.

## 6 Test whether the AI controller adds value

Compare three methods: Stage 2, your Ollama/Qwen agent, and a fixed or random search controller over the same permitted candidate space. Give the two search controllers the same twelve-candidate allowance, folds, confirmation rule, and data access. Include Stage 2 in each search. Fix random seeds and the candidate space before final assessment.

Use designated development datasets or replicates to write prompts and debug tools. Freeze the prompt, tools, candidate space, budget, and selection rule before running the assigned assessment replicates. Do not tune the policy using their test results or the hidden regime. Keep instructor test labels and oracle metadata out of the agent's context.

Submit test predictions for all three methods. Test labels are used only by the instructor for final assessment; test feedback must not become an input to the refinement loop. Report validation gains honestly and distinguish them from test gains supplied by the instructor. Do not treat the per-dataset best candidate chosen on test as an unbiased estimate of attainable improvement.

## 7 Submit reproducible evidence

Submit:

1. Runnable Python code, environment instructions, model identity/digest, settings, and seeds.
2. Complete LLM messages and tool-call traces for each assigned run, including hypotheses, results, errors, revisions, final selection, and any fallback.
3. A candidate table with out-of-fold scores, paired gains and uncertainty, confirmation results, evaluation counts, fit counts, and wall-clock runtime.
4. Test predictions for Stage 2, the AI agent, and the comparison controller on every assigned dataset and replicate.
5. A report explaining what the AI decisions added, which part of the MoE appeared misspecified, and whether improvements were consistent. Use all assigned replicates; when ten per regime are provided, report paired contrasts over those ten rather than one favorable run.

Credit depends on actual LLM-driven experimentation, correct MoE constraints, fair comparisons, and honest interpretation. A well-supported negative result or justified retention of Stage 2 receives full credit for the investigation. Model size and a positive performance gain are not grading requirements. A rule-based search alone, a fixed sweep with AI-written commentary, or a trace reconstructed after the experiment does not satisfy the AI-agent requirement.

## References

- Ollama local setup: https://docs.ollama.com/quickstart
- Ollama Python tool calling: https://docs.ollama.com/capabilities/tool-calling
- Course model: https://ollama.com/library/qwen3:8b

## Tools included in this student package

`cv_tools.py` supplies a common-fold, twelve-candidate evaluator with fold-specific preprocessing, descriptive paired error changes, and a single validation confirmation. `fixed_search.py` uses it for a seeded non-LLM comparison. Wire the same evaluator into your own Ollama agent so both controllers use the same candidate space and folds. Extend the evaluator if you implement another permitted link family, and expose that family to both controllers.

The `qwen_prs_agent.py` demonstration intentionally still uses one validation split. Replacing that tool with the CV evaluator is part of Stage 3. The supplied tools are scaffolding: submit your changes, derivations, and interpretation rather than treating a reference run as your implementation.
