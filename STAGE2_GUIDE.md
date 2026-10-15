# Stage 2 Use and evaluate a provided MoEPRS model

You do **not** need to derive EM or implement it from scratch. Use the supplied compact implementation as the default, or use the paper implementation with the same project inputs and evaluation rules.

## Default provided implementation

`moeprs.py` contains the model (about 120 lines, NumPy only). `stage2_demo.py` shows loading, fitting, validation, and test-prediction export. Run from this package:

```bash
python stage2_demo.py data/datasets/dataset_01/rep00
python stage2_baselines.py data/datasets/dataset_01/rep00 --out outputs/stage2_demo
```

To use your Stage 1 export, add `--scores outputs/stage1_demo/target_scores.csv` to either command. The following is the complete model API once training/validation arrays have been loaded:

```python
from moeprs import MoEPRS

# S: fixed source PRS columns; C: age, sex, PC1, ..., PC10.
# Preprocessing must be learned on the training data only.
model = MoEPRS(n_pc=10, l2_gate=0.01).fit(S_train, C_train, y_train)
predictions = model.predict(S_validation, C_validation)
gate_weights = model.mixing(C_validation)
```

The source variant weights stay fixed. EM learns scalar expert calibrations, shared global covariate effects, a shared residual variance, and a regularized PC-only softmax gate. Responsibilities in the E-step depend on both phenotype fit and gate probabilities; prediction uses gate probabilities alone because a new individual's phenotype is unknown. Age and sex enter the global adjustment, not the gate. Quadratic gate features and a fixed signed-power expert basis are supplied as possible Stage 3 extensions; Stage 2 uses linear options.

This is an independent teaching implementation of the same general framework, not a reproduction of the paper software or its reported results. The paper code offers additional losses, expert-specific variances/covariates and penalties, dataset/scaling utilities, and optimization controls. The teaching version uses a shared Gaussian variance, alternating weighted least-squares updates and an approximate Adam gate update. Its recorded training likelihood is unpenalized; because fitting includes a gate penalty and approximate updates, do not assume every step increases that trace or that the final point is a global optimum.

## What you must do

1. Explain fixed experts, gate weights, global covariate adjustment, and the E-step/M-step in your own words. A formal derivation is optional.
2. Run best single PRS, MultiPRS, genotype ridge, and the provided Stage 2 MoEPRS on the same splits and five experts.
3. Inspect `moe_training_loglik.csv`, `stage2_config.json`, and finite gate weights that sum to one. Plot weights against PC1 and PC2.
4. Run nested training sizes 250, 500, 1000 and 2000 with fixed validation/test individuals; tune only with allowed training labels.
5. Report paired comparisons across assigned replicates, explain uncertain or negative results, and freeze the Stage 2 incumbent before Stage 3.

The full baseline script exports validation metrics, gate weights, fitting diagnostics, and frozen test predictions. Test labels are withheld. You create the plots, replicate aggregation, learning curves, and scientific interpretation. Stage 3 must use the same declared incumbent and implementation in both controller comparisons.

## Optional paper implementation

Reference repository: https://github.com/shz9/moe-prs-paper

Read its README, `model/moe.py`, `model/PRSDataset.py`, and its example notebooks. Use a separate environment following that repository's dependency instructions; do not replace this package's pinned requirements. Record the exact repository commit, dependencies and settings. The full biobank preparation pipeline is unnecessary for the supplied simulated data.

Adapt the provided scores, labels, covariates and split IDs to the paper's dataset/API. Select the same five fixed PRS experts, PC1–PC10 for the gate, and age/sex/PCs for global adjustment; do not use expert-specific age/sex effects or test labels. Use a Gaussian outcome and record residual-variance, regularization, standardization, initialization, and optimizer settings. Fit scaling using training rows only and reuse it for validation/test. Consult the paper notebooks for the actual API rather than assuming the compact `fit(S, C, y)` interface applies to the research code.

If you choose this route, adapt both Stage 3 controllers to the same paper-code backend before comparing them; the included controllers use the compact backend by default. Additional adapter work is your choice, not a Stage 2 requirement. Numerical agreement with the compact model is not expected. A from-scratch implementation is also optional and earns no extra credit merely for replacing the provided code.
