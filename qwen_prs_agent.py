"""Teaching example: Qwen selects and evaluates fixed-expert MoE variants.

Run from the original PRSAgents directory, or pass --project explicitly.
This demonstrates tool calling, not an improved Stage 3 evaluation protocol.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--data", default="data/shared_effects")
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--scores", help="Optional Stage 1 target_scores.csv")
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("qwen_agent_output"))
    parser.add_argument("--offline-smoke", action="store_true")
    args = parser.parse_args()
    if args.budget < 1 or args.max_turns < 1:
        parser.error("budget and max-turns must be positive")
    sys.path.insert(0, str(args.project.resolve()))
    from harness import Dataset, Harness
    from stages.stage1_moeprs import MoEPRS

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = args.project / data_path
    d = Dataset(str(data_path), args.replicate)
    from stage2_baselines import load_experts
    d = load_experts(d, args.scores)
    harness = Harness(d, budget=args.budget)
    S, _, C, y, _ = d.part("train")
    Sv, _, Cv, yv, _ = d.part("val")
    results, predictors = {}, {}
    selected = None
    baseline_pred = None

    def evaluate_variant(n_pc: int, l2_gate: float, reason: str) -> str:
        """Fit a permitted MoE and return validation evidence.

        Args:
            n_pc: Number of gate PCs: 2, 3, 5, or 10.
            l2_gate: Gate penalty: 0.001, 0.01, 0.1, or 1.0.
            reason: Hypothesis motivating this experiment.
        """
        nonlocal baseline_pred
        if n_pc not in (2, 3, 5, 10) or l2_gate not in (0.001, 0.01, 0.1, 1.0):
            raise ValueError("Configuration outside the permitted search space")
        name = f"pc{n_pc}_l2{l2_gate:g}"
        if name in results:
            return json.dumps(results[name])
        if harness.used >= harness.budget:
            raise ValueError("Evaluation budget exhausted; select an evaluated model")
        model = MoEPRS(n_pc=n_pc, l2_gate=l2_gate).fit(S, C, y)
        predict = lambda S2, Z2, C2: model.predict(S2, C2)
        pred = model.predict(Sv, Cv)
        score = float(harness.evaluate(predict, note=reason))
        if baseline_pred is None:
            baseline_pred = pred.copy()
        gain = (yv - baseline_pred) ** 2 - (yv - pred) ** 2
        residual = yv - pred
        genetic = (model.mixing(Cv) * Sv * model.gamma).sum(axis=1)

        def corr(a, b):
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        result = {
            "name": name, "reason": reason,
            "validation_incremental_r2": score,
            "validation_mse": float(np.mean((yv - pred) ** 2)),
            "paired_mse_gain_vs_baseline": float(gain.mean()),
            "paired_gain_se_conditional_on_fits": float(gain.std(ddof=1) / np.sqrt(len(gain))),
            "residual_pc1_correlation": corr(residual, Cv[:, 2]),
            "residual_genetic_squared_correlation": corr(residual, genetic ** 2),
            "evaluations_remaining": harness.budget - harness.used,
        }
        results[name], predictors[name] = result, predict
        print(json.dumps(result, indent=2), flush=True)
        return json.dumps(result)

    def select_variant(name: str, reason: str) -> str:
        """Finalize one already evaluated model.

        Args:
            name: Model name returned by evaluate_variant.
            reason: Evidence supporting selection, including uncertainty.
        """
        nonlocal selected
        if name not in results:
            raise ValueError("Select an already evaluated model")
        selected = {"name": name, "reason": reason}
        return json.dumps(selected)

    baseline = evaluate_variant(10, 0.01, "Required linear Stage 2 baseline")
    baseline_name = json.loads(baseline)["name"]
    messages = [
        {"role": "system", "content": (
            "You are an experimental ML agent refining MoEPRS. Source PRS are fixed; "
            "the softmax gate reads PCs only and covariate effects are global. "
            "Use tools to test hypotheses by changing gate PC count or penalty. "
            "Explain each experiment briefly. Positive paired MSE gain is better. "
            "SE is conditional on fixed fits, not corrected for adaptive selection. "
            "Do not claim a significant improvement from a tiny noisy difference. "
            "You may retain the baseline. Finish by calling select_variant. "
            "You have no access to test labels."
        )},
        {"role": "user", "content": "Baseline evidence:\n" + baseline},
    ]
    if not args.offline_smoke:
        from ollama import Client
        client = Client(host="http://localhost:11434", timeout=300)
        tools = {"evaluate_variant": evaluate_variant, "select_variant": select_variant}
        for _ in range(args.max_turns):  # Bound model turns and ML evaluations.
            response = client.chat(
                model=args.model, messages=messages,
                tools=list(tools.values()), think=False,
                options={"temperature": 0, "num_predict": 2048},
            )
            message = response.message
            messages.append(message.model_dump(exclude_none=True))
            if message.content:
                print(message.content, flush=True)
            for call in message.tool_calls or []:
                try:
                    result = tools[call.function.name](**call.function.arguments)
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})
                messages.append({"role": "tool", "tool_name": call.function.name,
                                 "content": result})
            if selected is not None:
                break
            if not message.tool_calls:
                messages.append({"role": "user", "content":
                                 "Please call an experiment tool or select_variant."})
    if selected is None:
        selected = {"name": baseline_name, "reason": "Fallback: agent did not finalize"}
    args.out.mkdir(parents=True, exist_ok=True)
    trace = {"model": args.model, "offline_smoke": args.offline_smoke,
             "evaluation_protocol": "single validation split demonstration",
             "max_turns": args.max_turns,
             "selected": selected, "evaluations_used": harness.used,
             "experiments": results, "messages": messages}
    (args.out / "trace.json").write_text(json.dumps(trace, indent=2))
    harness.submit(predictors[selected["name"]], str(args.out / "predictions.csv"))
    print("Selected:", selected, "\nOutput:", args.out.resolve())


if __name__ == "__main__":
    main()
