#!/usr/bin/env python
"""Independent evaluation & visualisation script (stage-2 of the pipeline).

Example
-------
uv run python -m src.evaluate \
        results_dir=./outputs \
        run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats

# ----------------------------------------------------------------------------
#                         CLI / ARGUMENT PARSING                              
# ----------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multiple runs via WandB API")
    parser.add_argument("results_dir", type=str, help="Directory to store evaluation outputs")
    parser.add_argument("run_ids", type=str, help="JSON string list of run IDs to evaluate")
    return parser.parse_args()

# ----------------------------------------------------------------------------
#                             UTILITIES                                       
# ----------------------------------------------------------------------------

def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_fig(fig: plt.Figure, dest: Path):
    fig.tight_layout()
    fig.savefig(dest)
    print(dest)
    plt.close(fig)


# Binary-class confusion (Correct vs Incorrect) ------------------------------

def _binary_confusion(preds: Sequence[str], gts: Sequence[str]):
    tp = sum(int(p.strip() == g.strip()) for p, g in zip(preds, gts))  # true positives (correct)
    fn = len(preds) - tp                                               # false negatives (incorrect prediction)
    fp = 0                                                             # no negative label predicted as positive in this framing
    tn = 0                                                             # likewise
    # We still want a 2×2 matrix for visual consistency
    return np.array([[tp, fp], [fn, tn]]), ["Correct", "Incorrect"], ["Predicted Correct", "Predicted Incorrect"]


# Primary metric name used throughout the project ---------------------------
PRIMARY_METRIC = "best_val_em"  # ↗ exact-match accuracy (higher = better)

# ----------------------------------------------------------------------------
#                                  MAIN                                       
# ----------------------------------------------------------------------------

def main():
    args = _parse_args()
    out_root = _mkdir(Path(args.results_dir).expanduser().absolute())

    # ---------------------------------------------------------------------
    # 1)  Global WandB project info                                         
    # ---------------------------------------------------------------------
    cfg_global = OmegaConf.load(Path(__file__).resolve().parent.parent / "config" / "config.yaml")
    entity, project = cfg_global.wandb.entity, cfg_global.wandb.project

    api = wandb.Api()
    run_ids: List[str] = json.loads(args.run_ids)

    # Containers for aggregated statistics --------------------------------
    aggregated: Dict[str, Dict[str, float]] = {}
    primary_metric_values: Dict[str, float] = {}

    # ---------------------------------------------------------------------
    # 2)  Per-run processing                                                
    # ---------------------------------------------------------------------
    for rid in run_ids:
        print(f"[evaluate] processing {rid} …", file=sys.stderr)
        run_out_dir = _mkdir(out_root / rid)

        run: wandb.apis.public.Run = api.run(f"{entity}/{project}/{rid}")
        history_df: pd.DataFrame = run.history()  # all time-series metrics
        summary: Dict = run.summary._json_dict
        config: Dict = dict(run.config)

        # ----------------------  Export raw metrics  ----------------------
        metrics_json_path = run_out_dir / "metrics.json"
        with metrics_json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": history_df.to_dict(orient="list"),
                    "summary": summary,
                    "config": config,
                },
                f,
                indent=2,
            )
        print(metrics_json_path)

        # ----------------------  Learning curves  -------------------------
        if not history_df.empty:
            # Loss curves
            if {"train_loss", "val_loss"}.intersection(history_df.columns):
                fig, ax = plt.subplots(figsize=(6, 4))
                if "train_loss" in history_df:
                    sns.lineplot(x=history_df.index, y=history_df["train_loss"], label="train_loss", ax=ax)
                if "val_loss" in history_df:
                    sns.lineplot(x=history_df.index, y=history_df["val_loss"], label="val_loss", ax=ax)
                ax.set_xlabel("step")
                ax.set_ylabel("loss")
                ax.set_title(f"{rid} – Loss curve")
                _save_fig(fig, run_out_dir / f"{rid}_loss_curve.pdf")

            # EM curve
            if "val_em" in history_df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.lineplot(x=history_df.index, y=history_df["val_em"], ax=ax)
                ax.set_xlabel("step")
                ax.set_ylabel("Exact-match accuracy")
                ax.set_title(f"{rid} – EM curve")
                _save_fig(fig, run_out_dir / f"{rid}_em_curve.pdf")

        # ----------------------  Confusion matrix  ------------------------
        try:
            val_file = next((f for f in run.files() if f.name.endswith("val_preds.json")), None)
            preds, gts = [], []
            if val_file is not None:
                local_json = val_file.download(replace=True).name  # returns str path
                with open(local_json, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    preds = d.get("pred", [])
                    gts = d.get("gt", [])
            if preds and gts:
                conf_mat, y_labels, x_labels = _binary_confusion(preds, gts)
                fig, ax = plt.subplots(figsize=(4, 4))
                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                ax.set_xticklabels(x_labels)
                ax.set_yticklabels(y_labels, rotation=0)
                ax.set_title(f"{rid} – Confusion matrix")
                _save_fig(fig, run_out_dir / f"{rid}_confusion_matrix.pdf")
        except Exception as e:  # noqa: BLE001 – ensure evaluation never crashes
            print(f"[warn] confusion matrix for {rid} failed: {e}", file=sys.stderr)

        # ----------------------  Collect metrics --------------------------
        # Priority: summary first (already final/best), else last value in history
        def _get_metric_value(name: str):
            if name in summary:
                return summary[name]
            if name in history_df.columns:
                return history_df[name].dropna().iloc[-1]
            return None

        # Iterate over all numeric summary keys
        for key, val in summary.items():
            if isinstance(val, (int, float)):
                aggregated.setdefault(key, {})[rid] = float(val)
        # Some explicit keys from history we always want
        for explicit in ("val_em", "val_loss", "train_loss", "epoch_train_loss"):
            v = _get_metric_value(explicit)
            if v is not None:
                aggregated.setdefault(explicit, {})[rid] = float(v)

        # primary metric collection
        pm_val = _get_metric_value(PRIMARY_METRIC)
        if pm_val is not None:
            primary_metric_values[rid] = float(pm_val)

    # ---------------------------------------------------------------------
    # 3) Aggregated analysis                                                
    # ---------------------------------------------------------------------
    comp_dir = _mkdir(out_root / "comparison")

    # ----------------------  Bar chart (primary) -------------------------
    if primary_metric_values:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=list(primary_metric_values.keys()), y=list(primary_metric_values.values()), ax=ax)
        for idx, val in enumerate(primary_metric_values.values()):
            ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom")
        ax.set_ylabel(PRIMARY_METRIC)
        ax.set_title("Comparison – primary metric across runs")
        _save_fig(fig, comp_dir / "comparison_primary_metric_bar.pdf")

    # ----------------------  Box-plots per metric ------------------------
    for metric, run_dict in aggregated.items():
        if len(run_dict) < 2:
            continue  # box not meaningful for single value
        data = [v for v in run_dict.values()]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(y=data, ax=ax)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} – distribution across runs")
        _save_fig(fig, comp_dir / f"comparison_{metric}_box.pdf")

    # ----------------------  Metrics table figure ------------------------
    df_table = pd.DataFrame(aggregated).T  # metrics × runs
    fig, ax = plt.subplots(figsize=(1 + 1.2 * len(df_table.columns), 0.5 + 0.3 * len(df_table)))
    ax.axis("off")
    table = ax.table(
        cellText=np.round(df_table.values, 4),
        rowLabels=df_table.index,
        colLabels=df_table.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    _save_fig(fig, comp_dir / "comparison_metrics_table.pdf")

    # ----------------------  Identify best proposed / baseline ----------
    best_proposed = {"run_id": None, "value": -np.inf}
    best_baseline = {"run_id": None, "value": -np.inf}
    for rid, val in primary_metric_values.items():
        if any(tok in rid.lower() for tok in ("proposed", "lift", "ours")):
            if val > best_proposed["value"]:
                best_proposed = {"run_id": rid, "value": val}
        elif any(tok in rid.lower() for tok in ("baseline", "comparative", "mup", "fixed", "auto", "c-tap")):
            if val > best_baseline["value"]:
                best_baseline = {"run_id": rid, "value": val}

    gap = None
    if best_proposed["run_id"] and best_baseline["run_id"]:
        direction = "maximize" if not any(s in PRIMARY_METRIC.lower() for s in ("loss", "error", "perplex")) else "minimize"
        if direction == "maximize":
            gap = (best_proposed["value"] - best_baseline["value"]) / best_baseline["value"] * 100
        else:  # lower is better
            gap = (best_baseline["value"] - best_proposed["value"]) / best_baseline["value"] * 100

    # ----------------------  Statistical significance --------------------
    proposed_vals = [v for r, v in primary_metric_values.items() if any(tok in r.lower() for tok in ("proposed", "lift", "ours"))]
    baseline_vals = [v for r, v in primary_metric_values.items() if any(tok in r.lower() for tok in ("baseline", "comparative", "mup", "fixed", "auto", "c-tap"))]
    ttest_p = None
    if len(proposed_vals) >= 2 and len(baseline_vals) >= 2:
        t_res = stats.ttest_ind(proposed_vals, baseline_vals, equal_var=False)
        ttest_p = float(t_res.pvalue)

    # ----------------------  Aggregated JSON -----------------------------
    agg_json = {
        "primary_metric": "Exact-match accuracy on GSM8K dev; secondary: (a) divergence count, (b) per-seed variance, (c) fraction of steps b_t>1.",
        "metrics": aggregated,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "ttest_pvalue": ttest_p,
    }
    agg_path = comp_dir / "aggregated_metrics.json"
    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(agg_json, f, indent=2)
    print(agg_path)

    # ----------------------  Print summary paths -------------------------
    for generated in comp_dir.glob("*.pdf"):
        print(generated)


if __name__ == "__main__":
    main()