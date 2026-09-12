import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import json
from math import sqrt
import types as _types

# File: src/module_engine.py
# Minimal implementations for the pipeline's engines.
# Drop this file into your project's src/ directory. It exports run_all_modules
# and also injects lightweight scenario_engine, statistics_engine and
# report_generator modules into sys.modules so the main script can import them.

import matplotlib.pyplot as plt

def run_all_modules(n_users=1200, seed=20260316):
    """
    Produce synthetic outputs and simple evaluation metrics for 6 modules.
    Returns (outputs_dict, metrics_dict).
    outputs_dict: { "module_1": DataFrame, ... }
    metrics_dict: { "module_1": {"pass_rate": float, "avg_score": float}, ... }
    """
    rng = np.random.default_rng(seed)
    outputs = {}
    metrics = {}

    # Common user ids
    user_ids = np.arange(1, n_users + 1)

    # Module 1: Consent profiles
    m1 = pd.DataFrame({
        "user_id": user_ids,
        "consent_given": rng.choice([0, 1], size=n_users, p=[0.25, 0.75]),
        "executor_nomination": rng.choice(["exec_A", "exec_B", "exec_C"], size=n_users),
    })
    m1["m1_pass"] = (m1["consent_given"] == 1).astype(int)
    outputs["module_1"] = m1
    metrics["module_1"] = {
        "pass_rate": m1["m1_pass"].mean() * 100,
        "executor_A_pct": (m1["executor_nomination"] == "exec_A").mean() * 100
    }

    # Module 2: Identity events
    verified = rng.choice([0,1], size=n_users, p=[0.15,0.85])
    inactivity_days = rng.poisson(30, size=n_users)
    m2 = pd.DataFrame({
        "user_id": user_ids,
        "verified": verified,
        "inactivity_days": inactivity_days,
    })
    m2["confidence_score"] = np.clip( (verified * 80 + (30 - np.minimum(inactivity_days,30)) * rng.random(n_users)), 0, 100 )
    outputs["module_2"] = m2
    metrics["module_2"] = {
        "avg_confidence": float(m2["confidence_score"].mean()),
        "verified_pct": float((m2["verified"]==1).mean() * 100)
    }

    # Module 3: Verification cases
    cases = int(n_users * 0.2)
    m3 = pd.DataFrame({
        "case_id": np.arange(1, cases+1),
        "user_id": rng.choice(user_ids, size=cases),
        "evidence_strength": rng.uniform(0,100,size=cases),
        "executor_authority": rng.choice([0,1], size=cases, p=[0.3,0.7])
    })
    m3["decision_score"] = np.clip(m3["evidence_strength"] * (0.5 + 0.5 * m3["executor_authority"]), 0, 100)
    m3["decision"] = pd.cut(m3["decision_score"], bins=[-1,39,79,100], labels=["rejected","review","verified"])
    outputs["module_3"] = m3
    metrics["module_3"] = {
        "verified_pct": float((m3["decision"]=="verified").mean() * 100),
        "avg_decision_score": float(m3["decision_score"].mean())
    }

    # Module 4: Content objects
    obj_count = n_users
    categories = ["public","sensitive","restricted"]
    m4 = pd.DataFrame({
        "object_id": np.arange(1, obj_count+1),
        "owner_id": user_ids,
        "category": rng.choice(categories, size=obj_count, p=[0.6,0.3,0.1]),
        "consent_required": rng.choice([0,1], size=obj_count, p=[0.5,0.5])
    })
    m4["access_granted"] = ((m4["category"]!="restricted") & (m4["consent_required"]==0)).astype(int)
    outputs["module_4"] = m4
    metrics["module_4"] = {
        "restricted_granted_pct": float(((m4["category"]=="restricted") & (m4["access_granted"]==1)).mean() * 100),
        "access_granted_pct": float(m4["access_granted"].mean() * 100)
    }

    # Module 5: Threat events
    events = int(n_users * 0.3)
    m5 = pd.DataFrame({
        "event_id": np.arange(1, events+1),
        "user_id": rng.choice(user_ids, size=events),
        "anomaly_score": rng.uniform(0,100,size=events)
    })
    m5["risk_score"] = np.clip(m5["anomaly_score"] + rng.normal(0,10,size=events), 0, 100)
    m5["protective_action"] = (m5["risk_score"] > 70).astype(int)
    outputs["module_5"] = m5
    metrics["module_5"] = {
        "protective_action_pct": float(m5["protective_action"].mean() * 100),
        "avg_risk": float(m5["risk_score"].mean())
    }

    # Module 6: Governance audits
    audits = int(n_users * 0.1)
    m6 = pd.DataFrame({
        "audit_id": np.arange(1, audits+1),
        "module_reviewed": rng.choice(list(metrics.keys()), size=audits),
        "completeness_score": rng.uniform(0,100,size=audits)
    })
    outputs["module_6"] = m6
    metrics["module_6"] = {
        "avg_completeness": float(m6["completeness_score"].mean())
    }

    return outputs, metrics

# --- Create lightweight scenario_engine, statistics_engine, report_generator modules ---
def _make_scenario_engine_module():
    mod = types.ModuleType("scenario_engine")

    def run_scenario_simulation(metrics, n_runs=1000, seed=20260322):
        rng = np.random.default_rng(seed)
        # Build a simple scenario: each run samples module pass/score noise and computes two approaches
        rows = []
        for i in range(n_runs):
            # sample aggregated scores from metrics with noise
            m_scores = {k: max(0.0, min(100.0, rng.normal(v.get(next(iter(v)),0) if isinstance(v, dict) else 0, 5)))
                        for k, v in metrics.items()}
            # define two approach scores (ilf and google) as functions of module metrics
            ilf_score = (
                metrics["module_1"]["pass_rate"] * 0.2 +
                metrics["module_2"]["avg_confidence"] * 0.25 +
                metrics["module_4"]["access_granted_pct"] * 0.15 +
                rng.normal(0,5)
            )
            google_score = (
                metrics["module_2"]["avg_confidence"] * 0.3 +
                metrics["module_3"]["avg_decision_score"] * 0.2 +
                metrics["module_5"]["avg_risk"] * -0.1 +
                rng.normal(0,5)
            )
            rows.append({
                "run_id": i + 1,
                "ilf_score": float(np.clip(ilf_score, 0, 100)),
                "google_score": float(np.clip(google_score, 0, 100)),
            })
        scen = pd.DataFrame(rows)
        means = scen[["ilf_score","google_score"]].mean().rename("mean_score").reset_index()
        means.columns = ["approach","mean_score"]
        return scen, means

    mod.run_scenario_simulation = run_scenario_simulation
    return mod

def _make_statistics_engine_module():
    mod = types.ModuleType("statistics_engine")

    def statistical_summary(scen_df):
        # simple descriptive statistics for numeric columns
        num = scen_df.select_dtypes(include=[np.number])
        summary = num.agg(["mean","median","std","min","max"]).transpose().reset_index().rename(columns={"index":"metric"})
        return summary

    def comparative_tests(scen_df):
        # quick t-test style comparison on ilf vs google (use simple difference distribution)
        if {"ilf_score","google_score"}.issubset(scen_df.columns):
            diff = scen_df["ilf_score"] - scen_df["google_score"]
            mean_diff = float(diff.mean())
            std_diff = float(diff.std(ddof=1))
            t_stat = mean_diff / (std_diff / sqrt(len(diff))) if std_diff > 0 else float("nan")
            p_two_sided = float(2 * (1 - 0.5)) if len(diff)==0 else None  # placeholder, we avoid scipy dependency
            # keep conservative p placeholder
            p_two_sided = float(min(1.0, np.exp(-abs(t_stat)/10))) if (not np.isnan(t_stat)) else None
            return pd.DataFrame([{
                "comparison":"ilf_vs_google",
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "p_value_approx": p_two_sided
            }])
        else:
            return pd.DataFrame([])

    def hypothesis_summary():
        rows = [
            {"hypothesis":"H1","description":"ILF mean score > Google mean score","status":"exploratory"}
        ]
        return pd.DataFrame(rows)

    mod.statistical_summary = statistical_summary
    mod.comparative_tests = comparative_tests
    mod.hypothesis_summary = hypothesis_summary
    return mod

def _make_report_generator_module():
    mod = types.ModuleType("report_generator")

    def create_figures(module_metrics_df, statistical_summary_df, figures_dir):
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        # Bar chart of module metrics (pick a numeric metric per module if available)
        try:
            pivot = module_metrics_df.pivot(index="module", columns="metric", values="value").fillna(0)
            ax = pivot.plot(kind="bar", figsize=(8,4), title="Module metrics")
            fig_path = figures_dir / "module_metrics.png"
            ax.figure.tight_layout()
            ax.figure.savefig(fig_path)
            plt.close(ax.figure)
        except Exception:
            # fallback simple plot
            fig_path = figures_dir / "module_metrics.png"
            plt.figure(figsize=(6,3))
            plt.text(0.5,0.5,"module metrics unavailable", ha="center")
            plt.axis("off")
            plt.savefig(fig_path)
            plt.close()

        # Summary scatter of approach means if present
        try:
            if "metric" in statistical_summary_df.columns:
                p = Path(figures_dir) / "stat_summary.json"
                statistical_summary_df.to_json(p, orient="records")
        except Exception:
            pass
        return True

    def generate_overleaf(outputs_root, overleaf_dir):
        outputs_root = Path(outputs_root)
        overleaf_dir = Path(overleaf_dir)
        overleaf_dir.mkdir(parents=True, exist_ok=True)
        # create a minimal main.tex referencing CSVs
        tex = r"""\documentclass{article}
\begin{document}
\section*{Pipeline outputs}
This overleaf folder contains pipeline outputs (CSV files). See the outputs directory.
\end{document}
"""
        (overleaf_dir / "main.tex").write_text(tex)
        # write a small index of outputs
        index = {"outputs_root": str(outputs_root.resolve())}
        (overleaf_dir / "outputs_index.json").write_text(json.dumps(index, indent=2))
        return True

    mod.create_figures = create_figures
    mod.generate_overleaf = generate_overleaf
    return mod

# Inject fake modules into sys.modules so subsequent imports in the main script succeed.
sys.modules.setdefault("scenario_engine", _make_scenario_engine_module())
sys.modules.setdefault("statistics_engine", _make_statistics_engine_module())
sys.modules.setdefault("report_generator", _make_report_generator_module())

# Expose run_all_modules as the primary function of this module.
__all__ = ["run_all_modules"]