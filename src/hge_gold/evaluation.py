from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

from .config import PipelineConfig
from .io import atomic_json, write_csv, write_parquet
from .modeling import CLASS_ORDER, TASK_TARGET


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    truth = np.maximum(np.asarray(y_true, dtype=float), 1e-12)
    prediction = np.maximum(np.asarray(y_pred, dtype=float), 1e-12)
    return np.asarray(np.log(prediction) + truth / prediction, dtype=float)


def moving_block_bootstrap_mean(
    values: np.ndarray, n_bootstrap: int, block_length: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(values) / block_length))
    starts = rng.integers(0, max(1, len(values) - block_length + 1), size=(n_bootstrap, n_blocks))
    offsets = np.arange(block_length)
    indices = (starts[..., None] + offsets).reshape(n_bootstrap, -1)[:, : len(values)]
    means = values[indices].mean(axis=1)
    p_value = float(2 * min(np.mean(means <= 0), np.mean(means >= 0)))
    low, high = np.quantile(means, [0.025, 0.975])
    return p_value, float(low), float(high)


def _classification_probabilities(series: pd.Series) -> np.ndarray:
    rows: list[list[float]] = []
    for item in series:
        payload = json.loads(item)
        rows.append(
            [float(payload.get(str(label), payload.get(label, 0.0))) for label in CLASS_ORDER]
        )
    return np.asarray(rows)


def _metric_rows(group: pd.DataFrame) -> list[dict[str, object]]:
    task = str(group["task"].iloc[0])
    truth = group["y_true"].to_numpy()
    prediction = group["y_pred"].to_numpy()
    if task.endswith("classification"):
        values = {
            "macro_f1": f1_score(
                truth, prediction, labels=CLASS_ORDER, average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        }
        primary = "macro_f1"
    elif task == "volatility_regression":
        values = {
            "QLIKE": float(np.mean(qlike(truth, prediction))),
            "RMSE_variance": float(np.sqrt(mean_squared_error(truth, prediction))),
            "MAE_variance": float(mean_absolute_error(truth, prediction)),
        }
        primary = "QLIKE"
    else:
        values = {
            "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
            "MAE": float(mean_absolute_error(truth, prediction)),
            "directional_sign_accuracy": float(np.mean(np.sign(truth) == np.sign(prediction))),
        }
        primary = "RMSE"
    base = {
        key: group[key].iloc[0]
        for key in [
            "task",
            "horizon",
            "target_policy_id",
            "feature_set_id",
            "selected_model_id",
            "model_family",
        ]
    }
    return [
        {
            **base,
            "metric_name": name,
            "metric_value": float(value),
            "metric_valid": bool(np.isfinite(value)),
            "invalid_reason": None,
            "n_test": len(group),
            "primary_metric": name == primary,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        for name, value in values.items()
    ]


def run_evaluation(config: PipelineConfig) -> dict[str, Path]:
    paths = config.paths()
    base = pd.read_parquet(
        paths.data / "processed" / "modeling_base" / "phase3_modeling_base_gold_only.parquet"
    )
    locked = pd.read_parquet(
        paths.data / "predictions" / "phase4" / "phase4_locked_test_predictions.parquet"
    )
    selected_map = json.loads(
        (paths.artifacts / "metadata" / "phase4_selected_model_map.json").read_text(
            encoding="utf-8"
        )
    )
    selected_family = {
        item["selected_model_id"]: item["selected_model_family"]
        for item in selected_map["selected_models"]
    }
    target_frames: list[pd.DataFrame] = []
    for task, target in TASK_TARGET.items():
        subset = base[["row_id", "horizon", "target_policy_id", "feature_set_id", target]].copy()
        subset["task"] = task
        subset["target_name"] = target
        subset = subset.rename(columns={target: "y_true"})
        target_frames.append(subset)
    targets = pd.concat(target_frames, ignore_index=True)
    evaluation = locked.merge(
        targets,
        on=["row_id", "horizon", "target_policy_id", "feature_set_id", "task", "target_name"],
        how="left",
        validate="one_to_one",
    )
    if evaluation["y_true"].isna().any():
        raise RuntimeError("Required locked-test y_true is null after Phase 5 join")
    evaluation["model_family"] = evaluation["selected_model_id"].map(selected_family)
    evaluation_path = (
        paths.data / "evaluation" / "phase5" / "phase5_locked_test_evaluation_dataset.parquet"
    )
    write_parquet(evaluation_path, evaluation)

    metric_rows: list[dict[str, object]] = []
    for _, group in evaluation.groupby(["task", "horizon", "selected_model_id"], sort=False):
        metric_rows.extend(_metric_rows(group))
    metrics = pd.DataFrame(metric_rows)
    metadata = paths.artifacts / "metadata"
    metrics_path = metadata / "phase5_locked_test_metrics_report.csv"
    write_csv(metrics_path, metrics)

    locked_start = pd.to_datetime(evaluation["date"]).min()
    baseline_parts: list[pd.DataFrame] = []
    comparison_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    for (task, horizon, selected_id), group in evaluation.groupby(
        ["task", "horizon", "selected_model_id"], sort=False
    ):
        target = TASK_TARGET[str(task)]
        train = base[
            (base["horizon"] == horizon)
            & (pd.to_datetime(base["date"]) < locked_start)
            & base[target].notna()
        ]
        baseline = group[
            [
                "row_id",
                "date",
                "date_index",
                "horizon",
                "target_policy_id",
                "feature_set_id",
                "task",
                "target_name",
            ]
        ].copy()
        if str(task).endswith("classification"):
            counts = train[target].astype(int).value_counts()
            majority = int(counts.index[0])
            priors = np.array([counts.get(label, 0) for label in CLASS_ORDER], dtype=float)
            priors /= priors.sum()
            baseline["y_pred"] = majority
            baseline["y_pred_proba_json"] = json.dumps(
                dict(zip(CLASS_ORDER.tolist(), priors.tolist(), strict=True))
            )
            selected_metric = f1_score(
                group["y_true"],
                group["y_pred"],
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
            baseline_metric = f1_score(
                group["y_true"],
                baseline["y_pred"],
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )
            improvement = selected_metric - baseline_metric
            # Metric bootstrap is recomputed block-wise for temporal dependence.
            rng = np.random.default_rng(int(config.evaluation["bootstrap_seed"]))
            differences = np.empty(int(config.evaluation["bootstrap_iterations"]))
            truth = group["y_true"].to_numpy()
            pred = group["y_pred"].to_numpy()
            bpred = baseline["y_pred"].to_numpy()
            block = int(config.evaluation["block_length"])
            for index in range(len(differences)):
                starts = rng.integers(
                    0, max(1, len(truth) - block + 1), size=int(np.ceil(len(truth) / block))
                )
                indices = np.concatenate(
                    [np.arange(start, min(start + block, len(truth))) for start in starts]
                )[: len(truth)]
                differences[index] = f1_score(
                    truth[indices],
                    pred[indices],
                    labels=CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                ) - f1_score(
                    truth[indices],
                    bpred[indices],
                    labels=CLASS_ORDER,
                    average="macro",
                    zero_division=0,
                )
            lower_tail = float(np.mean(differences <= 0))
            upper_tail = float(np.mean(differences >= 0))
            p_value = 2 * min(lower_tail, upper_tail)
            ci_low, ci_high = np.quantile(differences, [0.025, 0.975])
            metric_name = "macro_f1"
            baseline_id = (
                "majority_class_static_v1"
                if task == "direction_classification"
                else "flat_or_majority_static_v1"
            )
        else:
            mean_value = float(train[target].mean())
            baseline["y_pred"] = mean_value
            baseline["y_pred_proba_json"] = None
            truth = group["y_true"].to_numpy()
            selected_pred = group["y_pred"].to_numpy()
            baseline_pred = baseline["y_pred"].to_numpy()
            if task == "volatility_regression":
                selected_loss = qlike(truth, selected_pred)
                baseline_loss = qlike(truth, baseline_pred)
                metric_name = "QLIKE"
                baseline_id = "historical_variance_static_v1"
            else:
                selected_loss = (truth - selected_pred) ** 2
                baseline_loss = (truth - baseline_pred) ** 2
                metric_name = "RMSE"
                baseline_id = "historical_mean_return_static_v1"
            selected_metric = (
                float(np.sqrt(selected_loss.mean()))
                if task == "return_regression"
                else float(selected_loss.mean())
            )
            baseline_metric = (
                float(np.sqrt(baseline_loss.mean()))
                if task == "return_regression"
                else float(baseline_loss.mean())
            )
            improvement = baseline_metric - selected_metric
            # Positive difference means selected model has lower loss.
            p_value, ci_low, ci_high = moving_block_bootstrap_mean(
                baseline_loss - selected_loss,
                int(config.evaluation["bootstrap_iterations"]),
                int(config.evaluation["block_length"]),
                int(config.evaluation["bootstrap_seed"]),
            )
        baseline["baseline_model_id"] = baseline_id
        baseline["baseline_generation_mode"] = "static_pre_locked_test_only"
        baseline["uses_locked_test_y_true"] = False
        baseline_parts.append(baseline)
        comparison_rows.append(
            {
                "task": task,
                "horizon": horizon,
                "selected_model_id": selected_id,
                "primary_comparison_baseline_id": baseline_id,
                "metric_name": metric_name,
                "selected_model_metric": selected_metric,
                "baseline_metric": baseline_metric,
                "improvement_absolute": improvement,
                "selected_better_than_baseline": improvement > 0,
                "baseline_selected_pre_locked_test_evaluation": True,
                "post_hoc_baseline_selection_used": False,
                "metric_valid": True,
            }
        )
        significance_rows.append(
            {
                "test_family_id": f"{task}_primary",
                "task": task,
                "horizon": horizon,
                "selected_model_id": selected_id,
                "baseline_model_id": baseline_id,
                "metric_name": metric_name,
                "test_method": "moving_block_bootstrap",
                "p_value": p_value,
                "confidence_interval_low": ci_low,
                "confidence_interval_high": ci_high,
                "n_test": len(group),
                "minimum_n_test": 30,
                "n_test_sufficient": len(group) >= 30,
                "n_bootstrap": int(config.evaluation["bootstrap_iterations"]),
                "bootstrap_seed": int(config.evaluation["bootstrap_seed"]),
                "block_length": int(config.evaluation["block_length"]),
                "test_valid": bool(np.isfinite(p_value)),
            }
        )

    baseline_predictions = pd.concat(baseline_parts, ignore_index=True)
    baseline_path = (
        paths.data / "evaluation" / "phase5" / "phase5_baseline_locked_test_predictions.parquet"
    )
    write_parquet(baseline_path, baseline_predictions)
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = metadata / "phase5_baseline_comparison_report.csv"
    write_csv(comparison_path, comparison)
    significance = pd.DataFrame(significance_rows)
    # Benjamini-Hochberg within task families.
    significance["q_value"] = np.nan
    for _, indexes in significance.groupby("test_family_id").groups.items():
        idx = list(indexes)
        p = significance.loc[idx, "p_value"].to_numpy(float)
        order = np.argsort(p)
        adjusted = np.empty_like(p)
        ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adjusted[order] = np.minimum(ranked, 1.0)
        significance.loc[idx, "q_value"] = adjusted
    significance_path = metadata / "phase5_statistical_significance_report.csv"
    write_csv(significance_path, significance)

    outer_agg = pd.read_csv(metadata / "phase4_validation_metric_aggregation_report.csv")
    gap_rows: list[dict[str, object]] = []
    for _, selected in pd.DataFrame(selected_map["selected_models"]).iterrows():
        test_metric = metrics[
            (metrics["task"] == selected.task)
            & (metrics["horizon"] == selected.horizon)
            & metrics["primary_metric"]
        ]
        validation = outer_agg[
            (outer_agg["task"] == selected.task)
            & (outer_agg["horizon"] == selected.horizon)
            & (outer_agg["candidate_model_id"] == selected.selected_model_id)
        ]
        if test_metric.empty or validation.empty:
            continue
        val = float(validation.iloc[0]["metric_value"])
        test = float(test_metric.iloc[0]["metric_value"])
        higher = str(selected.task).endswith("classification")
        deterioration = (val - test) if higher else ((test - val) / max(abs(val), 1e-12))
        large = deterioration > (0.10 if higher else 0.25)
        gap_rows.append(
            {
                "task": selected.task,
                "horizon": selected.horizon,
                "selected_model_id": selected.selected_model_id,
                "metric_name": test_metric.iloc[0]["metric_name"],
                "validation_metric_value": val,
                "locked_test_metric_value": test,
                "absolute_gap": test - val,
                "relative_gap": (test - val) / max(abs(val), 1e-12),
                "large_gap_flag": large,
                "claim_downgrade_required": large,
            }
        )
    gap_path = metadata / "phase5_validation_test_gap_report.csv"
    write_csv(gap_path, pd.DataFrame(gap_rows))

    backtest_output = run_backtest(config, evaluation)
    claims: list[dict[str, object]] = []
    for _, row in comparison.iterrows():
        stat = significance[
            (significance["task"] == row.task) & (significance["horizon"] == row.horizon)
        ].iloc[0]
        status = (
            "SUPPORTED"
            if row.selected_better_than_baseline and stat.q_value <= 0.05
            else (
                "CONDITIONALLY_SUPPORTED" if row.selected_better_than_baseline else "NOT_SUPPORTED"
            )
        )
        claims.append(
            {
                "claim_id": f"C_{row.task}_h{row.horizon}",
                "claim_type": "predictive_performance",
                "task": row.task,
                "horizon": int(row.horizon),
                "status": status,
                "claim_status_policy_id": "phase5_claim_status_policy_v1",
                "comparison_baseline_id": row.primary_comparison_baseline_id,
                "metric_evidence": str(metrics_path.relative_to(paths.root)),
                "baseline_evidence": str(comparison_path.relative_to(paths.root)),
                "statistical_evidence": str(significance_path.relative_to(paths.root)),
                "paper_grade_allowed": False,
            }
        )
    claim_path = metadata / "phase5_claim_registry.json"
    atomic_json(claim_path, {"claims": claims})
    limitation_path = metadata / "phase5_limitation_registry.json"
    atomic_json(
        limitation_path,
        {
            "limitations": [
                {
                    "limitation_id": "L1",
                    "limitation_text": (
                        "Execution uses deterministic non-confidential sample data and "
                        "gold-only features; results are not paper-grade market evidence."
                    ),
                    "severity": "HIGH",
                    "phase_to_resolve": "real_data_audit_extension",
                }
            ]
        },
    )
    return {
        "evaluation": evaluation_path,
        "metrics": metrics_path,
        "baseline": baseline_path,
        "comparison": comparison_path,
        "significance": significance_path,
        "gap": gap_path,
        "claims": claim_path,
        "limitations": limitation_path,
        **backtest_output,
    }


def run_backtest(config: PipelineConfig, evaluation: pd.DataFrame) -> dict[str, Path]:
    paths = config.paths()
    signals = evaluation[
        (evaluation["task"] == "trade_action_classification")
        & (evaluation["horizon"] == int(config.evaluation["primary_backtest_horizon"]))
    ].copy()
    prices = (
        pd.read_parquet(paths.data / "processed" / "prices" / "validated_gold_price_series.parquet")
        .sort_values("date")
        .reset_index(drop=True)
    )
    prices["execution_price"] = prices["close"].shift(-1)
    prices["exit_price"] = prices["close"].shift(-2)
    signals = signals.merge(
        prices[["date", "execution_price", "exit_price"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    signals["position"] = signals["y_pred"].astype(int)
    signals["is_executable"] = signals[["execution_price", "exit_price"]].notna().all(axis=1)
    signals["trade_status"] = np.where(signals["is_executable"], "EXECUTABLE", "NOT_EXECUTABLE")
    signals["included_in_performance_metrics"] = signals["is_executable"]
    executable = signals[signals["is_executable"]].copy()
    executable["position_previous"] = executable["position"].shift(fill_value=0)
    executable["position_change"] = (executable["position"] - executable["position_previous"]).abs()
    executable["gross_log_return"] = executable["position"] * np.log(
        executable["exit_price"] / executable["execution_price"]
    )
    base_cost = float(config.targets["transaction_cost_bps"]) + float(
        config.targets["slippage_bps"]
    )
    executable["round_trip_cost_bps"] = base_cost
    executable["total_cost"] = executable["position_change"] * base_cost / 10_000
    executable["net_log_return"] = executable["gross_log_return"] - executable["total_cost"]
    executable["turnover"] = executable["position_change"]
    executable["exposure"] = executable["position"].abs()
    signal_path = paths.data / "backtests" / "phase5" / "phase5_trade_signal_dataset.parquet"
    ledger_path = paths.data / "backtests" / "phase5" / "phase5_trade_ledger.parquet"
    write_parquet(signal_path, signals)
    write_parquet(ledger_path, executable)
    performance_rows: list[dict[str, object]] = []
    robustness_rows: list[dict[str, object]] = []
    for cost in config.evaluation["costs_bps"]:
        net = executable["gross_log_return"] - executable["position_change"] * float(cost) / 10_000
        equity = np.exp(net.cumsum())
        drawdown = equity / equity.cummax() - 1
        sharpe = (
            float(np.sqrt(252) * net.mean() / net.std(ddof=1))
            if net.std(ddof=1) > 0
            else float("nan")
        )
        total = float(equity.iloc[-1] - 1) if len(equity) else float("nan")
        metrics = {
            "net_return": total,
            "sharpe_ratio": sharpe,
            "max_drawdown": float(drawdown.min()),
            "turnover": float(executable["turnover"].sum()),
        }
        for name, value in metrics.items():
            performance_rows.append(
                {
                    "selected_model_id": executable["selected_model_id"].iloc[0]
                    if len(executable)
                    else None,
                    "task": "trade_action_classification",
                    "horizon": 1,
                    "backtest_type": "primary_tradable",
                    "backtest_mode": "daily_rebalanced_next_period",
                    "cost_scenario_bps": cost,
                    "n_signals": len(signals),
                    "n_executable_trades": len(executable),
                    "n_non_executable_trades": int((~signals["is_executable"]).sum()),
                    "n_performance_observations": len(executable),
                    "metric_name": name,
                    "metric_value": value,
                    "metric_valid": np.isfinite(value),
                    "used_for_model_selection": False,
                    "used_for_strategy_optimization": False,
                }
            )
        robustness_rows.append(
            {
                "cost_bps": cost,
                "net_return": total,
                "sharpe_ratio": sharpe,
                "max_drawdown": float(drawdown.min()),
                "turnover": float(executable["turnover"].sum()),
            }
        )
    perf_path = paths.data / "backtests" / "phase5" / "phase5_backtest_performance_report.csv"
    robust_path = (
        paths.data / "backtests" / "phase5" / "phase5_transaction_cost_robustness_report.csv"
    )
    write_csv(perf_path, pd.DataFrame(performance_rows))
    robustness = pd.DataFrame(robustness_rows)
    robustness["performance_degrades_monotonically"] = robustness[
        "net_return"
    ].is_monotonic_decreasing
    write_csv(robust_path, robustness)
    return {
        "signals": signal_path,
        "ledger": ledger_path,
        "backtest": perf_path,
        "cost_robustness": robust_path,
    }
