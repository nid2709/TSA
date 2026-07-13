from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _predict(model, X, batch_size=128):
    model.eval()
    predictions = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            X_batch = torch.tensor(X[start:start + batch_size], dtype=torch.float32)
            y_pred = model(X_batch).detach().cpu().numpy()
            predictions.append(y_pred)

    return np.vstack(predictions)


def _regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def grouped_permutation_importance(
    model,
    X_test,
    y_test,
    feature_cols,
    n_repeats=5,
    random_state=42,
    batch_size=128,
):
    print("\n========== Grouped Permutation Feature Importance ==========")

    rng = np.random.default_rng(random_state)
    baseline_pred = _predict(model, X_test, batch_size=batch_size)
    baseline_mae, baseline_rmse = _regression_metrics(y_test.reshape(-1, 1), baseline_pred)

    results = []

    for feature_idx, feature_name in enumerate(feature_cols):
        repeat_mae = []
        repeat_rmse = []

        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            permutation = rng.permutation(X_permuted.shape[0])

            # Shuffle one feature across samples, keeping all 480 lags grouped together.
            X_permuted[:, :, feature_idx] = X_permuted[permutation, :, feature_idx]

            permuted_pred = _predict(model, X_permuted, batch_size=batch_size)
            mae, rmse = _regression_metrics(y_test.reshape(-1, 1), permuted_pred)

            repeat_mae.append(mae)
            repeat_rmse.append(rmse)

        results.append(
            {
                "feature": feature_name,
                "baseline_mae": baseline_mae,
                "permuted_mae": np.mean(repeat_mae),
                "mae_increase": np.mean(repeat_mae) - baseline_mae,
                "baseline_rmse": baseline_rmse,
                "permuted_rmse": np.mean(repeat_rmse),
                "rmse_increase": np.mean(repeat_rmse) - baseline_rmse,
            }
        )

    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values("rmse_increase", ascending=False).reset_index(drop=True)

    print(importance_df[["feature", "mae_increase", "rmse_increase"]])
    return importance_df


def time_lag_occlusion_importance(
    model,
    X_train,
    X_test,
    y_test,
    steps_per_hour=4,
    batch_size=128,
):
    print("\n========== Time-Lag Occlusion Importance ==========")

    baseline_pred = _predict(model, X_test, batch_size=batch_size)
    baseline_mae, baseline_rmse = _regression_metrics(y_test.reshape(-1, 1), baseline_pred)

    n_steps = X_test.shape[1]
    feature_means = X_train.mean(axis=(0, 1))

    windows = [
        ("last_1_hour", n_steps - steps_per_hour, n_steps),
        ("last_6_hours", n_steps - 6 * steps_per_hour, n_steps),
        ("last_24_hours", n_steps - 24 * steps_per_hour, n_steps),
        ("previous_24_to_48_hours", n_steps - 48 * steps_per_hour, n_steps - 24 * steps_per_hour),
        ("oldest_history_before_48_hours", 0, max(n_steps - 48 * steps_per_hour, 0)),
    ]

    results = []

    for window_name, start, end in windows:
        start = max(start, 0)
        end = min(end, n_steps)

        if start >= end:
            continue

        X_occluded = X_test.copy()

        # Replace this time region with average training values.
        X_occluded[:, start:end, :] = feature_means

        occluded_pred = _predict(model, X_occluded, batch_size=batch_size)
        mae, rmse = _regression_metrics(y_test.reshape(-1, 1), occluded_pred)

        results.append(
            {
                "time_window": window_name,
                "start_lag_index": start,
                "end_lag_index": end,
                "baseline_mae": baseline_mae,
                "occluded_mae": mae,
                "mae_increase": mae - baseline_mae,
                "baseline_rmse": baseline_rmse,
                "occluded_rmse": rmse,
                "rmse_increase": rmse - baseline_rmse,
            }
        )

    importance_df = pd.DataFrame(results)
    importance_df = importance_df.sort_values("rmse_increase", ascending=False).reset_index(drop=True)

    print(importance_df[["time_window", "mae_increase", "rmse_increase"]])
    return importance_df


def _plot_bar(df, label_col, value_col, title, xlabel, output_path):
    plt.figure(figsize=(10, 5))
    ordered_df = df.sort_values(value_col, ascending=True)
    plt.barh(ordered_df[label_col], ordered_df[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_explainability(
    model,
    X_train,
    X_test,
    y_test,
    feature_cols,
    output_dir=None,
):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "explainability_outputs"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_importance = grouped_permutation_importance(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
    )

    time_importance = time_lag_occlusion_importance(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
    )

    feature_csv = output_dir / "grouped_permutation_feature_importance.csv"
    time_csv = output_dir / "time_lag_occlusion_importance.csv"
    feature_png = output_dir / "grouped_permutation_feature_importance.png"
    time_png = output_dir / "time_lag_occlusion_importance.png"

    feature_importance.to_csv(feature_csv, index=False)
    time_importance.to_csv(time_csv, index=False)

    _plot_bar(
        feature_importance,
        label_col="feature",
        value_col="rmse_increase",
        title="Grouped Permutation Feature Importance",
        xlabel="RMSE increase after feature permutation",
        output_path=feature_png,
    )

    _plot_bar(
        time_importance,
        label_col="time_window",
        value_col="rmse_increase",
        title="Time-Lag Occlusion Importance",
        xlabel="RMSE increase after time-window occlusion",
        output_path=time_png,
    )

    print("\nExplainability files saved:")
    print(feature_csv)
    print(time_csv)
    print(feature_png)
    print(time_png)

    return feature_importance, time_importance