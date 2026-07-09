import os

MPL_CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ".matplotlib"
)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.LSTM.LSTM_config import (
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_RESTORE_BEST_MODEL,
    get_lstm_results_dir,
)
from src.LSTM.LSTM_model import (
    LSTMModel,
    evaluate_model,
    train_model,
)


MAX_DEEP_ENSEMBLE_PLOT_POINTS = 1000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_deep_ensemble_uncertainty(
    actuals,
    mean_predictions,
    std_predictions,
    forecast_step=1,
    max_plot_points=MAX_DEEP_ENSEMBLE_PLOT_POINTS,
    target_label="CO2",
    results_dir=None
):
    step_index = forecast_step - 1

    if forecast_step < 1 or forecast_step > actuals.shape[1]:
        raise ValueError(
            f"forecast_step must be between 1 and {actuals.shape[1]}"
        )

    x_values = np.arange(len(actuals))

    actual_values = actuals[:, step_index]
    mean_values = mean_predictions[:, step_index]
    std_values = std_predictions[:, step_index]

    lower_bound = mean_values - 1.96 * std_values
    upper_bound = mean_values + 1.96 * std_values

    if max_plot_points is not None and len(x_values) > max_plot_points:
        print(
            f"Deep Ensemble plot limited to first {max_plot_points} "
            f"of {len(x_values)} test samples."
        )
        x_values = x_values[:max_plot_points]
        actual_values = actual_values[:max_plot_points]
        mean_values = mean_values[:max_plot_points]
        lower_bound = lower_bound[:max_plot_points]
        upper_bound = upper_bound[:max_plot_points]

    interval_width = upper_bound - lower_bound
    coverage = np.mean(
        (actual_values >= lower_bound) & (actual_values <= upper_bound)
    )

    print(f"\nDeep Ensemble bounds for forecast step {forecast_step}:")
    print("Lower bound shape:", lower_bound.shape)
    print("Upper bound shape:", upper_bound.shape)
    print("Lower bound first 5:", lower_bound[:5])
    print("Upper bound first 5:", upper_bound[:5])
    print(
        "Interval width mean/min/max:",
        np.mean(interval_width),
        np.min(interval_width),
        np.max(interval_width)
    )
    print("Empirical coverage:", coverage)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(x_values, actual_values, label="Actual")
    ax.plot(x_values, mean_values, label="Ensemble Mean Prediction")

    ax.fill_between(
        x_values,
        lower_bound,
        upper_bound,
        alpha=0.3,
        label="95% Ensemble Uncertainty Band"
    )

    ax.set_xlabel("Test sample index")
    ax.set_ylabel(f"Scaled {target_label}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    ax.set_title(
        f"Deep Ensemble Uncertainty for LSTM "
        f"(Forecast Step {forecast_step})"
    )

    ax.legend()
    fig.tight_layout()

    if results_dir is None:
        results_dir = get_lstm_results_dir()

    os.makedirs(os.path.join(results_dir, "deep_ensemble"), exist_ok=True)

    save_path = os.path.join(
        results_dir,
        "deep_ensemble",
        f"deep_ensemble_uncertainty_step_{forecast_step}.png"
    )
    plt.savefig(save_path, dpi=300)
    #print("Saved plot:", save_path)

    #plt.show()
    plt.close()


def run_deep_ensemble_uq(
    train_loader,
    val_loader,
    test_loader,
    input_size,
    actuals,
    output_seq_length,
    target_label="CO2",
    epochs=10,
    n_models=3,
    seeds=None,
    results_dir=None,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    num_layers=DEFAULT_NUM_LAYERS,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    restore_best_model=DEFAULT_RESTORE_BEST_MODEL
):
    if seeds is None:
        seeds = [11, 22, 33]

    if results_dir is None:
        results_dir = get_lstm_results_dir()

    ensemble_predictions = []

    print("\n========== DEEP ENSEMBLE UQ ==========")
    print("Number of ensemble models:", n_models)
    print("Saving Deep Ensemble plots to:", results_dir)
    print("Hidden size:", hidden_size)
    print("Number of LSTM layers:", num_layers)
    print("Dropout rate:", dropout_rate)
    print("Learning rate:", learning_rate)
    print("Weight decay:", weight_decay)
    print("Restore best validation checkpoint:", restore_best_model)

    for i in range(n_models):
        seed = seeds[i]
        set_seed(seed)

        print(f"\nTraining Ensemble Model {i + 1}/{n_models}")
        print("Seed:", seed)

        model = LSTMModel(
            input_size=input_size,
            output_seq_length=output_seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate
        )

        model, train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            restore_best_model=restore_best_model
        )

        predictions, _, mse, mae, rmse, r2 = evaluate_model(
            model,
            test_loader
        )

        print(
            f"Model {i + 1} Results -> "
            f"MSE: {mse:.6f}, MAE: {mae:.6f}, "
            f"RMSE: {rmse:.6f}, R2: {r2:.6f}"
        )

        ensemble_predictions.append(predictions)

    ensemble_predictions = np.array(ensemble_predictions)

    mean_predictions = ensemble_predictions.mean(axis=0)
    std_predictions = ensemble_predictions.std(axis=0)

    avg_uncertainty = np.mean(std_predictions)
    max_uncertainty = np.max(std_predictions)

    print("\n========== DEEP ENSEMBLE RESULTS ==========")
    print("Ensemble predictions shape:", ensemble_predictions.shape)
    print("Mean prediction shape:", mean_predictions.shape)
    print("Std prediction shape:", std_predictions.shape)

    print("\n========== ENSEMBLE UNCERTAINTY STATISTICS ==========")
    print(f"Average uncertainty: {avg_uncertainty:.6f}")
    print(f"Maximum uncertainty: {max_uncertainty:.6f}")

    plot_deep_ensemble_uncertainty(
        actuals,
        mean_predictions,
        std_predictions,
        forecast_step=1,
        max_plot_points=MAX_DEEP_ENSEMBLE_PLOT_POINTS,
        target_label=target_label,
        results_dir=results_dir
    )

    if output_seq_length > 1:
        plot_deep_ensemble_uncertainty(
            actuals,
            mean_predictions,
            std_predictions,
            forecast_step=output_seq_length,
            max_plot_points=MAX_DEEP_ENSEMBLE_PLOT_POINTS,
            target_label=target_label,
            results_dir=results_dir
        )

    return {
        "ensemble_predictions": ensemble_predictions,
        "mean_predictions": mean_predictions,
        "std_predictions": std_predictions,
        "average_uncertainty": avg_uncertainty,
        "maximum_uncertainty": max_uncertainty
    }
