import os

MPL_CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ".matplotlib"
)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.CNN_LSTM.CNN_LSTM_config import get_cnn_lstm_results_dir


def enable_dropout_during_inference(model):
    """
    Keep the full model in eval mode, but activate only Dropout layers.
    This is the key idea of Monte Carlo Dropout.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model,
    X_test,
    n_samples=100,
    batch_size=256
):
    """
    Run the same test data through the CNN-LSTM model many times with dropout ON.
    Returns:
        mean prediction
        standard deviation
        all predictions
    """

    mc_predictions = []

    with torch.no_grad():

        for sample_index in range(n_samples):

            enable_dropout_during_inference(model)

            sample_predictions = []

            for start_index in range(0, len(X_test), batch_size):
                end_index = start_index + batch_size
                X_batch = torch.tensor(
                    X_test[start_index:end_index],
                    dtype=torch.float32
                )
                batch_predictions = model(X_batch).cpu().numpy()
                sample_predictions.append(batch_predictions)

            predictions = np.concatenate(sample_predictions, axis=0)

            mc_predictions.append(predictions)

            if (sample_index + 1) % 10 == 0 or (sample_index + 1) == n_samples:
                print(
                    f"MC Dropout sample {sample_index + 1}/{n_samples} completed"
                )

    mc_predictions = np.array(mc_predictions)

    mean_predictions = mc_predictions.mean(axis=0)
    std_predictions = mc_predictions.std(axis=0)

    return mean_predictions, std_predictions, mc_predictions


def plot_mc_dropout_uncertainty(
    actuals,
    mean_predictions,
    std_predictions,
    forecast_step=1,
    max_plot_points=None,
    target_label="CO2",
    results_dir=None
):
    """
    Plot actual values, mean prediction, and uncertainty band.
    """

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
        x_values = x_values[:max_plot_points]
        actual_values = actual_values[:max_plot_points]
        mean_values = mean_values[:max_plot_points]
        lower_bound = lower_bound[:max_plot_points]
        upper_bound = upper_bound[:max_plot_points]

    interval_width = upper_bound - lower_bound
    coverage = np.mean(
        (actual_values >= lower_bound) & (actual_values <= upper_bound)
    )

    print(f"\nMC Dropout bounds for forecast step {forecast_step}:")
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

    ax.plot(
        x_values,
        actual_values,
        label="Actual"
    )

    ax.plot(
        x_values,
        mean_values,
        label="Mean Prediction"
    )

    ax.fill_between(
        x_values,
        lower_bound,
        upper_bound,
        alpha=0.3,
        label="95% Uncertainty Band"
    )

    ax.set_xlabel("Test sample index")
    ax.set_ylabel(f"Scaled {target_label}")

    ax.xaxis.set_major_locator(
        MaxNLocator(nbins=8, integer=True)
    )

    ax.set_title(
        f"MC Dropout Uncertainty for CNN-LSTM "
        f"(Forecast Step {forecast_step})"
    )

    ax.legend()
    fig.tight_layout()

    if results_dir is None:
        results_dir = get_cnn_lstm_results_dir()

    os.makedirs(os.path.join(results_dir, "mc_dropout"), exist_ok=True)

    save_path = os.path.join(
        results_dir,
        "mc_dropout",
        f"mc_dropout_uncertainty_step_{forecast_step}.png"
    )
    plt.savefig(save_path, dpi=300)
    #print("Saved plot:", save_path)

    #plt.show()
    plt.close()


def run_mc_dropout_uq(
    cnn_lstm_results,
    n_samples=100,
    batch_size=256
):
    """
    Main function to run MC Dropout UQ using results from run_cnn_lstm_model().
    """

    model = cnn_lstm_results["model"]
    X_test = cnn_lstm_results["X_test"]
    actuals = cnn_lstm_results["actuals"]
    target_label = cnn_lstm_results["target_label"]
    output_seq_length = cnn_lstm_results["output_seq_length"]
    results_dir = cnn_lstm_results.get(
        "results_dir",
        get_cnn_lstm_results_dir()
    )

    print("\n========== STARTING MONTE CARLO DROPOUT UQ ==========")
    print("MC samples:", n_samples)
    print("MC batch size:", batch_size)

    mean_predictions, std_predictions, mc_predictions = mc_dropout_predict(
        model,
        X_test,
        n_samples=n_samples,
        batch_size=batch_size
    )

    avg_uncertainty = np.mean(std_predictions)
    max_uncertainty = np.max(std_predictions)

    print("\n========== UNCERTAINTY STATISTICS ==========")
    print(f"Average uncertainty: {avg_uncertainty:.6f}")
    print(f"Maximum uncertainty: {max_uncertainty:.6f}")

    print("\n========== MONTE CARLO DROPOUT UQ ==========")
    print("MC samples:", n_samples)
    print("Mean prediction shape:", mean_predictions.shape)
    print("Std prediction shape:", std_predictions.shape)
    print("MC predictions shape:", mc_predictions.shape)

    plot_mc_dropout_uncertainty(
        actuals,
        mean_predictions,
        std_predictions,
        forecast_step=1,
        target_label=target_label,
        results_dir=results_dir
    )

    if output_seq_length > 1:
        plot_mc_dropout_uncertainty(
            actuals,
            mean_predictions,
            std_predictions,
            forecast_step=output_seq_length,
            target_label=target_label,
            results_dir=results_dir
        )

    return {
        "mean_predictions": mean_predictions,
        "std_predictions": std_predictions,
        "mc_predictions": mc_predictions
    }
