import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch

from src.CNN_LSTM.CNN_LSTM_config import (
    DEFAULT_N_SCATTERING_FEATURES,
    DEFAULT_SCATTERING_J,
    DEFAULT_SCATTERING_Q,
    TARGET,
    get_cnn_lstm_results_dir,
    get_target_label,
)


def plot_loss_curves(train_losses, val_losses, results_dir=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            "train_validation_loss.png"
        )
        plt.savefig(save_path, dpi=300)
        #print("Saved plot:", save_path)

    #plt.show()
    plt.close()

def plot_scattering_wavelet_features(
    X_train,
    model_features,
    n_scattering_features,
    scattering_j,
    scattering_q,
    results_dir
):
    if n_scattering_features <= 0 or len(X_train) == 0:
        return None

    target_index = model_features.index(TARGET)
    scattering_feature_names = get_scattering_feature_names(
        n_scattering_features
    )
    scattering_indices = [
        model_features.index(feature_name)
        for feature_name in scattering_feature_names
    ]

    sample_window = X_train[0]
    co2_signal = sample_window[:, target_index]

    # Scattering features are static within a sequence, so the first timestep
    # contains the same coefficient values supplied at every timestep.
    scattering_values = sample_window[0, scattering_indices]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(
        np.arange(len(co2_signal)),
        co2_signal,
        color="tab:blue",
        linewidth=1.8
    )
    axes[0].set_title("Representative Input Window")
    axes[0].set_xlabel("Input timestep")
    axes[0].set_ylabel("Scaled CO2")
    axes[0].grid(alpha=0.25)

    feature_labels = [
        f"S{i + 1}"
        for i in range(n_scattering_features)
    ]
    axes[1].bar(
        feature_labels,
        scattering_values,
        color="tab:orange"
    )
    axes[1].set_title(
        f"Static Scattering Features Used by CNN-LSTM "
        f"(J={scattering_j}, Q={scattering_q})"
    )
    axes[1].set_xlabel("Selected scattering coefficient")
    axes[1].set_ylabel("Mean coefficient value")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Scattering Wavelet Transform Feature Example",
        fontsize=14
    )
    fig.tight_layout()

    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)
    save_path = os.path.join(
        main_plots_dir,
        "scattering_wavelet_features.png"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved scattering wavelet plot:", save_path)

    return save_path

def plot_attention_weights(model, test_loader, results_dir):
    if not getattr(model, "use_attention", False):
        return None

    model.eval()
    X_batch, _ = next(iter(test_loader))

    with torch.no_grad():
        _, attention_weights = model.forward_with_attention(X_batch[0:1])

    if attention_weights is None:
        return None

    weights_matrix = attention_weights[0].cpu().numpy()
    final_query_weights = weights_matrix[-1]

    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights_matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xlabel("Key timestep / past context")
    ax.set_ylabel("Query timestep")
    ax.set_title("CNN-LSTM Temporal Attention Heatmap")
    fig.tight_layout()

    heatmap_path = os.path.join(
        main_plots_dir,
        "attention_weights_heatmap.png"
    )
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        np.arange(len(final_query_weights)),
        final_query_weights,
        linewidth=1.8,
        color="tab:purple"
    )
    ax.set_xlabel("Input timestep")
    ax.set_ylabel("Attention weight")
    ax.set_title("Final Forecast Context Attention Over Input Window")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    final_query_path = os.path.join(
        main_plots_dir,
        "attention_final_context_weights.png"
    )
    fig.savefig(final_query_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved attention heatmap:", heatmap_path)
    print("Saved final-context attention plot:", final_query_path)

    return {
        "heatmap_path": heatmap_path,
        "final_context_path": final_query_path
    }

def plot_predictions(
    actuals,
    predictions,
    forecast_step=1,
    max_plot_points=5000,
    results_dir=None
):
    step_index = forecast_step - 1

    if forecast_step < 1 or forecast_step > actuals.shape[1]:
        raise ValueError(f"forecast_step must be between 1 and {actuals.shape[1]}")

    x_values = np.arange(len(actuals))
    actual_values = actuals[:, step_index]
    predicted_values = predictions[:, step_index]

    if max_plot_points is not None and  len(x_values) > max_plot_points:
        x_values = x_values[:max_plot_points]
        actual_values = actual_values[:max_plot_points]
        predicted_values = predicted_values[:max_plot_points]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_values, actual_values, label="Actual")
    ax.plot(x_values, predicted_values, label="Predicted")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Scaled CO2")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend()
    ax.set_title(f"Actual vs Predicted CO2 for CNN-LSTM (Forecast Step {forecast_step})")
    fig.tight_layout()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            f"actual_vs_predicted_step_{forecast_step}.png"
        )
        plt.savefig(save_path, dpi=300)
        #print("Saved plot:", save_path)

    #plt.show()
    plt.close()

def plot_actual_vs_predicted_scatter(
    actuals,
    predictions,
    max_points=5000,
    results_dir=None
):
    actual_values = actuals.flatten()
    predicted_values = predictions.flatten()

    if max_points is not None and len(actual_values) > max_points:
        actual_values = actual_values[:max_points]
        predicted_values = predicted_values[:max_points]

    min_value = min(actual_values.min(), predicted_values.min())
    max_value = max(actual_values.max(), predicted_values.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        actual_values,
        predicted_values,
        alpha=0.25,
        s=12,
        color="tab:orange"
    )
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="tab:blue",
        linewidth=1.5,
        label="Perfect prediction"
    )
    ax.set_xlabel("Actual scaled CO2")
    ax.set_ylabel("Predicted scaled CO2")
    ax.set_title("Actual vs Predicted Scatter for CNN-LSTM")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            "actual_vs_predicted_scatter.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved scatter plot:", save_path)

    plt.close(fig)

def plot_forecast_comparison(actuals, predictions, results_dir=None):
    output_seq_length = actuals.shape[1]
    
    # Dynamic Plot 1: Horizon Step 1
    plot_predictions(
        actuals,
        predictions,
        forecast_step=1,
        results_dir=results_dir
    )

    # Dynamic Plot 2: Final Horizon Output Length Step
    if output_seq_length > 1:
        plot_predictions(
            actuals,
            predictions,
            forecast_step=output_seq_length,
            results_dir=results_dir
        )

    plot_actual_vs_predicted_scatter(
        actuals,
        predictions,
        results_dir=results_dir
    )
