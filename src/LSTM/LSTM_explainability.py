import os
import copy

MPL_CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ".matplotlib"
)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from captum.attr import IntegratedGradients

from src.LSTM.dataLoad import load_prepare_data
from src.LSTM.LSTM_config import (
    get_lstm_results_dir,
    DEFAULT_OUTPUT_SEQ_LENGTH
)
from src.LSTM.LSTM_co2 import run_lstm_model

# This helper is for SHAP Explainability because different SHAP versions can
# return multi-output values as either samples-first or outputs-first arrays.
def calculate_mean_abs_shap(shap_values, feature_names):
    # Reduces SHAP outputs into one mean absolute importance value per feature.

    shap_array = np.array(shap_values)

    if shap_array.ndim == 4:
        if shap_array.shape[2] == len(feature_names):
            return np.abs(shap_array).mean(axis=(0, 1, 3))

        if shap_array.shape[3] == len(feature_names):
            return np.abs(shap_array).mean(axis=(0, 1, 2))

    if shap_array.ndim == 3 and shap_array.shape[2] == len(feature_names):
        return np.abs(shap_array).mean(axis=(0, 1))

    raise ValueError(
        "Unexpected SHAP values shape for LSTM: "
        f"{shap_array.shape}"
    )

#============================== START:SHAP FOR LSTM =============================
def run_shap_experiment(results=None):
    # Runs SHAP, PFI, and Integrated Gradients explanations for the LSTM results.

    if results is None:

        df = load_prepare_data("data/indoorAir2.csv")

        results = run_lstm_model(
            df,
            output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
            show_prediction_plot=False
        )

    model = results["model"]

    X_train = results["X_train"]
    X_test = results["X_test"]

    feature_names = results["model_features"]
    results_dir = results.get("results_dir", get_lstm_results_dir())

    output_seq_length = results["output_seq_length"]
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    print("Features:")
    print(feature_names)

    background = torch.tensor(
        X_train[:200],
        dtype=torch.float32
    )

    test_samples = torch.tensor(
        X_test[:100],
        dtype=torch.float32
    )

    print("\nBackground shape:", background.shape)
    print("Test samples shape:", test_samples.shape)

    # SHAP GradientExplainer is most stable on CPU. The original trained model
    # can stay on MPS/CUDA for PFI and Integrated Gradients below.
    shap_model = copy.deepcopy(model).cpu()
    shap_model.eval()

    print("\nCreating SHAP GradientExplainer for LSTM...")

    explainer = shap.GradientExplainer(
        shap_model,
        background
    )

    print("Calculating SHAP values for LSTM...")

    shap_values = explainer.shap_values(test_samples)

    print("\nSHAP calculation successful!")

    print(type(shap_values))

    print("\nSHAP Values Shape:")
    print(np.array(shap_values).shape)

    mean_abs_shap = calculate_mean_abs_shap(
        shap_values,
        feature_names
    )

    print("\n========== GLOBAL FEATURE IMPORTANCE ==========")

    for feature, importance in sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature:25s} {importance:.6f}")

    # ==============================
    # SHAP GLOBAL FEATURE IMPORTANCE PLOT for LSTM
    # ==============================

    sorted_items = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_features = [item[0] for item in sorted_items]
    sorted_importance = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("SHAP Global Feature Importance for LSTM")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(os.path.join(results_dir, "shap"), exist_ok=True)
    save_path = os.path.join(
        results_dir,
        "shap",
        "shap_global_feature_importance_lstm.png"
    )
    plt.savefig(save_path, dpi=300)
    #print("Saved plot:", save_path)

    #plt.show()
    plt.close()

    del shap_values, mean_abs_shap, explainer, background, test_samples
    del shap_model

    # for PFI
    actuals = results["actuals"]

    pfi_results = run_pfi_analysis(
        model=model,
        # X_test=X_test[:2000],
        # actuals=actuals[:2000],
        X_test=X_test,
        actuals=actuals,
        feature_names=feature_names,
        results_dir=results_dir,
        max_samples=2000,
        batch_size=256
    )

    # for IG - compare first forecast step with final output horizon only
    forecast_horizons = sorted(set([1, output_seq_length]))

    print(
        "\nRunning Integrated Gradients for selected forecast horizons:"
        f" {forecast_horizons}"
    )

    for horizon in forecast_horizons:

        print(f"\n========== IG FOR FORECAST HORIZON {horizon} ==========")

        run_integrated_gradients_analysis(
            model=model,
            X_test=X_test,
            feature_names=feature_names,
            forecast_step=horizon,
            num_samples=32,
            results_dir=results_dir
        )

        print(f"Integrated Gradients finished for forecast horizon {horizon}")

    print("\n========== LSTM EXPLAINABILITY FINISHED ==========")

    return pfi_results
#============================== END:SHAP FOR LSTM =============================

#============================== START:PFI FOR LSTM =============================
def run_pfi_analysis(
    model,
    X_test,
    actuals,
    feature_names,
    results_dir=None,
    max_samples=2000,
    batch_size=256
):
    # Measures feature importance by permuting one feature and tracking RMSE change.

    print("\n========== PERMUTATION FEATURE IMPORTANCE ==========")

    model.eval()

    if max_samples is not None and len(X_test) > max_samples:
        X_test = X_test[:max_samples]
        actuals = actuals[:max_samples]

    print("PFI samples used:", len(X_test))
    print("PFI batch size:", batch_size)

    baseline_predictions = predict_in_batches(
        model,
        X_test,
        batch_size=batch_size
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            actuals.flatten(),
            baseline_predictions.flatten()
        )
    )

    print("Baseline RMSE:", baseline_rmse)

    pfi_results = []

    for feature_index, feature_name in enumerate(feature_names):

        X_permuted = X_test.copy()

        # Shuffle this feature across samples for all timesteps
        shuffled_values = X_permuted[:, :, feature_index].copy()
        np.random.shuffle(shuffled_values)
        X_permuted[:, :, feature_index] = shuffled_values

        permuted_predictions = predict_in_batches(
            model,
            X_permuted,
            batch_size=batch_size
        )

        permuted_rmse = np.sqrt(
            mean_squared_error(
                actuals.flatten(),
                permuted_predictions.flatten()
            )
        )

        rmse_increase = permuted_rmse - baseline_rmse

        pfi_results.append(
            (feature_name, rmse_increase)
        )

        print(
            f"{feature_name:25s} "
            f"RMSE Increase: {rmse_increase:.6f}"
        )

    pfi_results = sorted(
        pfi_results,
        key=lambda x: x[1],
        reverse=True
    )

    # Plot PFI feature importance for LSTM
    sorted_features = [item[0] for item in pfi_results]
    sorted_importance = [item[1] for item in pfi_results]
    colors = [
        "tab:red" if importance < 0 else "tab:blue"
        for importance in sorted_importance
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(sorted_features, sorted_importance, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("RMSE Increase After Permutation")
    ax.set_ylabel("Feature")
    ax.set_title("Permutation Feature Importance for LSTM")
    ax.invert_yaxis()

    min_importance = min(sorted_importance)
    max_importance = max(sorted_importance)
    x_min = min(0, min_importance)
    x_max = max(0, max_importance)
    x_padding = max((x_max - x_min) * 0.05, 1e-6)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    negative_features = [
        (feature, importance)
        for feature, importance in pfi_results
        if importance < 0
    ]

    if negative_features:
        print("\nNegative PFI values found:")
        for feature, importance in negative_features:
            print(f"{feature:25s} {importance:.8f}")
    else:
        print("\nNo negative PFI values found in this run.")

    plt.tight_layout()

    if results_dir is None:
        results_dir = get_lstm_results_dir()

    os.makedirs(os.path.join(results_dir, "pfi"), exist_ok=True)
    save_path = os.path.join(
        results_dir,
        "pfi",
        "pfi_feature_importance_lstm.png"
    )
    plt.savefig(save_path, dpi=300)
    # print("Saved plot:", save_path)

    #plt.show()
    plt.close()

    return pfi_results
#============================== END:PFI FOR LSTM ===============================


def predict_in_batches(model, X_values, batch_size=256):
    # Runs model prediction in smaller batches to avoid memory issues.
    predictions = []
    device = next(model.parameters()).device

    with torch.no_grad():
        for start_index in range(0, len(X_values), batch_size):
            end_index = start_index + batch_size
            X_batch = torch.tensor(
                X_values[start_index:end_index],
                dtype=torch.float32
            ).to(device)
            batch_predictions = model(X_batch).cpu().numpy()
            predictions.append(batch_predictions)

    return np.concatenate(predictions, axis=0)

#============================== START:INTEGRATED GRADIENTS FOR LSTM =============================
def run_integrated_gradients_analysis(
    model,
    X_test,
    feature_names,
    forecast_step=1,
    num_samples=32,
    n_steps=20,
    internal_batch_size=None,
    max_plot_points=None,
    results_dir=None
):
    # Computes Integrated Gradients feature and timestep importance for one horizon.

    print("\n========== INTEGRATED GRADIENTS ==========")

    model.eval()
    device = next(model.parameters()).device

    step_index = forecast_step - 1
    num_samples = min(num_samples, len(X_test))
    if internal_batch_size is None:
        internal_batch_size = num_samples

    class ForecastStepWrapper(torch.nn.Module):
        def __init__(self, model, step_index):
            super().__init__()
            self.model = model
            self.step_index = step_index

        def forward(self, x):
            output = self.model(x)
            return output[:, self.step_index]

    wrapped_model = ForecastStepWrapper(
        model,
        step_index
    ).to(device)

    integrated_gradients = IntegratedGradients(
        wrapped_model
    )

    input_tensor = torch.tensor(
        X_test[:num_samples],
        dtype=torch.float32
    ).to(device)

    baseline = torch.zeros_like(input_tensor)

    print("Input tensor shape:", input_tensor.shape)
    print("Baseline shape:", baseline.shape)
    print(f"Explaining forecast step: {forecast_step}")
    print("IG steps:", n_steps)
    print("IG internal batch size:", internal_batch_size)

    attributions = integrated_gradients.attribute(
        input_tensor,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size
    )

    attributions = attributions.detach().cpu().numpy()

    print("Attributions shape:", attributions.shape)

    # Mean absolute attribution per feature for LSTM
    feature_importance = np.abs(attributions).mean(axis=(0, 1))

    print("\n========== IG FEATURE IMPORTANCE ==========")

    for feature, importance in sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{feature:25s} {importance:.6f}")

    # Plot IG feature importance for LSTM
    sorted_items = sorted(
        zip(feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_features = [item[0] for item in sorted_items]
    sorted_importance = [item[1] for item in sorted_items]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance)
    plt.xlabel("Mean Absolute Integrated Gradients Attribution")
    plt.ylabel("Feature")
    plt.title(
        f"Integrated Gradients Feature Importance for LSTM "
        f"(Forecast Step {forecast_step})"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if results_dir is None:
        results_dir = get_lstm_results_dir()

    os.makedirs(
        os.path.join(results_dir, "integrated_gradients"),
        exist_ok=True
    )

    save_path = os.path.join(
        results_dir,
        "integrated_gradients",
        f"ig_feature_importance_lstm_step_{forecast_step}.png"
    )
    plt.savefig(save_path, dpi=300)
    #print("Saved plot:", save_path)

    #plt.show()
    plt.close()

    # Mean absolute attribution per timestep for LSTM
    timestep_importance = np.abs(attributions).mean(axis=(0, 2))
    timestep_importance_plot = timestep_importance[:max_plot_points]

    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(timestep_importance_plot) + 1),
        timestep_importance_plot,
        marker="o"
    )
    plt.xlabel("Input Timestep")
    plt.ylabel("Mean Absolute Attribution")
    plt.title(
        f"Integrated Gradients Temporal Importance for LSTM "
        f"(Forecast Step {forecast_step})"
    )
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(
        results_dir,
        "integrated_gradients",
        f"ig_temporal_importance_lstm_step_{forecast_step}.png"
    )
    plt.savefig(save_path, dpi=300)
    #print("Saved plot:", save_path)

    #plt.show()
    plt.close()

    return attributions, feature_importance, timestep_importance
#============================== END:INTEGRATED GRADIENTS FOR LSTM ===============================

if __name__ == "__main__":
    run_shap_experiment()
