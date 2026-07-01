import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.CNN_LSTM.dataLoad import (
    load_prepare_data,
    plot_time_series,
    plot_heatmap,
    plot_pca_analysis
)

from src.CNN_LSTM.CNN_LSTM_co2 import (
    run_cnn_lstm_model,
    DEFAULT_OUTPUT_SEQ_LENGTH,
    DEFAULT_MAX_FILL_STEPS
)
from src.CNN_LSTM.CNN_LSTM_UQ import run_mc_dropout_uq
from src.CNN_LSTM.CNN_LSTM_DeepEnsemble import run_deep_ensemble_uq
from src.CNN_LSTM.CNN_LSTM_explainability import run_shap_experiment


def format_elapsed_time(seconds):
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"

    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    return f"{remaining_seconds:.2f}s"


def run_pipeline():
    pipeline_start_time = time.perf_counter()
    print("Libraries imported..!")

    csv_path = os.path.join(BASE_DIR, "data", "indoorAir2.csv")

    df = load_prepare_data(csv_path)

    # Optional analysis plots
    # plot_time_series(df)
    # plot_heatmap(df)
    # plot_pca_analysis(df)

    # Train CNN-LSTM and keep returned values needed for Explainability techniques.
    # CNN_LSTM_explainability.py uses model, X_train, X_test, actuals, and features.
    output_seq_length = DEFAULT_OUTPUT_SEQ_LENGTH
    cnn_start_time = time.perf_counter()
    cnn_results = run_cnn_lstm_model(
        df,
        output_seq_length=output_seq_length,
        max_fill_steps=DEFAULT_MAX_FILL_STEPS
    )
    cnn_elapsed_time = time.perf_counter() - cnn_start_time

    print(cnn_results.keys()) # to check SHAP / Explainability
    print("\n========== DATA GAP HANDLING ==========")
    print("Maximum feature fill steps:", cnn_results["max_fill_steps"])
    print("Gap-aware sequence generation:", cnn_results["gap_aware_sequences"])
    print("Drop short stations:", cnn_results["drop_short_stations"])
    print("Clip outliers:", cnn_results["clip_outliers"])
    print("Outlier clip factor:", cnn_results["outlier_clip_factor"])
    print("Restore best validation checkpoint:", cnn_results["restore_best_model"])

    print("\n========== CNN-LSTM VS CNN-LSTM + SCATTERING TAGS ==========")
    print("Use scattering:", cnn_results["use_scattering"])
    print("Scattering J:", cnn_results["scattering_j"])
    print("Scattering Q:", cnn_results["scattering_q"])
    print("Scattering features:", cnn_results["n_scattering_features"])
    print("Final input feature count:", cnn_results["input_size"])
    print("Scattering plot:", cnn_results["scattering_plot_path"])
    print("Per-horizon metrics:", cnn_results["horizon_metrics_path"])
    print("Future target reference:", cnn_results["future_target_reference_path"])
    print("Results directory:", cnn_results["results_dir"])
    print("CNN-LSTM runtime:", cnn_results["training_runtime_formatted"])

    # Explainability techniques - saves SHAP, PFI and Integrated Gradients images
    # using the already trained CNN-LSTM model.
    explainability_start_time = time.perf_counter()
    explainability_results = run_shap_experiment(
        cnn_results
    )
    explainability_elapsed_time = (
        time.perf_counter() - explainability_start_time
    )

    print("\n========== EXPLAINABILITY FINISHED ==========")
    print(
        "Explainability runtime:",
        format_elapsed_time(explainability_elapsed_time)
    )

    # Uncertainty Quantifiers
    print("\n========== RUNNING MONTE CARLO DROPOUT ==========")
    uq_start_time = time.perf_counter()
    uq_results = run_mc_dropout_uq(
        cnn_results,
        n_samples=100,
        batch_size=256
    )
    uq_elapsed_time = time.perf_counter() - uq_start_time

    print("\n========== MONTE CARLO DROPOUT FINISHED ==========")
    print("MC Dropout runtime:", format_elapsed_time(uq_elapsed_time))
    print("\n========== RUNNING DEEP ENSEMBLE ==========")

    deep_ensemble_start_time = time.perf_counter()
    deep_ensemble_results = run_deep_ensemble_uq(
        train_loader=cnn_results["train_loader"],
        val_loader=cnn_results["val_loader"],
        test_loader=cnn_results["test_loader"],
        input_size=cnn_results["input_size"],
        actuals=cnn_results["actuals"],
        output_seq_length=cnn_results["output_seq_length"],
        target_label=cnn_results["target_label"],
        epochs=10,
        n_models=3,
        results_dir=cnn_results["results_dir"],
        hidden_size=cnn_results["hidden_size"],
        num_layers=cnn_results["num_layers"],
        dropout_rate=cnn_results["dropout_rate"],
        learning_rate=cnn_results["learning_rate"],
        weight_decay=cnn_results["weight_decay"],
        restore_best_model=cnn_results["restore_best_model"]
    )
    deep_ensemble_elapsed_time = (
        time.perf_counter() - deep_ensemble_start_time
    )

    pipeline_elapsed_time = time.perf_counter() - pipeline_start_time

    print("\n========== PIPELINE RUNTIME SUMMARY ==========")
    print("CNN-LSTM runtime:", format_elapsed_time(cnn_elapsed_time))
    print(
        "Explainability runtime:",
        format_elapsed_time(explainability_elapsed_time)
    )
    print("MC Dropout runtime:", format_elapsed_time(uq_elapsed_time))
    print(
        "Deep Ensemble runtime:",
        format_elapsed_time(deep_ensemble_elapsed_time)
    )
    print("Total pipeline runtime:", format_elapsed_time(pipeline_elapsed_time))

    return (
        cnn_results,
        explainability_results,
        uq_results,
        deep_ensemble_results
    )

    #return cnn_results


if __name__ == "__main__":
    run_pipeline()
