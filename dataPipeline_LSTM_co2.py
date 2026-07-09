import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

from src.LSTM.dataLoad import (
    load_prepare_data,
    plot_time_series,
    plot_heatmap,
    plot_pca_analysis
)
from src.LSTM.LSTM_co2 import (
    run_lstm_model,
    DEFAULT_OUTPUT_SEQ_LENGTH,
    DEFAULT_MAX_FILL_STEPS
)


RUN_EXPLAINABILITY = True
RUN_MC_DROPOUT = True
RUN_DEEP_ENSEMBLE = True
SAVE_EDA_PLOTS = False


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

    # Train LSTM and keep returned values needed for Explainability techniques.
    # LSTM_explainability.py uses model, X_train, X_test, actuals, and features.
    output_seq_length = DEFAULT_OUTPUT_SEQ_LENGTH
    lstm_start_time = time.perf_counter()
    lstm_results = run_lstm_model(
        df,
        output_seq_length=output_seq_length,
        max_fill_steps=DEFAULT_MAX_FILL_STEPS
    )
    lstm_elapsed_time = time.perf_counter() - lstm_start_time

    print(lstm_results.keys()) # to check SHAP / Explainability
    print("\n========== DATA GAP HANDLING ==========")
    print("Maximum feature fill steps:", lstm_results["max_fill_steps"])
    print("Gap-aware sequence generation:", lstm_results["gap_aware_sequences"])
    print("Drop short stations:", lstm_results["drop_short_stations"])
    print("Clip outliers:", lstm_results["clip_outliers"])
    print("Outlier clip factor:", lstm_results["outlier_clip_factor"])
    print("Restore best validation checkpoint:", lstm_results["restore_best_model"])
    print("Use station one-hot encoding:", lstm_results["use_station_one_hot"])

    print("\n========== LSTM VS LSTM + SCATTERING TAGS ==========")
    print("Use scattering:", lstm_results["use_scattering"])
    print("Scattering J:", lstm_results["scattering_j"])
    print("Scattering Q:", lstm_results["scattering_q"])
    print("Scattering features:", lstm_results["n_scattering_features"])
    print("Final input feature count:", lstm_results["input_size"])
    print("Scattering plot:", lstm_results["scattering_plot_path"])
    print("Per-horizon metrics:", lstm_results["horizon_metrics_path"])
    print("Future target reference:", lstm_results["future_target_reference_path"])
    print("Results directory:", lstm_results["results_dir"])
    print("LSTM runtime:", lstm_results["training_runtime_formatted"])

    if SAVE_EDA_PLOTS:
        print("\n========== SAVING EDA PLOTS ==========")
        plot_time_series(
            df,
            results_dir=lstm_results["results_dir"],
            target_column=lstm_results["target_column"],
            target_label=lstm_results["target_label"]
        )
        plot_heatmap(df, results_dir=lstm_results["results_dir"])
        plot_pca_analysis(df, results_dir=lstm_results["results_dir"])
    else:
        print(
            "\nEDA plots skipped. Set SAVE_EDA_PLOTS=True to save all EDA plots."
        )

    explainability_results = None
    uq_results = None
    deep_ensemble_results = None
    explainability_elapsed_time = 0
    uq_elapsed_time = 0
    deep_ensemble_elapsed_time = 0

    if RUN_EXPLAINABILITY:
        from src.LSTM.LSTM_explainability import run_shap_experiment

        explainability_start_time = time.perf_counter()
        explainability_results = run_shap_experiment(lstm_results)
        explainability_elapsed_time = (
            time.perf_counter() - explainability_start_time
        )

        print("\n========== EXPLAINABILITY FINISHED ==========")
        print(
            "Explainability runtime:",
            format_elapsed_time(explainability_elapsed_time)
        )
    else:
        print("\nExplainability skipped. Set RUN_EXPLAINABILITY=True to run SHAP/PFI/IG.")

    if RUN_MC_DROPOUT:
        from src.LSTM.LSTM_UQ import run_mc_dropout_uq

        print("\n========== RUNNING MONTE CARLO DROPOUT ==========")
        uq_start_time = time.perf_counter()
        uq_results = run_mc_dropout_uq(
            lstm_results,
            n_samples=100,
            batch_size=256
        )
        uq_elapsed_time = time.perf_counter() - uq_start_time

        print("\n========== MONTE CARLO DROPOUT FINISHED ==========")
        print("MC Dropout runtime:", format_elapsed_time(uq_elapsed_time))
    else:
        print("MC Dropout skipped. Set RUN_MC_DROPOUT=True to run it.")

    if RUN_DEEP_ENSEMBLE:
        from src.LSTM.LSTM_DeepEnsemble import run_deep_ensemble_uq

        print("\n========== RUNNING DEEP ENSEMBLE ==========")
        deep_ensemble_start_time = time.perf_counter()
        deep_ensemble_results = run_deep_ensemble_uq(
            train_loader=lstm_results["train_loader"],
            val_loader=lstm_results["val_loader"],
            test_loader=lstm_results["test_loader"],
            input_size=lstm_results["input_size"],
            actuals=lstm_results["actuals"],
            output_seq_length=lstm_results["output_seq_length"],
            target_label=lstm_results["target_label"],
            epochs=10,
            n_models=3,
            results_dir=lstm_results["results_dir"],
            hidden_size=lstm_results["hidden_size"],
            num_layers=lstm_results["num_layers"],
            dropout_rate=lstm_results["dropout_rate"],
            learning_rate=lstm_results["learning_rate"],
            weight_decay=lstm_results["weight_decay"],
            restore_best_model=lstm_results["restore_best_model"]
        )
        deep_ensemble_elapsed_time = (
            time.perf_counter() - deep_ensemble_start_time
        )
    else:
        print("Deep Ensemble skipped. Set RUN_DEEP_ENSEMBLE=True to run it.")

    pipeline_elapsed_time = time.perf_counter() - pipeline_start_time

    print("\n========== PIPELINE RUNTIME SUMMARY ==========")
    print("LSTM runtime:", format_elapsed_time(lstm_elapsed_time))
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
        lstm_results,
        explainability_results,
        uq_results,
        deep_ensemble_results
    )


if __name__ == "__main__":
    run_pipeline()
