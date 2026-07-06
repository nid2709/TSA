import os
import time


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MPL_CONFIG_DIR = os.path.join(BASE_DIR, ".matplotlib")
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

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


RUN_EXPLAINABILITY = False
RUN_MC_DROPOUT = False
RUN_DEEP_ENSEMBLE = False
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

    if SAVE_EDA_PLOTS:
        print("\n========== SAVING EDA PLOTS ==========")
        plot_time_series(
            df,
            results_dir=cnn_results["results_dir"],
            target_column=cnn_results["target_column"],
            target_label=cnn_results["target_label"]
        )
        plot_heatmap(df, results_dir=cnn_results["results_dir"])
        plot_pca_analysis(df, results_dir=cnn_results["results_dir"])
    else:
        print(
            "\nEDA plots skipped. Set SAVE_EDA_PLOTS=True to save all EDA plots."
        )

    if RUN_EXPLAINABILITY:
        from src.CNN_LSTM.CNN_LSTM_explainability import run_shap_experiment

        explainability_start_time = time.perf_counter()
        run_shap_experiment(cnn_results)
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
        from src.CNN_LSTM.CNN_LSTM_UQ import run_mc_dropout_uq

        mc_start_time = time.perf_counter()
        run_mc_dropout_uq(cnn_results)
        mc_elapsed_time = time.perf_counter() - mc_start_time

        print("\n========== MC DROPOUT FINISHED ==========")
        print("MC Dropout runtime:", format_elapsed_time(mc_elapsed_time))
    else:
        print("MC Dropout skipped. Set RUN_MC_DROPOUT=True to run it.")

    if RUN_DEEP_ENSEMBLE:
        from src.CNN_LSTM.CNN_LSTM_DeepEnsemble import run_deep_ensemble_uq

        ensemble_start_time = time.perf_counter()
        run_deep_ensemble_uq(
            cnn_results["train_loader"],
            cnn_results["val_loader"],
            cnn_results["test_loader"],
            cnn_results["input_size"],
            cnn_results["actuals"],
            cnn_results["output_seq_length"],
            target_label=cnn_results["target_label"],
            results_dir=cnn_results["results_dir"],
            hidden_size=cnn_results["hidden_size"],
            num_layers=cnn_results["num_layers"],
            dropout_rate=cnn_results["dropout_rate"],
            learning_rate=cnn_results["learning_rate"],
            weight_decay=cnn_results["weight_decay"],
            restore_best_model=cnn_results["restore_best_model"],
            use_attention=cnn_results["use_attention"]
        )
        ensemble_elapsed_time = time.perf_counter() - ensemble_start_time

        print("\n========== DEEP ENSEMBLE FINISHED ==========")
        print("Deep Ensemble runtime:", format_elapsed_time(ensemble_elapsed_time))
    else:
        print("Deep Ensemble skipped. Set RUN_DEEP_ENSEMBLE=True to run it.")

    pipeline_elapsed_time = time.perf_counter() - pipeline_start_time

    print("\n========== PIPELINE FINISHED ==========")
    print("Total pipeline runtime:", format_elapsed_time(pipeline_elapsed_time))


if __name__ == "__main__":
    run_pipeline()
