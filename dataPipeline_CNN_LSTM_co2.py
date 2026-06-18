import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.CNN_LSTM.dataLoad import (
    load_prepare_data,
    plot_time_series,
    plot_heatmap,
    plot_pca_analysis
)

from src.CNN_LSTM.CNN_LSTM_co2 import (
    run_cnn_lstm_model,
    DEFAULT_OUTPUT_SEQ_LENGTH
)
from src.CNN_LSTM.CNN_LSTM_UQ import run_mc_dropout_uq
from src.CNN_LSTM.CNN_LSTM_DeepEnsemble import run_deep_ensemble_uq
from src.CNN_LSTM.CNN_LSTM_explainability import run_shap_experiment


def run_pipeline():
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
    cnn_results = run_cnn_lstm_model(
        df,
        output_seq_length=output_seq_length
    )

    print(cnn_results.keys()) # to check SHAP / Explainability

    # Explainability techniques - saves SHAP, PFI and Integrated Gradients images
    # using the already trained CNN-LSTM model.
    explainability_results = run_shap_experiment(
        cnn_results
    )

    print("\n========== EXPLAINABILITY FINISHED ==========")

    # Uncertainty Quantifiers
    print("\n========== RUNNING MONTE CARLO DROPOUT ==========")
    uq_results = run_mc_dropout_uq(
        cnn_results,
        n_samples=100
    )

    print("\n========== MONTE CARLO DROPOUT FINISHED ==========")
    print("\n========== RUNNING DEEP ENSEMBLE ==========")

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
        results_dir=cnn_results["results_dir"]
    )

    return (
        cnn_results,
        explainability_results,
        uq_results,
        deep_ensemble_results
    )


if __name__ == "__main__":
    run_pipeline()
