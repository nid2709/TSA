import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.LSTM.dataLoad import load_prepare_data, plot_time_series, plot_heatmap, plot_pca_analysis
from src.LSTM.LSTM_co2 import run_lstm_model
from src.LSTM.LSTM_UQ import run_mc_dropout_uq
from src.LSTM.LSTM_DeepEnsemble import run_deep_ensemble_uq
from src.LSTM.LSTM_explainability import run_shap_experiment


def run_pipeline():
    print("Libraries imported..!")

    csv_path = os.path.join(BASE_DIR, "data", "indoorAir2.csv")
    df = load_prepare_data(csv_path)

    # Optional analysis plots. Keep them commented during normal training because
    # plot windows can slow or block the pipeline.
    # plot_time_series(df)
    # plot_heatmap(df)
    # plot_pca_analysis(df)

    # Train and evaluate all LSTM target features. Change this value to 12 to
    # show horizon 1 and horizon 12 prediction graphs for each feature.
    output_seq_length = 12
    lstm_results = run_lstm_model(
        df,
        output_seq_length=output_seq_length
    )

    # Explainability techniques - saves SHAP, PFI and Integrated Gradients images
    # using the already trained LSTM model.
    explainability_results = run_shap_experiment(
        lstm_results
    )

    # Uncertainty Quantifiers
    uq_results = run_mc_dropout_uq(
        lstm_results,
        n_samples=100
    )

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
        results_dir=lstm_results["results_dir"]
    )

    print(lstm_results.keys()) # to check SHAP

    return (
        lstm_results,
        explainability_results,
        uq_results,
        deep_ensemble_results
    )


if __name__ == "__main__":
    run_pipeline()
