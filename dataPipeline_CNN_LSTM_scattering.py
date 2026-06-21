import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.CNN_LSTM.dataLoad import (
    load_prepare_data,
    plot_time_series,
    plot_heatmap,
    plot_pca_analysis
)

from src.CNN_LSTM.CNN_LSTM_co2_scattering import (
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
        output_seq_length=output_seq_length,
        use_scattering=True,
        scattering_j=6,
        scattering_q=8,
        n_scattering_features=8
    )

    print(cnn_results.keys())

    # For Question 3, first run only the scattering-enhanced CNN-LSTM model.
    # After you confirm the metrics, you can run explainability/UQ separately.
    return cnn_results

    #return cnn_results


if __name__ == "__main__":
    run_pipeline()
