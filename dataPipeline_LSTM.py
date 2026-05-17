import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.LSTM.dataLoad import load_prepare_data, plot_time_series, plot_heatmap, plot_pca_analysis
from src.LSTM.LSTM import run_lstm_model


def run_pipeline():
    print("Libraries imported..!")

    csv_path = os.path.join(BASE_DIR, "data", "indoorAir.csv")
    df = load_prepare_data(csv_path)

    # Optional analysis plots. Keep them commented during normal training because
    # plot windows can slow or block the pipeline.
    # plot_time_series(df)
    # plot_heatmap(df)
    # plot_pca_analysis(df)

    # Train and evaluate the LSTM model.
    lstm_results = run_lstm_model(df)

    return lstm_results


if __name__ == "__main__":
    run_pipeline()
