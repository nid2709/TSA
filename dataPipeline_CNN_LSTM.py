import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from src.CNN_LSTM.dataLoad import (
    load_prepare_data,
    plot_time_series,
    plot_heatmap,
    plot_pca_analysis
)
from src.CNN_LSTM.CNN_LSTM import run_cnn_lstm_model


def run_pipeline():
    print("Libraries imported..!")

    csv_path = os.path.join(BASE_DIR, "data", "indoorAir2.csv")
    df = load_prepare_data(csv_path)

    # Optional analysis plots
    # plot_time_series(df)
    # plot_heatmap(df)
    # plot_pca_analysis(df)

    # Change this value to automatically plot Horizon 1 and Horizon X graphs
    output_seq_length = 6 
    
    cnn_results = run_cnn_lstm_model(
        df,
        output_seq_length=output_seq_length
    )
    
    #print(cnn_results.keys())  # Matches your LSTM output logic check for SHAP

    return cnn_results


if __name__ == "__main__":
    run_pipeline()