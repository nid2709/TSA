import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PCA_FEATURES = [
    'ens160_aqi',
    'ens160_tvoc',
    'bme688_gas_resistance',
    'bme688_pressure',
    'scd41_temperature',
    'scd41_humidity',
]


def load_prepare_data(csv_path="data/indoorAir.csv"):
    # Loads the raw indoor air CSV and prepares timestamp ordering for time series use.
    df = pd.read_csv(csv_path)

    print("\n========== Data Loading ==========")
    print(df.shape)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df = df.sort_values(by='timestamp')
    df.set_index('timestamp', inplace=True)

    return df


def get_eda_plots_dir(results_dir):
    # Creates and returns the folder used for EDA plot outputs.
    eda_dir = os.path.join(results_dir, "eda_plots")
    os.makedirs(eda_dir, exist_ok=True)
    return eda_dir


def plot_time_series(
    df,
    results_dir=None,
    max_plot_points=10000,
    target_column='scd41_co2',
    target_label='CO2'
):
    # Plots the selected target variable over time for EDA review.
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    plot_df = df
    if max_plot_points is not None and len(df) > max_plot_points:
        step = max(1, len(df) // max_plot_points)
        plot_df = df.iloc[::step]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_df.index, plot_df[target_column])
    ax.set_title(f"{target_label} Time Series")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(target_label)
    fig.tight_layout()

    if results_dir is not None:
        save_path = os.path.join(
            get_eda_plots_dir(results_dir),
            f"{target_column}_time_series.png"
        )
        fig.savefig(save_path, dpi=300)
        print(f"Saved {target_label} time series plot:", save_path)

    # plt.show()
    plt.close(fig)


def plot_heatmap(df, results_dir=None):
    # Plots numeric feature correlations to inspect relationships between sensors.
    corr = df.select_dtypes(include=['float64', 'int64']).corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    if results_dir is not None:
        save_path = os.path.join(
            get_eda_plots_dir(results_dir),
            "correlation_heatmap.png"
        )
        fig.savefig(save_path, dpi=300)
        print("Saved correlation heatmap:", save_path)

    # plt.show()
    plt.close(fig)


def plot_pca_analysis(df, results_dir=None):
    # Runs PCA on selected sensor features and plots cumulative explained variance.
    numerical_df = df[PCA_FEATURES].copy()

    numerical_df = (
        numerical_df
        .interpolate(method='time')
        .ffill()
        .bfill()
        .dropna()
    )

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numerical_df)

    pca = PCA()
    pca.fit(scaled_data)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()

    print("\n========== PCA ANALYSIS ==========")
    print("Explained Variance Ratio:")
    print(explained_variance)

    print("\nCumulative Variance:")
    print(cumulative_variance)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o'
    )

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Cumulative Explained Variance")
    ax.grid(True)
    fig.tight_layout()

    if results_dir is not None:
        save_path = os.path.join(
            get_eda_plots_dir(results_dir),
            "pca_cumulative_explained_variance.png"
        )
        fig.savefig(save_path, dpi=300)
        print("Saved PCA cumulative explained variance plot:", save_path)

    # plt.show()
    plt.close(fig)
