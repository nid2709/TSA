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
    df = pd.read_csv(csv_path)

    print("\n========== Data Loading ==========")
    print(df.shape)

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    df = df.sort_values(by='timestamp')
    df.set_index('timestamp', inplace=True)

    return df


def plot_time_series(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df['scd41_co2'])
    plt.title("CO2 Time Series")
    plt.show()


def plot_heatmap(df):
    corr = df.select_dtypes(include=['float64', 'int64']).corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")
    plt.show()


def plot_pca_analysis(df):
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

    plt.figure(figsize=(8, 4))

    plt.plot(
        range(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o'
    )

    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True)
    plt.show()
