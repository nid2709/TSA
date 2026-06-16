from pathlib import Path
import pandas as pd


def data_loading():
    print("\n========== Data Loading ==========")

    file_path = Path(__file__).resolve().parents[2] / "data" / "indoorAir2.csv"

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values(by="timestamp")

    print(df.shape)
    df.head()

    return df


# def choose_feature(df):
#     print("\n========== Choose Feature ==========")

#     feature_df = df[
#         [
#             "ens160_aqi",
#             "ens160_tvoc",
#             "bme688_gas_resistance",
#             "bme688_pressure",
#             "scd41_temperature",
#             "scd41_humidity",
#             "timestamp",
#             "scd41_co2",
#             "station_id",
#         ]
#     ].copy()

#     print(feature_df.shape)
#     feature_df.head()

#     return feature_df

def choose_feature(df):
    print("\n========== Feature Selecting ==========")

    df = df[df["station_id"] != 6].copy()

    feature_df = df[
        [
            "ens160_aqi",
            "ens160_tvoc",
            "bme688_gas_resistance",
            "bme688_pressure",
            "scd41_temperature",
            "scd41_humidity",
            "scd41_co2",
            "timestamp",
            "station_id",
        ]
    ].copy()

    feature_df["timestamp_seconds"] = (
        feature_df["timestamp"].astype("int64") // 10**9
    )

    hour = (
        feature_df["timestamp"].dt.hour
        + feature_df["timestamp"].dt.minute / 60
        + feature_df["timestamp"].dt.second / 3600
    )

    dayofweek = feature_df["timestamp"].dt.dayofweek

    feature_df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feature_df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    feature_df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    feature_df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    feature_df["is_weekend"] = dayofweek.isin([5, 6]).astype(int)

    print(feature_df.shape)
    feature_df.head()

    return feature_df


def resample_with_15_min(feature_df):
    print("\n========== Resample With 15 Min ==========")

    resample_freq = "15min"

    value_cols = [
        "ens160_aqi",
        "ens160_tvoc",
        "bme688_gas_resistance",
        "bme688_pressure",
        "scd41_temperature",
        "scd41_humidity",
        "scd41_co2",
    ]

    feature_df = (
        feature_df
        .sort_values(["station_id", "timestamp"])
        .set_index("timestamp")
        .groupby("station_id")[value_cols]
        .resample(resample_freq)
        .mean()
        .reset_index()
    )

    feature_df[value_cols] = (
        feature_df
        .groupby("station_id")[value_cols]
        .transform(lambda x: x.interpolate(method="linear").ffill().bfill())
    )

    feature_df = feature_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print(feature_df.shape)
    feature_df.head()

    return feature_df


def add_15_min_ahead_column(feature_df):
    print("\n========== Add 15 Min Ahead Column ==========")

    print(feature_df.shape)

    feature_df = feature_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    target_df = feature_df[["station_id", "timestamp", "ens160_aqi"]].copy()

    target_df = target_df.rename(
        columns={
            "timestamp": "actual_target_timestamp",
            "ens160_aqi": "target_ens160_aqi_15min",
        }
    )

    feature_df["target_timestamp"] = feature_df["timestamp"] + pd.Timedelta(minutes=15)

    feature_df = feature_df.sort_values(["target_timestamp", "station_id"]).reset_index(drop=True)
    target_df = target_df.sort_values(["actual_target_timestamp", "station_id"]).reset_index(drop=True)

    feature_df = pd.merge_asof(
        feature_df,
        target_df,
        left_on="target_timestamp",
        right_on="actual_target_timestamp",
        by="station_id",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=2),
    )

    feature_df["target_diff_seconds"] = (
        feature_df["actual_target_timestamp"] - feature_df["target_timestamp"]
    ).abs().dt.total_seconds()

    feature_df = feature_df.dropna(subset=["target_ens160_aqi_15min"]).reset_index(drop=True)

    feature_df = feature_df.drop(
        columns=[
            "target_timestamp",
            "actual_target_timestamp",
            "target_diff_seconds",
        ]
    )

    print(feature_df.shape)
    feature_df.head()

    return feature_df


def missing_value(feature_df):
    print("\n========== Missing Value ==========")

    cols_to_fill = ["bme688_gas_resistance", "bme688_pressure"]

    feature_df = feature_df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    feature_df[cols_to_fill] = (
        feature_df
        .groupby("station_id")[cols_to_fill]
        .transform(lambda x: x.interpolate(method="linear"))
    )

    feature_df[cols_to_fill] = (
        feature_df
        .groupby("station_id")[cols_to_fill]
        .ffill()
        .bfill()
    )

    print(feature_df.shape)
    feature_df.head()

    return feature_df


def clipping_outliers(feature_df):
    print("\n========== Clipping Outliers ==========")

    numeric_cols = [
        "ens160_aqi",
        "ens160_tvoc",
        "bme688_gas_resistance",
        "bme688_pressure",
        "scd41_temperature",
        "scd41_humidity",
        "scd41_co2",
        "target_ens160_aqi_15min",
    ]

    for col in numeric_cols:
        Q1 = feature_df[col].quantile(0.25)
        Q3 = feature_df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        feature_df[col] = feature_df[col].clip(lower=lower, upper=upper)

    print(feature_df.shape)
    feature_df.head()

    return feature_df


def data_preprocessing_AQI():
    df = data_loading()
    feature_df = choose_feature(df)
    feature_df = resample_with_15_min(feature_df)
    feature_df = add_15_min_ahead_column(feature_df)
    feature_df = missing_value(feature_df)
    feature_df = clipping_outliers(feature_df)

    return feature_df