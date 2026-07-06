import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_time_features(df):
    df = df.copy()

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"])
    else:
        timestamps = pd.Series(pd.to_datetime(df.index), index=df.index)

    hour = timestamps.dt.hour
    dayofweek = timestamps.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    df["is_weekend"] = (dayofweek >= 5).astype(int)

    return df


def preprocess_data(
    df,
    base_features,
    station_column,
    resample_time
):
    df = df.copy()

    print("\n========== Feature Selecting ==========")

    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    df = df.reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    sensor_features = [
        column for column in base_features
        if column in df.columns
    ]

    # Station 6 is removed to match the N-BEATS preprocessing pipeline.
    df = df[df[station_column] != 6].copy()
    df = df[sensor_features + ["timestamp", station_column]]
    print(df.shape)

    resample_label = (
        "15 Minutes"
        if str(resample_time).lower() == "15min"
        else resample_time
    )
    print(f"\n========== Data Resampling with {resample_label} ==========")
    df = (
        df
        .sort_values([station_column, "timestamp"])
        .set_index("timestamp")
        .groupby(station_column)
        [sensor_features]
        .resample(resample_time)
        .mean()
        .reset_index()
    )

    filled_station_parts = []
    for _, station_data in df.groupby(station_column, sort=False):
        station_data = station_data.copy()
        station_data[sensor_features] = (
            station_data[sensor_features]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )
        filled_station_parts.append(station_data)

    df = pd.concat(filled_station_parts, ignore_index=True)

    # Match the usable rows after the N-BEATS 15-minute-ahead step without
    # adding a separate ahead target column. LSTM creates multi-step targets
    # during window generation.
    df = (
        df
        .sort_values([station_column, "timestamp"])
        .groupby(station_column, group_keys=False)
        .head(-1)
        .reset_index(drop=True)
    )

    print(df.shape)

    df = add_time_features(df)

    return df[base_features + ["timestamp", station_column]]


def train_val_test_spliting(
    feature_df,
    station_column,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15
):
    print("\n========== TRAIN, VALIDATION AND TEST SPLIT ==========")

    feature_df = feature_df.copy()

    if "timestamp" not in feature_df.columns:
        feature_df["timestamp"] = feature_df.index

    # Keep timestamp only as a column. After resampling, pandas can also keep
    # timestamp as the index name, which makes sort_values("timestamp") ambiguous.
    feature_df = feature_df.reset_index(drop=True)

    def split_station_data(station_data):
        station_data = station_data.sort_values("timestamp")
        n_rows = len(station_data)
        train_end = int(n_rows * train_ratio)
        val_end = train_end + int(n_rows * val_ratio)

        return (
            station_data.iloc[:train_end],
            station_data.iloc[train_end:val_end],
            station_data.iloc[val_end:],
        )

    train_parts = []
    val_parts = []
    test_parts = []

    for station_id, station_data in feature_df.groupby(station_column, sort=True):
        station_train, station_val, station_test = split_station_data(
            station_data
        )

        train_parts.append(station_train)
        val_parts.append(station_val)
        test_parts.append(station_test)

        print(f"\nStation {station_id}")
        print(
            f"Train: "
            f"{station_train['timestamp'].min()} "
            f"-> "
            f"{station_train['timestamp'].max()} "
            f"Shape: {station_train.shape}"
        )
        print(
            f"Validation: "
            f"{station_val['timestamp'].min()} "
            f"-> "
            f"{station_val['timestamp'].max()} "
            f"Shape: {station_val.shape}"
        )
        print(
            f"Test: "
            f"{station_test['timestamp'].min()} "
            f"-> "
            f"{station_test['timestamp'].max()} "
            f"Shape: {station_test.shape}"
        )

    train_df = pd.concat(train_parts).sort_values(
        [station_column, "timestamp"]
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(
        [station_column, "timestamp"]
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(
        [station_column, "timestamp"]
    ).reset_index(drop=True)

    total_rows = len(feature_df)

    print("\nOverall Train shape:", train_df.shape)
    print("Overall Validation shape:", val_df.shape)
    print("Overall Test shape:", test_df.shape)

    print("Train percentage:", len(train_df) / total_rows * 100)
    print("Validation percentage:", len(val_df) / total_rows * 100)
    print("Test percentage:", len(test_df) / total_rows * 100)

    split_counts = pd.concat(
        [
            train_df.groupby(station_column).size().rename("train"),
            val_df.groupby(station_column).size().rename("val"),
            test_df.groupby(station_column).size().rename("test"),
        ],
        axis=1
    ).fillna(0).astype(int)

    print("\nRows per station:")
    print(split_counts)

    return train_df, val_df, test_df


def drop_short_stations_for_windowing(
    feature_df,
    station_column,
    input_seq_length,
    output_seq_length
):
    required_rows = input_seq_length + output_seq_length + 1
    station_counts = feature_df.groupby(station_column).size()
    keep_station_ids = station_counts[
        station_counts >= required_rows
    ].index.tolist()
    dropped_station_ids = station_counts[
        station_counts < required_rows
    ].index.tolist()

    print("\n========== SHORT STATION FILTER ==========")
    print("Required rows per station:", required_rows)
    print("Stations kept:", keep_station_ids)
    print("Stations removed:", dropped_station_ids)

    if len(keep_station_ids) == 0:
        raise ValueError(
            "No station has enough rows for the selected input/output window."
        )

    return (
        feature_df[feature_df[station_column].isin(keep_station_ids)]
        .copy()
        .reset_index(drop=True)
    )


def fill_short_feature_gaps(series, max_fill_steps):
    if max_fill_steps <= 0 or not series.isna().any():
        return series

    missing_mask = series.isna()
    run_ids = missing_mask.ne(missing_mask.shift()).cumsum()
    missing_run_lengths = missing_mask.groupby(run_ids).transform("sum")
    short_gap_mask = missing_mask & (
        missing_run_lengths <= max_fill_steps
    )

    nearby_values = series.ffill().bfill()
    filled_series = series.copy()
    filled_series.loc[short_gap_mask] = nearby_values.loc[short_gap_mask]

    return filled_series


def fill_missing_dataframe(
    df,
    base_features,
    target_column,
    station_column,
    segment_column,
    resample_time,
    max_fill_steps,
    use_gap_aware_segments=False
):
    cleaned_parts = []
    input_features = [col for col in base_features if col != target_column]
    expected_interval = pd.Timedelta(resample_time)
    expected_interval_seconds = expected_interval.total_seconds()
    for station_id, station_df in df.groupby(station_column, sort=True):
        station_df = station_df.copy()
        station_df["timestamp"] = pd.to_datetime(
            station_df["timestamp"],
            errors="coerce"
        )
        station_df = (
            station_df
            .dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .copy()
        )

        print(f"\nMissing values before limited filling (Station {station_id}):")
        print(station_df.isna().sum().sum())

        station_df = station_df.dropna(subset=[target_column])
        station_df[input_features] = (
            station_df[input_features]
            .apply(
                fill_short_feature_gaps,
                max_fill_steps=max_fill_steps
            )
        )
        station_df = station_df.dropna()

        print(f"Missing values after limited filling (Station {station_id}):")
        print(station_df.isna().sum().sum())

        if len(station_df) > 0:
            if use_gap_aware_segments:
                timestamp_differences = station_df["timestamp"].diff()
                gap_seconds = (
                    timestamp_differences
                    .dt.total_seconds()
                    .fillna(0)
                )
                gap_mask = gap_seconds > expected_interval_seconds
                station_df[segment_column] = gap_mask.cumsum().astype(int)

                detected_gaps = int(gap_mask.sum())
                segment_count = int(station_df[segment_column].nunique())
                maximum_gap = pd.to_timedelta(gap_seconds.max(), unit="s")

                print(f"Detected timestamp gaps (Station {station_id}): {detected_gaps}")
                print(f"Continuous segments (Station {station_id}): {segment_count}")
                print(f"Largest timestamp difference (Station {station_id}): {maximum_gap}")
            else:
                station_df[segment_column] = 0
            cleaned_parts.append(station_df)

    if len(cleaned_parts) == 0:
        raise ValueError("No rows left after missing-value handling.")

    return pd.concat(cleaned_parts).sort_values(
        [station_column, "timestamp"]
    ).reset_index(drop=True)


def add_station_features(df, station_ids, station_column):
    df = df.copy()

    for station_id in station_ids:
        df[f"station_{station_id}"] = (
            df[station_column] == station_id
        ).astype(int)

    return df


def clip_outliers_from_train(
    train_df,
    val_df,
    test_df,
    numeric_columns,
    clip_factor
):
    print("\n========== TRAIN-DERIVED OUTLIER CLIPPING ==========")
    print("Clip factor:", clip_factor)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    for column in numeric_columns:
        q1 = train_df[column].quantile(0.25)
        q3 = train_df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - clip_factor * iqr
        upper = q3 + clip_factor * iqr

        print(f"{column}: lower={lower:.6f}, upper={upper:.6f}")

        train_df[column] = train_df[column].clip(lower=lower, upper=upper)
        val_df[column] = val_df[column].clip(lower=lower, upper=upper)
        test_df[column] = test_df[column].clip(lower=lower, upper=upper)

    return train_df, val_df, test_df


def scale_data(train_df, val_df, test_df, model_features, target_column):
    scaler = MinMaxScaler()

    scaler.fit(train_df[model_features])

    print("\nScaler fitted ONLY on training data.")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[model_features] = scaler.transform(train_df[model_features])
    val_df[model_features] = scaler.transform(val_df[model_features])
    test_df[model_features] = scaler.transform(test_df[model_features])

    print(
        "Scaled X train min/max:",
        train_df[model_features].min().min(),
        train_df[model_features].max().max()
    )
    print(
        "Scaled y train min/max:",
        train_df[target_column].min(),
        train_df[target_column].max()
    )

    return train_df, val_df, test_df, scaler


def add_future_target_columns(
    data,
    output_seq_length,
    target_column,
    station_column,
    segment_column=None
):
    data = data.copy()

    group_columns = [station_column]
    if segment_column is not None and segment_column in data.columns:
        group_columns.append(segment_column)

    data = data.sort_values(group_columns + ["timestamp"]).reset_index(drop=True)

    for step in range(1, output_seq_length + 1):
        future_column = f"{target_column}_ahead_step_{step}"
        data[future_column] = (
            data
            .groupby(group_columns, sort=False)[target_column]
            .shift(-step)
        )

    return data


def build_future_target_reference(
    train_df,
    val_df,
    test_df,
    output_seq_length,
    target_column,
    station_column,
    segment_column=None,
    max_rows_per_split=1000
):
    reference_parts = []

    split_frames = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    for split_name, split_df in split_frames.items():
        future_df = add_future_target_columns(
            split_df,
            output_seq_length=output_seq_length,
            target_column=target_column,
            station_column=station_column,
            segment_column=segment_column
        )

        future_columns = [
            f"{target_column}_ahead_step_{step}"
            for step in range(1, output_seq_length + 1)
        ]

        base_columns = [station_column, "timestamp", target_column]
        if segment_column is not None and segment_column in future_df.columns:
            base_columns.insert(1, segment_column)

        future_df = future_df[base_columns + future_columns].copy()
        future_df.insert(0, "split", split_name)

        reference_parts.append(future_df.head(max_rows_per_split))

    return pd.concat(reference_parts, ignore_index=True)


def save_future_target_reference(
    future_target_reference,
    results_dir
):
    if future_target_reference is None or len(future_target_reference) == 0:
        return None

    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)

    save_path = os.path.join(
        main_plots_dir,
        "future_target_reference.csv"
    )

    future_target_reference.to_csv(save_path, index=False)
    print("Saved future target reference:", save_path)

    return save_path
