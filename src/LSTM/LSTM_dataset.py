import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset

from src.LSTM.LSTM_config import (
    BASE_FEATURES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_OUTLIERS,
    DEFAULT_DROP_SHORT_STATIONS,
    DEFAULT_INPUT_SEQ_LENGTH,
    DEFAULT_MAX_FILL_STEPS,
    DEFAULT_N_SCATTERING_FEATURES,
    DEFAULT_OUTLIER_CLIP_FACTOR,
    DEFAULT_OUTPUT_SEQ_LENGTH,
    DEFAULT_RESAMPLE_TIME,
    DEFAULT_SCATTERING_J,
    DEFAULT_SCATTERING_Q,
    DEFAULT_USE_GAP_AWARE_SEGMENTS,
    DEFAULT_USE_STATION_ONE_HOT,
    DEFAULT_USE_SCATTERING,
    SEGMENT_COLUMN,
    STATION_COLUMN,
    TARGET,
)
from src.LSTM.LSTM_data_processing import (
    add_continuous_segments,
    add_station_features,
    add_time_features,
    build_future_target_reference,
    clip_outliers_dataframe,
    # drop_short_stations_for_windowing,
    fill_missing_dataframe,
    preprocess_data,
    scale_data,
    train_val_test_spliting,
)
from src.LSTM.LSTM_scattering import (
    build_scattering_transform,
    compute_static_scattering_features,
    get_scattering_feature_names,
)


def create_sequences(
    data,
    model_features,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_transform=None,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES,
    split_name=None
):
    X_parts, y_parts = [], []
    target_index = model_features.index(TARGET)
    segment_summary = []

    if SEGMENT_COLUMN not in data.columns:
        raise ValueError(
            f"Missing {SEGMENT_COLUMN}. Run fill_missing_dataframe() "
            "before creating sequences."
        )

    data = data.sort_values(
        [STATION_COLUMN, SEGMENT_COLUMN, "timestamp"]
    )

    grouped_segments = data.groupby(
        [STATION_COLUMN, SEGMENT_COLUMN],
        sort=True
    )

    for (station_id, segment_id), segment_data in grouped_segments:
        values = np.ascontiguousarray(
            segment_data[model_features].to_numpy(dtype=np.float32)
        )

        required_length = input_seq_length + output_seq_length
        created_count = max(0, len(values) - required_length + 1)

        if len(values) <= required_length:
            segment_summary.append(
                {
                    "station_id": station_id,
                    "segment": segment_id,
                    "rows": len(values),
                    "windows": 0,
                    "status": "skipped_short",
                }
            )
            continue

        segment_summary.append(
            {
                "station_id": station_id,
                "segment": segment_id,
                "rows": len(values),
                "windows": created_count,
                "status": "used",
            }
        )

        if not use_scattering:
            input_windows = sliding_window_view(
                values,
                window_shape=input_seq_length,
                axis=0
            )
            input_windows = input_windows[:created_count].transpose(0, 2, 1)

            target_values = values[input_seq_length:, target_index]
            target_windows = sliding_window_view(
                target_values,
                window_shape=output_seq_length
            )[:created_count]

            X_parts.append(
                np.ascontiguousarray(input_windows, dtype=np.float32)
            )
            y_parts.append(
                np.ascontiguousarray(target_windows, dtype=np.float32)
            )
            continue

        for i in range(created_count):
            input_window = values[i:i + input_seq_length]

            if scattering_transform is None:
                raise ValueError(
                    "scattering_transform must be provided when "
                    "use_scattering=True"
                )

            co2_window = input_window[:, target_index]
            static_scattering_vector = compute_static_scattering_features(
                co2_window,
                scattering_transform,
                n_scattering_features=n_scattering_features
            )
            repeated_scattering = np.repeat(
                static_scattering_vector.reshape(1, -1),
                input_seq_length,
                axis=0
            )
            input_window = np.concatenate(
                [input_window, repeated_scattering],
                axis=1
            )

            X_parts.append(input_window.astype(np.float32, copy=False))
            y_parts.append(
                values[
                    i + input_seq_length:
                    i + input_seq_length + output_seq_length,
                    target_index
                ].astype(np.float32, copy=False)
            )

    if use_scattering:
        X = np.asarray(X_parts, dtype=np.float32)
        y = np.asarray(y_parts, dtype=np.float32)
    elif X_parts:
        X = np.concatenate(X_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
    else:
        X = np.empty(
            (0, input_seq_length, len(model_features)),
            dtype=np.float32
        )
        y = np.empty((0, output_seq_length), dtype=np.float32)

    label = f"{split_name} " if split_name else ""
    print(f"{label}sequence input shape:", X.shape)
    print(f"{label}sequence target shape:", y.shape)

    segment_summary_df = pd.DataFrame(segment_summary)
    if len(segment_summary_df) > 0:
        skipped_segments = int(
            (segment_summary_df["status"] == "skipped_short").sum()
        )
        print(
            f"{label}continuous segments:",
            len(segment_summary_df),
            "| skipped short:",
            skipped_segments
        )

    if len(X) == 0:
        raise ValueError(
            "\nNo LSTM sequences were created.\n"
            "Possible reasons:\n"
            "- selected station has too few rows\n"
            "- too many missing values removed\n"
            "- input/output sequence lengths are too large\n"
            "- train/validation/test split is too small"
        )

    return X, y

def create_loader(X, y, batch_size=32, shuffle=False, split_name=None):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)

    return loader

def prepare_lstm_data(
    df,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    resample_time=DEFAULT_RESAMPLE_TIME,
    max_fill_steps=DEFAULT_MAX_FILL_STEPS,
    drop_short_stations=DEFAULT_DROP_SHORT_STATIONS,
    clip_outliers=DEFAULT_CLIP_OUTLIERS,
    outlier_clip_factor=DEFAULT_OUTLIER_CLIP_FACTOR,
    use_gap_aware_segments=DEFAULT_USE_GAP_AWARE_SEGMENTS,
    use_station_one_hot=DEFAULT_USE_STATION_ONE_HOT,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    df = preprocess_data(
        df,
        base_features=BASE_FEATURES,
        station_column=STATION_COLUMN,
        resample_time=resample_time
    )

    # Disabled for N-BEATS-matched preprocessing output. Window creation already
    # skips segments that are too short after the train/validation/test split.
    # if drop_short_stations:
    #     df = drop_short_stations_for_windowing(
    #         df,
    #         station_column=STATION_COLUMN,
    #         input_seq_length=input_seq_length,
    #         output_seq_length=output_seq_length
    #     )

    station_ids = sorted(df[STATION_COLUMN].unique().tolist())

    print("\n========== Data Gap Handling ==========")
    print("Expected timestamp interval:", resample_time)
    print("Gap-aware sequence generation:", use_gap_aware_segments)
    # max_fill_steps is kept in the config for folder naming/backward
    # compatibility, but the N-BEATS-matched missing-value step interpolates
    # after resampling instead of doing limited gap filling.
    print(
        "Sequences crossing detected timestamp gaps:",
        "disabled" if use_gap_aware_segments else "allowed"
    )

    df = fill_missing_dataframe(
        df,
        base_features=BASE_FEATURES,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN,
        resample_time=resample_time,
        max_fill_steps=max_fill_steps
    )

    if clip_outliers:
        numeric_columns = [
            column for column in BASE_FEATURES
            if column in df.columns
        ]

        df = clip_outliers_dataframe(
            df,
            numeric_columns,
            clip_factor=outlier_clip_factor
        )
    else:
        print("\n========== OUTLIER CLIPPING DISABLED ==========")

    train_df, val_df, test_df = train_val_test_spliting(
        df,
        station_column=STATION_COLUMN
    )

    train_df = add_time_features(train_df)
    val_df = add_time_features(val_df)
    test_df = add_time_features(test_df)

    if use_gap_aware_segments:
        train_df = add_continuous_segments(
            train_df,
            station_column=STATION_COLUMN,
            segment_column=SEGMENT_COLUMN,
            resample_time=resample_time
        )
        val_df = add_continuous_segments(
            val_df,
            station_column=STATION_COLUMN,
            segment_column=SEGMENT_COLUMN,
            resample_time=resample_time
        )
        test_df = add_continuous_segments(
            test_df,
            station_column=STATION_COLUMN,
            segment_column=SEGMENT_COLUMN,
            resample_time=resample_time
        )
    else:
        train_df[SEGMENT_COLUMN] = 0
        val_df[SEGMENT_COLUMN] = 0
        test_df[SEGMENT_COLUMN] = 0

    print("\n========== Future Target Reference ==========")
    print(
        "Creating ahead target columns for analysis only:",
        f"step 1 to step {output_seq_length}"
    )
    print("These columns are not used as LSTM input features.")
    future_target_reference = build_future_target_reference(
        train_df,
        val_df,
        test_df,
        output_seq_length=output_seq_length,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN
    )
    print("Future target reference shape:", future_target_reference.shape)

    print("\n========== Station Feature Encoding ==========")
    print("Use station one-hot features:", use_station_one_hot)

    if use_station_one_hot:
        train_df = add_station_features(
            train_df,
            station_ids,
            station_column=STATION_COLUMN
        )
        val_df = add_station_features(
            val_df,
            station_ids,
            station_column=STATION_COLUMN
        )
        test_df = add_station_features(
            test_df,
            station_ids,
            station_column=STATION_COLUMN
        )

        station_features = [
            f"station_{station_id}"
            for station_id in station_ids
        ]
        print("Station one-hot features:", station_features)
    else:
        station_features = []
        print("Station one-hot features: disabled")

    # station_id itself is still only used for splitting/window creation.
    model_features = BASE_FEATURES + station_features

    train_df, val_df, test_df, scaler = scale_data(
        train_df,
        val_df,
        test_df,
        model_features,
        target_column=TARGET
    )

    scattering_transform = None
    scattering_feature_names = []

    print("\n========== Feature Configuration ==========")
    print("Base dynamic feature count:", len(model_features))
    print("Use scattering features:", use_scattering)

    if use_scattering:
        print("\n========== SCATTERING WAVELET FEATURES ==========")
        print(
            "Scattering source signal:",
            f"scaled {TARGET} input window"
        )
        print("Scattering J:", scattering_j)
        print("Scattering Q:", scattering_q)
        print("Static scattering features:", n_scattering_features)

        scattering_transform = build_scattering_transform(
            input_seq_length=input_seq_length,
            scattering_j=scattering_j,
            scattering_q=scattering_q
        )
        scattering_feature_names = get_scattering_feature_names(
            n_scattering_features
        )
    else:
        print("Static scattering features: 0")

    print("\n========== Creating Window ==========")
    X_train, y_train = create_sequences(
        train_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features,
        split_name="Train"
    )
    X_val, y_val = create_sequences(
        val_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features,
        split_name="Val"
    )
    X_test, y_test = create_sequences(
        test_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features,
        split_name="Test"
    )

    model_features = model_features + scattering_feature_names

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("Final model feature count:", len(model_features))
    if scattering_feature_names:
        print("Scattering feature names:", scattering_feature_names)

    print("\n========== Tensor and DataLoader ==========")
    train_loader = create_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        split_name="Train"
    )
    val_loader = create_loader(
        X_val,
        y_val,
        batch_size,
        split_name="Val"
    )
    test_loader = create_loader(
        X_test,
        y_test,
        batch_size,
        split_name="Test"
    )
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[2],
        X_train,
        X_test,
        model_features,
        future_target_reference
    )
