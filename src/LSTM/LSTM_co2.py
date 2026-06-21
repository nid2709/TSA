import copy
import os

MPL_CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ".matplotlib"
)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

try:
    from kymatio.numpy import Scattering1D
except ImportError:
    Scattering1D = None


BASE_FEATURES = [
    'ens160_aqi',
    'ens160_tvoc',
    'bme688_gas_resistance',
    'bme688_pressure',
    'scd41_temperature',
    'scd41_humidity',

    # Historical sensor value
    'scd41_co2',

    # Temporal cyclical features
    'hour_sin',
    'hour_cos',
    'dayofweek_sin',
    'dayofweek_cos',
    'is_weekend',
]

DEFAULT_TARGET = 'scd41_co2'
TARGET = DEFAULT_TARGET
STATION_COLUMN = 'station_id'

DEFAULT_INPUT_SEQ_LENGTH = 144
DEFAULT_OUTPUT_SEQ_LENGTH = 24
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 0.00001
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_RESAMPLE_TIME = '30min'
DEFAULT_DROPOUT_RATE = 0.15
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_LAYERS = 2

DEFAULT_USE_SCATTERING = False
DEFAULT_SCATTERING_J = 4
DEFAULT_SCATTERING_Q = 8
DEFAULT_N_SCATTERING_FEATURES = 8

TARGET_LABELS = {
    'scd41_co2': 'CO2',
    'scd41_temperature': 'Temperature',
    'scd41_humidity': 'Humidity',
    'ens160_aqi': 'AQI',
    'ens160_tvoc': 'TVOC',
    'bme688_pressure': 'Pressure',
    'bme688_gas_resistance': 'Gas Resistance',
}


def get_target_label(target_column):
    return TARGET_LABELS.get(
        target_column,
        target_column.replace('_', ' ').title()
    )


def get_lstm_results_dir(
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    resample_time=DEFAULT_RESAMPLE_TIME,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    num_layers=DEFAULT_NUM_LAYERS,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    folder_name = (
        f"LSTM_results_"
        f"IL{input_seq_length}_"
        f"OL{output_seq_length}_"
        f"BS{batch_size}_"
        f"EPOCH{epochs}_"
        f"LR{learning_rate}_"
        f"HS{hidden_size}_"
        f"RS{resample_time}_"
        f"DR{dropout_rate}_"
        f"WD{weight_decay}_"
        f"NL{num_layers}_"
        f"SWT{int(use_scattering)}_"
        f"SWJ{scattering_j if use_scattering else 0}_"
        f"SWQ{scattering_q if use_scattering else 0}_"
        f"SWF{n_scattering_features if use_scattering else 0}"
    )

    return os.path.join(project_root, folder_name)


def get_feature_columns(target_column):
    feature_columns = list(BASE_FEATURES)

    if target_column not in feature_columns:
        feature_columns.append(target_column)

    return feature_columns


def add_time_features(df):
    df = df.copy()

    hour = df.index.hour
    dayofweek = df.index.dayofweek

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    df['is_weekend'] = (dayofweek >= 5).astype(int)

    return df


def preprocess_data(
    df,
    target_column=DEFAULT_TARGET,
    resample_time=DEFAULT_RESAMPLE_TIME
):
    df = df.copy()
    feature_columns = get_feature_columns(target_column)

    print("\n========== PREPROCESSING ==========")
    print("Original dataset shape:", df.shape)


    print("\nDataset shape after station filtering:", df.shape)
    #print("Stations used:", sorted(useful_stations.tolist()))
    print(
        "\nStations available:",
        sorted(df[STATION_COLUMN].unique().tolist())
    )

    # Add time features
    df = add_time_features(df)

    missing_columns = [
        col for col in feature_columns + [STATION_COLUMN]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required column(s): "
            f"{missing_columns}"
        )

    # Feature selection
    df = df[feature_columns + [STATION_COLUMN]]

    print("\nTarget used:", target_column)
    print("Features used:", feature_columns)
    print("Dataset shape after feature selection:", df.shape)

    # Resample each station independently
    df = (
        df.groupby(STATION_COLUMN)
        .resample(resample_time)
        .mean()
        .drop(columns=STATION_COLUMN, errors='ignore')
        .reset_index(level=0)
    )

    print(f"\nDataset shape after {resample_time} resampling:", df.shape)
    print("Rows after resampling:", len(df))

    return df[feature_columns + [STATION_COLUMN]]


#This function is for all station data with chronological train/validation/test
# split per station. It follows the N-BEATS logic and uses station_id only for
# splitting/window creation, not as a model input feature.
def train_val_test_spliting(feature_df):

    print("\n========== TRAIN, VALIDATION AND TEST SPLIT ==========")

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

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

    for station_id, station_data in feature_df.groupby(STATION_COLUMN, sort=True):

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
        [STATION_COLUMN, "timestamp"]
    ).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(
        [STATION_COLUMN, "timestamp"]
    ).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(
        [STATION_COLUMN, "timestamp"]
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
            train_df.groupby(STATION_COLUMN).size().rename("train"),
            val_df.groupby(STATION_COLUMN).size().rename("val"),
            test_df.groupby(STATION_COLUMN).size().rename("test"),
        ],
        axis=1
    ).fillna(0).astype(int)

    print("\nRows per station:")
    print(split_counts)

    return train_df, val_df, test_df


def fill_missing_parts(
    parts,
    target_column=DEFAULT_TARGET,
    feature_columns=None
):

    cleaned_parts = []

    if feature_columns is None:
        feature_columns = get_feature_columns(target_column)

    input_features = [
        col for col in feature_columns
        if col != target_column
    ]

    for i, part in enumerate(parts):

        part = part.copy()

        print(f"\nMissing values BEFORE filling (Part {i+1}):")
        print(part.isna().sum().sum())

        # Remove rows with missing target
        part = part.dropna(subset=[target_column])

        # Fill only input features
        part[input_features] = (
            part[input_features]
            .ffill()
            .bfill()
        )

        print(f"Missing values AFTER filling (Part {i+1}):")
        print(part.isna().sum().sum())

        part = part.dropna()

        if len(part) > 0:
            cleaned_parts.append(part)

    return cleaned_parts


def build_model_parts(
    parts,
    station_ids,
    feature_columns=None,
    target_column=DEFAULT_TARGET
):

    if feature_columns is None:
        feature_columns = get_feature_columns(target_column)

    # One-hot encode station IDs so SHAP, PFI and IG can show station
    # contribution as part of the model input features.
    station_features = [
        f"station_{station_id}"
        for station_id in station_ids
    ]

    model_features = feature_columns + station_features

    model_parts = []

    for part in parts:

        part = part.copy()

        for station_id in station_ids:
            part[f"station_{station_id}"] = (
                part[STATION_COLUMN] == station_id
            ).astype(int)

        values = part[model_features].values

        if len(values) > 0:
            model_parts.append(values)

    return model_parts, model_features


def add_station_features(df, station_ids):

    df = df.copy()

    for station_id in station_ids:
        df[f"station_{station_id}"] = (
            df[STATION_COLUMN] == station_id
        ).astype(int)

    return df


def fill_missing_dataframe(
    df,
    target_column=DEFAULT_TARGET,
    feature_columns=None
):

    if feature_columns is None:
        feature_columns = get_feature_columns(target_column)

    input_features = [
        col for col in feature_columns
        if col != target_column
    ]

    cleaned_parts = []

    for station_id, station_df in df.groupby(STATION_COLUMN, sort=True):

        station_df = station_df.sort_values("timestamp").copy()

        print(f"\nMissing values BEFORE filling (Station {station_id}):")
        print(station_df.isna().sum().sum())

        # Remove rows with missing target
        station_df = station_df.dropna(subset=[target_column])

        # Fill only input features within the same station
        station_df[input_features] = (
            station_df[input_features]
            .ffill()
            .bfill()
        )

        print(f"Missing values AFTER filling (Station {station_id}):")
        print(station_df.isna().sum().sum())

        station_df = station_df.dropna()

        if len(station_df) > 0:
            cleaned_parts.append(station_df)

    if len(cleaned_parts) == 0:
        raise ValueError("No rows left after missing-value handling.")

    return pd.concat(cleaned_parts).sort_values(
        [STATION_COLUMN, "timestamp"]
    ).reset_index(drop=True)


def scale_data(
    train_df,
    val_df,
    test_df,
    model_features
):

    scaler = MinMaxScaler()

    # Fit ONLY on training data
    scaler.fit(train_df[model_features])

    print("\n========== SCALING ==========")
    print("MinMaxScaler fitted ONLY on training data")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[model_features] = scaler.transform(
        train_df[model_features]
    )

    val_df[model_features] = scaler.transform(
        val_df[model_features]
    )

    test_df[model_features] = scaler.transform(
        test_df[model_features]
    )

    return train_df, val_df, test_df


def get_scattering_feature_names(
    n_scattering_features,
    target_column=DEFAULT_TARGET
):
    return [
        f"scatter_{target_column}_{i + 1}"
        for i in range(n_scattering_features)
    ]


def build_scattering_transform(
    input_seq_length,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q
):
    if Scattering1D is None:
        raise ImportError(
            "Kymatio is required for scattering wavelet features. "
            "Install it with: pip install kymatio"
        )

    return Scattering1D(
        J=scattering_j,
        shape=input_seq_length,
        Q=scattering_q
    )


def compute_static_scattering_features(
    signal_window,
    scattering_transform,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    signal_window = np.asarray(signal_window, dtype=np.float32)
    scattering_coefficients = scattering_transform(signal_window)
    scattering_coefficients = np.asarray(scattering_coefficients)

    if scattering_coefficients.ndim == 2:
        static_vector = scattering_coefficients.mean(axis=1)
    else:
        static_vector = scattering_coefficients.reshape(-1)

    static_vector = static_vector[:n_scattering_features]

    if len(static_vector) < n_scattering_features:
        static_vector = np.pad(
            static_vector,
            (0, n_scattering_features - len(static_vector)),
            mode="constant"
        )

    return static_vector.astype(np.float32)


def create_sequences(
    data,
    model_features,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    target_column=DEFAULT_TARGET,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_transform=None,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):

    X, y = [], []

    target_index = model_features.index(target_column)

    data = data.sort_values([STATION_COLUMN, "timestamp"])

    for station_id, station_data in data.groupby(STATION_COLUMN, sort=True):

        # if len(data) <= (
        #     input_seq_length + output_seq_length
        # ):
        #     continue
        required_length = (
            input_seq_length + output_seq_length
        )

        values = station_data[model_features].values

        if len(values) <= required_length:

            print(
                "\nSkipping sequence generation:"
            )

            print(
                f"Station: {station_id}"
            )

            print(
                f"Available rows: {len(values)}"
            )

            print(
                f"Required minimum rows: "
                f"{required_length + 1}"
            )

            continue

        print(
            "\nOriginal station data shape before sequencing:",
            station_data.shape
        )

        for i in range(
            len(values)
            - input_seq_length
            - output_seq_length
        ):

            input_window = values[i:i + input_seq_length]

            if use_scattering:
                if scattering_transform is None:
                    raise ValueError(
                        "scattering_transform must be provided when "
                        "use_scattering=True"
                    )

                target_window = input_window[:, target_index]
                static_scattering_vector = compute_static_scattering_features(
                    target_window,
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

            X.append(input_window)

            # Multi-step output sequence
            y.append(
                values[
                    i + input_seq_length:
                    i + input_seq_length + output_seq_length,
                    target_index
                ]
            )

    if len(X) == 0:
        raise ValueError(
            "No sequences were created."
        )

    X = np.array(X)
    y = np.array(y)

    print("Sequence input shape:", X.shape)
    print("Sequence target shape:", y.shape)

    return X, y


def create_loader(
    X,
    y,
    batch_size=DEFAULT_BATCH_SIZE,
    shuffle=False,
    show_shape=False
):

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle
    )

    if show_shape:

        sample_X, sample_y = next(iter(loader))

        print("\n========== DATALOADER ==========")
        print("Batch input shape:", sample_X.shape)
        print("Batch target shape:", sample_y.shape)

    return loader


def prepare_lstm_data(
    df,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    target_column=DEFAULT_TARGET,
    resample_time=DEFAULT_RESAMPLE_TIME,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):

    feature_columns = get_feature_columns(target_column)

    df = preprocess_data(
        df,
        target_column,
        resample_time
    )

    station_ids = sorted(
        df[STATION_COLUMN].unique().tolist()
    )

    # Split BEFORE filling/scaling
    train_df, val_df, test_df = train_val_test_spliting(df)

    train_df = fill_missing_dataframe(
        train_df,
        target_column,
        feature_columns
    )
    val_df = fill_missing_dataframe(
        val_df,
        target_column,
        feature_columns
    )
    test_df = fill_missing_dataframe(
        test_df,
        target_column,
        feature_columns
    )

    train_df = add_station_features(train_df, station_ids)
    val_df = add_station_features(val_df, station_ids)
    test_df = add_station_features(test_df, station_ids)

    station_features = [
        f"station_{station_id}"
        for station_id in station_ids
    ]

    # Model features include station one-hot columns for Explainability plots.
    # station_id itself is still only used for splitting/window creation.
    model_features = feature_columns + station_features

    train_df, val_df, test_df = scale_data(
        train_df,
        val_df,
        test_df,
        model_features
    )

    scattering_transform = None
    scattering_feature_names = []

    print("\n========== FEATURE CONFIGURATION ==========")
    print("Base dynamic feature count:", len(model_features))
    print("Use scattering features:", use_scattering)

    if use_scattering:
        print("\n========== SCATTERING WAVELET FEATURES ==========")
        print(f"Scattering source signal: scaled {target_column} input window")
        print("Scattering J:", scattering_j)
        print("Scattering Q:", scattering_q)
        print("Static scattering features:", n_scattering_features)

        scattering_transform = build_scattering_transform(
            input_seq_length=input_seq_length,
            scattering_j=scattering_j,
            scattering_q=scattering_q
        )
        scattering_feature_names = get_scattering_feature_names(
            n_scattering_features,
            target_column=target_column
        )
    else:
        print("Static scattering features: 0")

    X_train, y_train = create_sequences(
        train_df,
        model_features,
        input_seq_length,
        output_seq_length,
        target_column,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )

    X_val, y_val = create_sequences(
        val_df,
        model_features,
        input_seq_length,
        output_seq_length,
        target_column,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )

    X_test, y_test = create_sequences(
        test_df,
        model_features,
        input_seq_length,
        output_seq_length,
        target_column,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )

    model_features = model_features + scattering_feature_names

    print("\n========== FINAL DATA SHAPES ==========")

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("Final input feature count:", X_train.shape[2])
    print("Final model features:", model_features)

    train_loader = create_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True,
        show_shape=True
    )

    val_loader = create_loader(
        X_val,
        y_val,
        batch_size
    )

    test_loader = create_loader(
        X_test,
        y_test,
        batch_size
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[2],
        X_train,
        X_test,
        model_features
    )


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout_rate=DEFAULT_DROPOUT_RATE
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        # for Monte carlo Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Multi-step forecasting output
        self.fc = nn.Linear(
            hidden_size,
            output_seq_length
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        # Use last timestep output
        output = output[:, -1, :]

        output = self.dropout(output) # monte carlo

        return self.fc(output)


def evaluate_loss(
    model,
    data_loader,
    criterion
):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            predictions = model(X_batch)

            total_loss += criterion(
                predictions,
                y_batch
            ).item()

    return total_loss / len(data_loader)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=DEFAULT_EPOCHS,
    patience=10,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    min_delta=1e-6
):

    # SmoothL1/Huber loss is less sensitive to rare CO2 spikes than MSE,
    # which can make the validation curve less jumpy.
    criterion = nn.SmoothL1Loss(beta=0.01)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-5
    )

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}, "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Best Val Loss:", best_val_loss)

    return model, train_losses, val_losses


def evaluate_model(model, test_loader):

    model.eval()

    predictions, actuals = [], []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            output = model(X_batch)

            predictions.extend(output.numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals.flatten(), predictions.flatten())

    print("\n========== MODEL EVALUATION ==========")

    print("Overall MSE:", mse)
    print("Overall MAE:", mae)
    print("Overall RMSE:", rmse)
    print("Overall R2 Score:", r2)


    return predictions, actuals, mse, mae, rmse, r2


def plot_loss_curves(
    train_losses,
    val_losses,
    results_dir=None
):

    plt.figure(figsize=(8, 4))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Train vs Validation Loss")

    plt.legend()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            "train_validation_loss.png"
        )
        plt.savefig(save_path, dpi=300)
        print("Saved plot:", save_path)

    #plt.show()
    plt.close()


def plot_predictions(
    actuals,
    predictions,
    forecast_step=1,
    max_plot_points=1500,
    target_column=DEFAULT_TARGET,
    results_dir=None
):

    step_index = forecast_step - 1
    target_label = get_target_label(target_column)

    if forecast_step < 1 or forecast_step > actuals.shape[1]:
        raise ValueError(
            f"forecast_step must be between 1 and {actuals.shape[1]}"
        )

    x_values = np.arange(len(actuals))
    actual_values = actuals[:, step_index]
    predicted_values = predictions[:, step_index]

    if max_plot_points is not None and len(x_values) > max_plot_points:
        x_values = x_values[:max_plot_points]
        actual_values = actual_values[:max_plot_points]
        predicted_values = predicted_values[:max_plot_points]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot one forecast horizon only. Flattening multi-step outputs mixes
    # overlapping windows and creates a misleading zig-zag plot.
    ax.plot(
        x_values,
        actual_values,
        label="Actual"
    )

    ax.plot(
        x_values,
        predicted_values,
        label="Predicted"
    )

    ax.set_xlabel("Test sample index")
    ax.set_ylabel(f"Scaled {target_label}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

    ax.legend()

    ax.set_title(
        f"Actual vs Predicted {target_label} for LSTM "
        f"(Forecast Step {forecast_step})"
    )

    fig.tight_layout()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            f"actual_vs_predicted_step_{forecast_step}.png"
        )
        plt.savefig(save_path, dpi=300)
        print("Saved plot:", save_path)

    #plt.show()
    plt.close()


def plot_forecast_comparison(
    actuals,
    predictions,
    target_column=DEFAULT_TARGET,
    results_dir=None
):

    output_seq_length = actuals.shape[1]

    plot_predictions(
        actuals,
        predictions,
        forecast_step=1,
        target_column=target_column,
        results_dir=results_dir
    )

    if output_seq_length > 1:
        plot_predictions(
            actuals,
            predictions,
            forecast_step=output_seq_length,
            target_column=target_column,
            results_dir=results_dir
        )


def run_lstm_model(
    df,
    epochs=DEFAULT_EPOCHS,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    target_column=DEFAULT_TARGET,
    learning_rate=DEFAULT_LEARNING_RATE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    num_layers=DEFAULT_NUM_LAYERS,
    resample_time=DEFAULT_RESAMPLE_TIME,
    show_prediction_plot=True,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    print("\n========== LSTM RUN CONFIGURATION ==========")
    print("Input sequence length:", input_seq_length)
    print("Output sequence length:", output_seq_length)
    print("Batch size:", batch_size)
    print("Epochs:", epochs)
    print("Learning rate:", learning_rate)
    print("Hidden size:", hidden_size)
    print("Resample time:", resample_time)
    print("Dropout rate:", dropout_rate)
    print("Weight decay:", weight_decay)
    print("Number of LSTM layers:", num_layers)
    print("Target column:", target_column)
    print("Use scattering:", use_scattering)
    if use_scattering:
        print("Scattering J:", scattering_j)
        print("Scattering Q:", scattering_q)
        print("Number of scattering features:", n_scattering_features)

    (
        train_loader,
        val_loader,
        test_loader,
        input_size,
        X_train,
        X_test,
        model_features
    ) = prepare_lstm_data (
        df,
        input_seq_length,
        output_seq_length,
        batch_size,
        target_column=target_column,
        resample_time=resample_time,
        use_scattering=use_scattering,
        scattering_j=scattering_j,
        scattering_q=scattering_q,
        n_scattering_features=n_scattering_features
    )

    model = LSTMModel(
        input_size=input_size,
        output_seq_length=output_seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    )

    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    predictions, actuals, mse, mae, rmse, r2 = evaluate_model(
        model,
        test_loader
    )

    results_dir = get_lstm_results_dir(
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        resample_time=resample_time,
        dropout_rate=dropout_rate,
        weight_decay=weight_decay,
        num_layers=num_layers,
        use_scattering=use_scattering,
        scattering_j=scattering_j,
        scattering_q=scattering_q,
        n_scattering_features=n_scattering_features
    )

    plot_loss_curves(
        train_losses,
        val_losses,
        results_dir=results_dir
    )

    if show_prediction_plot:
        plot_forecast_comparison(
            actuals,
            predictions,
            target_column,
            results_dir=results_dir
        )

    return {
        "model": model,
        "predictions": predictions,
        "actuals": actuals,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "target_column": target_column,
        "target_label": get_target_label(target_column),
        "results_dir": results_dir,

        # ADD THESE for Explainability techniques
        "X_train": X_train,
        "X_test": X_test,
        "model_features": model_features,
        "output_seq_length": output_seq_length,
        "resample_time": resample_time,
        "dropout_rate": dropout_rate,
        "learning_rate": learning_rate,
        "hidden_size": hidden_size,
        "weight_decay": weight_decay,
        "num_layers": num_layers,
        "use_scattering": use_scattering,
        "scattering_j": scattering_j,
        "scattering_q": scattering_q,
        "n_scattering_features": n_scattering_features,

        # ADD THESE for Deep Ensemble
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_size": input_size,
    }
