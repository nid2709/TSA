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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


BASE_FEATURES = [
    'ens160_aqi',
    'ens160_tvoc',
    'bme688_gas_resistance',
    'bme688_pressure',
    'scd41_temperature',
    'scd41_humidity',
    'scd41_co2',

    'hour_sin',
    'hour_cos',
    'dayofweek_sin',
    'dayofweek_cos',
    'is_weekend',
]

TARGET = 'scd41_co2'
STATION_COLUMN = 'station_id'

DEFAULT_INPUT_SEQ_LENGTH = 144
DEFAULT_OUTPUT_SEQ_LENGTH = 48
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.00001
DEFAULT_HIDDEN_SIZE = 128


def get_target_label(target_column):
    target_labels = {
        'scd41_co2': 'CO2',
        'scd41_temperature': 'Temperature',
        'scd41_humidity': 'Humidity',
        'ens160_aqi': 'AQI',
        'ens160_tvoc': 'TVOC',
        'bme688_pressure': 'Pressure',
        'bme688_gas_resistance': 'Gas Resistance',
    }

    return target_labels.get(
        target_column,
        target_column.replace('_', ' ').title()
    )


def get_cnn_lstm_results_dir(
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    epochs=DEFAULT_EPOCHS,
    learning_rate=DEFAULT_LEARNING_RATE,
    hidden_size=DEFAULT_HIDDEN_SIZE
):
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    folder_name = (
        f"CNN_LSTM_results_"
        f"IL{input_seq_length}_"
        f"OL{output_seq_length}_"
        f"BS{batch_size}_"
        f"EPOCH{epochs}_"
        f"LR{learning_rate}_"
        f"HS{hidden_size}"
    )

    return os.path.join(project_root, folder_name)


def add_time_features(df):
    df = df.copy()

    hour = df.index.hour
    dayofweek = df.index.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    df['dayofweek_sin'] = np.sin(2 * np.pi * dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * dayofweek / 7)

    df['is_weekend'] = (dayofweek >= 5).astype(int)

    return df


def preprocess_data(df):
    df = df.copy()

    print("\n========== PREPROCESSING ==========")
    print("Raw dataset shape:", df.shape)

    print(
        "\nStations available:",
        sorted(df[STATION_COLUMN].unique().tolist())
    )

    df = add_time_features(df)
    df = df[BASE_FEATURES + [STATION_COLUMN]]

    print("\nSelected features:")
    print(BASE_FEATURES)
    print("\nAfter feature selection:", df.shape)

    df = (
        df.groupby(STATION_COLUMN)
        .resample('15min')
        .mean()
        .drop(columns=STATION_COLUMN, errors='ignore')
        .reset_index(level=0)
    )

    print("\nAfter resampling:", df.shape)

    return df[BASE_FEATURES + [STATION_COLUMN]]


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


def fill_missing_parts(parts):
    cleaned_parts = []
    input_features = [col for col in BASE_FEATURES if col != TARGET]

    for part in parts:
        part = part.copy()

        print("\nMissing values before interpolation:")
        print(part.isna().sum().sum())

        part = part.dropna(subset=[TARGET])
        part[input_features] = part[input_features].ffill().bfill()
        part = part.dropna()

        print("Missing values after interpolation:")
        print(part.isna().sum().sum())

        if len(part) > 0:
            cleaned_parts.append(part)

    return cleaned_parts


def fill_missing_dataframe(df):
    cleaned_parts = []
    input_features = [col for col in BASE_FEATURES if col != TARGET]

    for station_id, station_df in df.groupby(STATION_COLUMN, sort=True):
        station_df = station_df.sort_values("timestamp").copy()

        print(f"\nMissing values before interpolation (Station {station_id}):")
        print(station_df.isna().sum().sum())

        station_df = station_df.dropna(subset=[TARGET])
        station_df[input_features] = station_df[input_features].ffill().bfill()
        station_df = station_df.dropna()

        print(f"Missing values after interpolation (Station {station_id}):")
        print(station_df.isna().sum().sum())

        if len(station_df) > 0:
            cleaned_parts.append(station_df)

    if len(cleaned_parts) == 0:
        raise ValueError("No rows left after missing-value handling.")

    return pd.concat(cleaned_parts).sort_values(
        [STATION_COLUMN, "timestamp"]
    ).reset_index(drop=True)


def add_station_features(df, station_ids):
    df = df.copy()

    for station_id in station_ids:
        df[f"station_{station_id}"] = (
            df[STATION_COLUMN] == station_id
        ).astype(int)

    return df


def scale_data(train_df, val_df, test_df, model_features):
    scaler = MinMaxScaler()
    scaler.fit(train_df[model_features])

    print("\nScaler fitted ONLY on training data.")

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[model_features] = scaler.transform(train_df[model_features])
    val_df[model_features] = scaler.transform(val_df[model_features])
    test_df[model_features] = scaler.transform(test_df[model_features])

    return train_df, val_df, test_df


def create_sequences(
    data,
    model_features,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH
):
    X, y = [], []
    target_index = model_features.index(TARGET)

    data = data.sort_values([STATION_COLUMN, "timestamp"])

    for station_id, station_data in data.groupby(STATION_COLUMN, sort=True):
        values = station_data[model_features].values

        print("\nOriginal station data shape before sequencing:", station_data.shape)
        required_length = input_seq_length + output_seq_length

        if len(values) <= required_length:
            print("\nSkipping sequence generation:")
            print(f"Station: {station_id}")
            print(f"Available rows: {len(values)}")
            print(f"Required minimum rows: {required_length + 1}")
            continue

        for i in range(len(values) - input_seq_length - output_seq_length):
            X.append(values[i:i + input_seq_length])
            y.append(
                values[
                    i + input_seq_length:
                    i + input_seq_length + output_seq_length,
                    target_index
                ]
            )

    X = np.array(X)
    y = np.array(y)

    print("Sequence input shape:", X.shape)
    print("Sequence target shape:", y.shape)

    if len(X) == 0:
        raise ValueError(
            "\nNo CNN-LSTM sequences were created.\n"
            "Possible reasons:\n"
            "- selected station has too few rows\n"
            "- too many missing values removed\n"
            "- input/output sequence lengths are too large\n"
            "- train/validation/test split is too small"
        )

    return X, y


def create_loader(X, y, batch_size=32, shuffle=False):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
    sample_X, sample_y = next(iter(loader))

    print("\n========== DATALOADER ==========")
    print("Batch input shape:", sample_X.shape)
    print("Batch target shape:", sample_y.shape)

    return loader


def prepare_cnn_lstm_data(
    df,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE
):
    df = preprocess_data(df)

    station_ids = sorted(df[STATION_COLUMN].unique().tolist())

    train_df, val_df, test_df = train_val_test_spliting(df)

    train_df = fill_missing_dataframe(train_df)
    val_df = fill_missing_dataframe(val_df)
    test_df = fill_missing_dataframe(test_df)

    train_df = add_station_features(train_df, station_ids)
    val_df = add_station_features(val_df, station_ids)
    test_df = add_station_features(test_df, station_ids)

    station_features = [
        f"station_{station_id}"
        for station_id in station_ids
    ]

    # Model features include station one-hot columns for Explainability plots.
    # station_id itself is still only used for splitting/window creation.
    model_features = BASE_FEATURES + station_features

    train_df, val_df, test_df = scale_data(
        train_df,
        val_df,
        test_df,
        model_features
    )

    X_train, y_train = create_sequences(train_df, model_features, input_seq_length, output_seq_length)
    X_val, y_val = create_sequences(val_df, model_features, input_seq_length, output_seq_length)
    X_test, y_test = create_sequences(test_df, model_features, input_seq_length, output_seq_length)

    print("\n========== FINAL DATA SHAPES ==========")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    train_loader = create_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = create_loader(X_val, y_val, batch_size)
    test_loader = create_loader(X_test, y_test, batch_size)

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train.shape[2],
        X_train,
        X_test,
        model_features
    )


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(128, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_seq_length)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output)


def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=DEFAULT_EPOCHS,
    patience=10,
    learning_rate=DEFAULT_LEARNING_RATE,
    min_delta=1e-6
):
    # SmoothL1/Huber loss is less sensitive to rare CO2 spikes than MSE,
    # which can make the validation curve less jumpy.
    criterion = nn.SmoothL1Loss(beta=0.01)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
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
            loss = criterion(predictions, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, criterion)

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
            outputs = model(X_batch)
            predictions.extend(outputs.numpy())
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


def plot_loss_curves(train_losses, val_losses, results_dir=None):
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
        #print("Saved plot:", save_path)

    plt.show()
    plt.close()


def plot_predictions(
    actuals,
    predictions,
    forecast_step=1,
    max_plot_points=500,
    results_dir=None
):
    step_index = forecast_step - 1

    if forecast_step < 1 or forecast_step > actuals.shape[1]:
        raise ValueError(f"forecast_step must be between 1 and {actuals.shape[1]}")

    x_values = np.arange(len(actuals))
    actual_values = actuals[:, step_index]
    predicted_values = predictions[:, step_index]

    if len(x_values) > max_plot_points:
        x_values = x_values[:max_plot_points]
        actual_values = actual_values[:max_plot_points]
        predicted_values = predicted_values[:max_plot_points]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_values, actual_values, label="Actual")
    ax.plot(x_values, predicted_values, label="Predicted")
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Scaled CO2")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend()
    ax.set_title(f"Actual vs Predicted CO2 for CNN-LSTM (Forecast Step {forecast_step})")
    fig.tight_layout()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            f"actual_vs_predicted_step_{forecast_step}.png"
        )
        plt.savefig(save_path, dpi=300)
        #print("Saved plot:", save_path)

    plt.show()
    plt.close()


def plot_forecast_comparison(actuals, predictions, results_dir=None):
    output_seq_length = actuals.shape[1]
    
    # Dynamic Plot 1: Horizon Step 1
    plot_predictions(
        actuals,
        predictions,
        forecast_step=1,
        results_dir=results_dir
    )

    # Dynamic Plot 2: Final Horizon Output Length Step
    if output_seq_length > 1:
        plot_predictions(
            actuals,
            predictions,
            forecast_step=output_seq_length,
            results_dir=results_dir
        )


def run_cnn_lstm_model(
    df,
    epochs=DEFAULT_EPOCHS,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    show_prediction_plot=True
):
    (
        train_loader,
        val_loader,
        test_loader,
        input_size,
        X_train,
        X_test,
        model_features
    ) = prepare_cnn_lstm_data(
        df,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        batch_size=batch_size
    )

    model = CNNLSTMModel(
        input_size=input_size,
        output_seq_length=output_seq_length
    )

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, epochs)
    predictions, actuals, mse, mae, rmse, r2 = evaluate_model(model, test_loader)

    results_dir = get_cnn_lstm_results_dir(
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        batch_size=batch_size,
        epochs=epochs
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
        "target_column": TARGET,
        "target_label": get_target_label(TARGET),
        "results_dir": results_dir,

        # ADD THESE for Explainability techniques
        "X_train": X_train,
        "X_test": X_test,
        "model_features": model_features,
        "output_seq_length": output_seq_length,

        # ADD THESE for Deep Ensemble
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "input_size": input_size,
    }
