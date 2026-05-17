import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


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

    station_counts = df[STATION_COLUMN].value_counts()

    useful_stations = station_counts[station_counts >= 100].index

    df = df[df[STATION_COLUMN].isin(useful_stations)]

    print("After station filtering:", df.shape)

    df = add_time_features(df)

    df = df[BASE_FEATURES + [STATION_COLUMN]]

    print("Selected features:")
    print(BASE_FEATURES)

    print("After feature selection:", df.shape)

    df = (
        df.groupby(STATION_COLUMN)
        .resample('15min')
        .mean()
        .drop(columns=STATION_COLUMN, errors='ignore')
        .reset_index(level=0)
    )

    print("After resampling:", df.shape)

    return df[BASE_FEATURES + [STATION_COLUMN]]


def split_data_by_station(df):
    train_parts = []
    val_parts = []
    test_parts = []

    for station_id, station_df in df.groupby(STATION_COLUMN):

        station_df = station_df.sort_index()

        train_end = int(len(station_df) * 0.70)
        val_end = int(len(station_df) * 0.85)

        train_df = station_df.iloc[:train_end]
        val_df = station_df.iloc[train_end:val_end]
        test_df = station_df.iloc[val_end:]

        print(f"\nStation {station_id}")

        print(
            f"Train range: "
            f"{train_df.index.min()} -> {train_df.index.max()}"
        )

        print(
            f"Validation range: "
            f"{val_df.index.min()} -> {val_df.index.max()}"
        )

        print(
            f"Test range: "
            f"{test_df.index.min()} -> {test_df.index.max()}"
        )

        train_parts.append(train_df)
        val_parts.append(val_df)
        test_parts.append(test_df)

    return train_parts, val_parts, test_parts


def fill_missing_parts(parts):
    cleaned_parts = []

    input_features = [col for col in BASE_FEATURES if col != TARGET]

    for part in parts:
        part = part.copy()

        print("\nMissing values before interpolation:")
        print(part.isna().sum().sum())

        part = part.dropna(subset=[TARGET])

        part[input_features] = (
            part[input_features]
            .ffill()
            .bfill()
        )

        part = part.dropna()

        print("Missing values after interpolation:")
        print(part.isna().sum().sum())

        if len(part) > 0:
            cleaned_parts.append(part)

    return cleaned_parts


def build_model_parts(parts, station_ids):
    station_features = [f"station_{sid}" for sid in station_ids]

    model_features = BASE_FEATURES + station_features

    model_parts = []

    for part in parts:
        part = part.copy()

        for sid in station_ids:
            part[f"station_{sid}"] = (
                part[STATION_COLUMN] == sid
            ).astype(int)

        values = part[model_features].values

        if len(values) > 0:
            model_parts.append(values)

    return model_parts, model_features


def scale_data(train_parts, val_parts, test_parts):
    scaler = MinMaxScaler()

    scaler.fit(np.vstack(train_parts))

    print("\nScaler fitted ONLY on training data.")

    train_parts = [scaler.transform(p) for p in train_parts if len(p) > 0]
    val_parts = [scaler.transform(p) for p in val_parts if len(p) > 0]
    test_parts = [scaler.transform(p) for p in test_parts if len(p) > 0]

    return train_parts, val_parts, test_parts


def create_sequences(data_parts, model_features, seq_length=24):
    X, y = [], []

    target_index = model_features.index(TARGET)

    for data in data_parts:

        print("\nOriginal data shape before sequencing:", data.shape)

        if len(data) <= seq_length:
            continue

        for i in range(len(data) - seq_length):

            X.append(data[i:i + seq_length])

            y.append(data[i + seq_length, target_index])

    X = np.array(X)
    y = np.array(y)

    print("Sequence input shape:", X.shape)
    print("Sequence target shape:", y.shape)

    if len(X) == 0:
        raise ValueError(
            "No CNN-LSTM sequences were created. Use a shorter seq_length or check missing data."
        )

    return X, y


def create_loader(X, y, batch_size=32, shuffle=False):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle
    )

    sample_X, sample_y = next(iter(loader))

    print("Batch input shape:", sample_X.shape)
    print("Batch target shape:", sample_y.shape)

    return loader


def prepare_cnn_lstm_data(df, seq_length=24, batch_size=32):
    df = preprocess_data(df)

    station_ids = sorted(df[STATION_COLUMN].unique().tolist())

    train_parts, val_parts, test_parts = split_data_by_station(df)

    train_parts = fill_missing_parts(train_parts)
    val_parts = fill_missing_parts(val_parts)
    test_parts = fill_missing_parts(test_parts)

    train_parts, model_features = build_model_parts(
        train_parts,
        station_ids
    )

    val_parts, _ = build_model_parts(
        val_parts,
        station_ids
    )

    test_parts, _ = build_model_parts(
        test_parts,
        station_ids
    )

    train_parts, val_parts, test_parts = scale_data(
        train_parts,
        val_parts,
        test_parts
    )

    X_train, y_train = create_sequences(
        train_parts,
        model_features,
        seq_length
    )

    X_val, y_val = create_sequences(
        val_parts,
        model_features,
        seq_length
    )

    X_test, y_test = create_sequences(
        test_parts,
        model_features,
        seq_length
    )

    print("\n========== FINAL DATA SHAPES ==========")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    train_loader = create_loader(
        X_train,
        y_train,
        batch_size,
        shuffle=True
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
        X_train.shape[2]
    )


class CNNLSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            input_size,
            128,
            kernel_size=5,
            padding=2
        )

        self.conv2 = nn.Conv1d(
            128,
            128,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            128,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.dropout(
            self.relu(self.conv1(x))
        )

        x = self.dropout(
            self.relu(self.conv2(x))
        )

        x = x.permute(0, 2, 1)

        output, _ = self.lstm(x)

        return self.fc(output[:, -1, :])


def evaluate_loss(model, loader, criterion):
    model.eval()

    total_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            predictions = model(X_batch).view(-1)

            loss = criterion(predictions, y_batch)

            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=10
):
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch).view(-1)

            loss = criterion(predictions, y_batch)

            loss.backward()

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
            f"Val Loss: {val_loss:.6f}"
        )

    return model, train_losses, val_losses


def evaluate_model(model, test_loader):
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            outputs = model(X_batch).view(-1)

            predictions.extend(outputs.numpy())
            actuals.extend(y_batch.numpy())

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)

    print("\n========== MODEL EVALUATION ==========")

    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

    return predictions, actuals, mse, mae, rmse


def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(8, 4))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Train vs Validation Loss")

    plt.legend()

    plt.show()


def plot_predictions(actuals, predictions):
    plt.figure(figsize=(10, 4))

    plt.plot(actuals, label="Actual")
    plt.plot(predictions, label="Predicted")

    plt.legend()

    plt.title("Actual vs Predicted CO2 for CNN-LSTM")

    plt.show()


def run_cnn_lstm_model(
    df,
    epochs=10,
    seq_length=24,
    batch_size=32,
    show_prediction_plot=True
):
    (
        train_loader,
        val_loader,
        test_loader,
        input_size
    ) = prepare_cnn_lstm_data(
        df,
        seq_length=seq_length,
        batch_size=batch_size
    )

    model = CNNLSTMModel(
        input_size=input_size
    )

    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs
    )

    predictions, actuals, mse, mae, rmse = evaluate_model(
        model,
        test_loader
    )

    plot_loss_curves(
        train_losses,
        val_losses
    )

    if show_prediction_plot:
        plot_predictions(actuals, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "actuals": actuals,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
