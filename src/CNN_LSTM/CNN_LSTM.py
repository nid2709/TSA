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


# def split_data_by_station(df):

#     train_parts = []
#     val_parts = []
#     test_parts = []

#     for station_id, station_df in df.groupby(STATION_COLUMN):

#         station_df = station_df.sort_index()

#         train_end = int(len(station_df) * 0.70)
#         val_end = int(len(station_df) * 0.85)

#         train_df = station_df.iloc[:train_end]
#         val_df = station_df.iloc[train_end:val_end]
#         test_df = station_df.iloc[val_end:]

#         print(f"\nStation {station_id}")

#         print(
#             f"Train range: "
#             f"{train_df.index.min()} -> {train_df.index.max()}"
#         )

#         print(
#             f"Validation range: "
#             f"{val_df.index.min()} -> {val_df.index.max()}"
#         )

#         print(
#             f"Test range: "
#             f"{test_df.index.min()} -> {test_df.index.max()}"
#         )

#         train_parts.append(train_df)
#         val_parts.append(val_df)
#         test_parts.append(test_df)

#     return train_parts, val_parts, test_parts

#New with cross-station experiments
def split_data_by_station(df):

    train_station = 3
    val_station = 4
    test_station = 5

    train_df = df[
        df[STATION_COLUMN] == train_station
    ].sort_index()

    val_df = df[
        df[STATION_COLUMN] == val_station
    ].sort_index()

    test_df = df[
        df[STATION_COLUMN] == test_station
    ].sort_index()

    print("\n========== CROSS-STATION SPLIT ==========")

    print(f"\nTraining Station: {train_station}")
    print("Train shape:", train_df.shape)

    print(f"\nValidation Station: {val_station}")
    print("Validation shape:", val_df.shape)

    print(f"\nTesting Station: {test_station}")
    print("Test shape:", test_df.shape)

    train_parts = [train_df]
    val_parts = [val_df]
    test_parts = [test_df]

    return train_parts, val_parts, test_parts

def fill_missing_parts(parts):

    cleaned_parts = []

    input_features = [
        col for col in BASE_FEATURES
        if col != TARGET
    ]

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

    station_features = [
        f"station_{sid}"
        for sid in station_ids
    ]

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

    train_parts = [
        scaler.transform(p)
        for p in train_parts
        if len(p) > 0
    ]

    val_parts = [
        scaler.transform(p)
        for p in val_parts
        if len(p) > 0
    ]

    test_parts = [
        scaler.transform(p)
        for p in test_parts
        if len(p) > 0
    ]

    return train_parts, val_parts, test_parts


def create_sequences(
    data_parts,
    model_features,
    input_seq_length=24,
    output_seq_length=6
):

    X, y = [], []

    target_index = model_features.index(TARGET)

    for data in data_parts:

        print(
            "\nOriginal data shape before sequencing:",
            data.shape
        )

        required_length = (
            input_seq_length + output_seq_length
        )

        if len(data) <= required_length:

            print(
                "\nSkipping sequence generation:"
            )

            print(
                f"Available rows: {len(data)}"
            )

            print(
                f"Required minimum rows: "
                f"{required_length + 1}"
            )

            continue

        for i in range(
            len(data)
            - input_seq_length
            - output_seq_length
        ):

            # Input sequence
            X.append(
                data[
                    i:i + input_seq_length
                ]
            )

            # Multi-step target sequence
            y.append(
                data[
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


def create_loader(
    X,
    y,
    batch_size=32,
    shuffle=False
):

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle
    )

    sample_X, sample_y = next(iter(loader))

    print("\n========== DATALOADER ==========")

    print("Batch input shape:", sample_X.shape)
    print("Batch target shape:", sample_y.shape)

    return loader


def prepare_cnn_lstm_data(
    df,
    input_seq_length=24,
    output_seq_length=6,
    batch_size=32
):

    df = preprocess_data(df)

    station_ids = sorted(
        df[STATION_COLUMN].unique().tolist()
    )

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
        input_seq_length,
        output_seq_length
    )

    X_val, y_val = create_sequences(
        val_parts,
        model_features,
        input_seq_length,
        output_seq_length
    )

    X_test, y_test = create_sequences(
        test_parts,
        model_features,
        input_seq_length,
        output_seq_length
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
        output_seq_length=6,
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

        # Multi-step forecasting output
        self.fc = nn.Linear(
            hidden_size,
            output_seq_length
        )

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

        output = output[:, -1, :]

        return self.fc(output)


def evaluate_loss(model, loader, criterion):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in loader:

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

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

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

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

            outputs = model(X_batch)

            predictions.extend(outputs.numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(
        actuals.flatten(),
        predictions.flatten()
    )

    mae = mean_absolute_error(
        actuals.flatten(),
        predictions.flatten()
    )

    rmse = np.sqrt(mse)

    print("\n========== MODEL EVALUATION ==========")

    print("Overall MSE:", mse)
    print("Overall MAE:", mae)
    print("Overall RMSE:", rmse)

    # print("\n========== PER-STEP FORECAST METRICS ==========")

    # for step in range(actuals.shape[1]):

    #     step_mse = mean_squared_error(
    #         actuals[:, step],
    #         predictions[:, step]
    #     )

    #     step_mae = mean_absolute_error(
    #         actuals[:, step],
    #         predictions[:, step]
    #     )

    #     step_rmse = np.sqrt(step_mse)

    #     print(f"\nForecast Step {step + 1}")

    #     print(f"MSE: {step_mse:.6f}")
    #     print(f"MAE: {step_mae:.6f}")
    #     print(f"RMSE: {step_rmse:.6f}")

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


def plot_predictions(
    actuals,
    predictions,
    forecast_step=1
):

    step_index = forecast_step - 1
    plt.figure(figsize=(10, 4))
    plt.plot(
        actuals[:, step_index],
        label="Actual"
    )
    plt.plot(
        predictions[:, step_index],
        label="Predicted"
    )
    plt.legend()
    plt.title(
        f"Actual vs Predicted CO2 for CNN-LSTM "
        f"(Forecast Step {forecast_step})"
    )
    plt.show()


def run_cnn_lstm_model(
    df,
    epochs=10,
    input_seq_length=24,
    output_seq_length=6,
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
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        batch_size=batch_size
    )

    model = CNNLSTMModel(
        input_size=input_size,
        output_seq_length=output_seq_length
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

        plot_predictions(
            actuals,
            predictions,
            forecast_step=1
        )

    return {
        "model": model,
        "predictions": predictions,
        "actuals": actuals,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
    }
