import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


BASE_FEATURES = [
    'ens160_aqi',
    'ens160_tvoc',
    'bme688_gas_resistance',
    'bme688_pressure',
    'scd41_temperature',
    'scd41_humidity',

    # Historical CO2 values
    'scd41_co2',

    # Temporal cyclical features
    'hour_sin',
    'hour_cos',
    'dayofweek_sin',
    'dayofweek_cos',
    'is_weekend',
]

TARGET = 'scd41_co2'
STATION_COLUMN = 'station_id'

TARGET_FEATURES = [
    'scd41_co2',
    'scd41_temperature',
    'bme688_pressure',
    'scd41_humidity',
    'ens160_aqi',
    'ens160_tvoc',
]

TARGET_DISPLAY_NAMES = {
    'scd41_co2': 'CO2',
    'scd41_temperature': 'Temperature',
    'bme688_pressure': 'Pressure',
    'scd41_humidity': 'Humidity',
    'ens160_aqi': 'AQI',
    'ens160_tvoc': 'TVOC',
}


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


def preprocess_data(df):
    df = df.copy()

    print("\n========== PREPROCESSING ==========")
    print("Original dataset shape:", df.shape)

    # Remove stations with too few records
    station_counts = df[STATION_COLUMN].value_counts()
   # useful_stations = station_counts[station_counts >= 100].index
   # df = df[df[STATION_COLUMN].isin(useful_stations)]

    print("\nDataset shape after station filtering:", df.shape)
    #print("Stations used:", sorted(useful_stations.tolist()))
    print(
        "\nStations available:",
        sorted(df[STATION_COLUMN].unique().tolist())
    )

    # Add time features
    df = add_time_features(df)

    # Feature selection
    df = df[BASE_FEATURES + [STATION_COLUMN]]

    print("\nFeatures used:", BASE_FEATURES)
    print("Dataset shape after feature selection:", df.shape)

    # Resample each station independently
    df = (
        df.groupby(STATION_COLUMN)
        .resample('15min')
        .mean()
        .drop(columns=STATION_COLUMN, errors='ignore')
        .reset_index(level=0)
    )

    print("\nDataset shape after 15-minute resampling:", df.shape)
    print("Rows after resampling:", len(df))

    return df[BASE_FEATURES + [STATION_COLUMN]]

#This function is for all station data
# def split_data_by_station(df):
#     train_parts, val_parts, test_parts = [], [], []
#     for station_id, station_df in df.groupby(STATION_COLUMN):

#         station_df = station_df.sort_index()
#         # Chronological split
#         train_end = int(len(station_df) * 0.70)
#         val_end = int(len(station_df) * 0.85)

#         train_parts.append(station_df.iloc[:train_end])
#         val_parts.append(station_df.iloc[train_end:val_end])
#         test_parts.append(station_df.iloc[val_end:])

#         print(f"\nStation {station_id}")
#         print(
#             f"Train: "
#             f"{station_df.iloc[:train_end].index.min()} "
#             f"-> "
#             f"{station_df.iloc[:train_end].index.max()}"
#         )
#         print(
#             f"Validation: "
#             f"{station_df.iloc[train_end:val_end].index.min()} "
#             f"-> "
#             f"{station_df.iloc[train_end:val_end].index.max()}"
#         )
#         print(
#             f"Test: "
#             f"{station_df.iloc[val_end:].index.min()} "
#             f"-> "
#             f"{station_df.iloc[val_end:].index.max()}"
#         )
#     return train_parts, val_parts, test_parts

#Cross-Station Forecasting - This function is experiment with 3 stations
#If test station is unseen during training,its one-hot column will always be zero during training. 
# That is actually acceptable for this experiment.
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

    print("\n========== STATION-BASED SPLIT ==========")

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

#Leave-One-Station-Out - Better dynamoc approoach- only used 1 station for testing
# def split_data_by_station(df, test_station):

#     train_df = df[
#         df[STATION_COLUMN] != test_station
#     ].sort_index()

#     test_df = df[
#         df[STATION_COLUMN] == test_station
#     ].sort_index()

#     # Validation from training data
#     train_end = int(len(train_df) * 0.85)

#     final_train_df = train_df.iloc[:train_end]
#     val_df = train_df.iloc[train_end:]

#     print("\n========== LOSO SPLIT ==========")
#     print(f"\nTest Station: {test_station}")
#     print("Train shape:", final_train_df.shape)
#     print("Validation shape:", val_df.shape)
#     print("Test shape:", test_df.shape)

#     train_parts = [final_train_df]
#     val_parts = [val_df]
#     test_parts = [test_df]

#     return train_parts, val_parts, test_parts

def fill_missing_parts(parts, target=TARGET):

    cleaned_parts = []

    input_features = [
        col for col in BASE_FEATURES
        if col != target
    ]

    for i, part in enumerate(parts):

        part = part.copy()

        print(f"\nMissing values BEFORE filling (Part {i+1}):")
        print(part.isna().sum().sum())

        # Remove rows with missing target
        part = part.dropna(subset=[target])

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


def build_model_parts(parts, station_ids):

    # One-hot encode station IDs
    station_features = [
        f"station_{station_id}"
        for station_id in station_ids
    ]

    model_features = BASE_FEATURES + station_features

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


def scale_data(train_parts, val_parts, test_parts):

    scaler = MinMaxScaler()

    # Fit ONLY on training data
    scaler.fit(np.vstack(train_parts))

    print("\n========== SCALING ==========")
    print("MinMaxScaler fitted ONLY on training data")

    train_parts = [
        scaler.transform(part)
        for part in train_parts
    ]

    val_parts = [
        scaler.transform(part)
        for part in val_parts
        if len(part) > 0
    ]

    test_parts = [
        scaler.transform(part)
        for part in test_parts
        if len(part) > 0
    ]

    return train_parts, val_parts, test_parts


def create_sequences(
    data_parts,
    model_features,
    input_seq_length=24,
    output_seq_length=6,
    target=TARGET
):

    X, y = [], []

    target_index = model_features.index(target)

    for data in data_parts:

        # if len(data) <= (
        #     input_seq_length + output_seq_length
        # ):
        #     continue
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

        print("\nOriginal data shape before sequencing:", data.shape)

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

            # Multi-step output sequence
            y.append(
                data[
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
    batch_size=32,
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
    input_seq_length=24,
    output_seq_length=6,
    batch_size=32,
    target=TARGET
):

    df = preprocess_data(df)

    station_ids = sorted(
        df[STATION_COLUMN].unique().tolist()
    )

    # Split BEFORE filling/scaling
    train_parts, val_parts, test_parts = split_data_by_station(df)

    train_parts = fill_missing_parts(train_parts, target)
    val_parts = fill_missing_parts(val_parts, target)
    test_parts = fill_missing_parts(test_parts, target)

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
        output_seq_length,
        target
    )

    X_val, y_val = create_sequences(
        val_parts,
        model_features,
        input_seq_length,
        output_seq_length,
        target
    )

    X_test, y_test = create_sequences(
        test_parts,
        model_features,
        input_seq_length,
        output_seq_length,
        target
    )

    print("\n========== FINAL DATA SHAPES ==========")

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

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

    return train_loader, val_loader, test_loader, X_train.shape[2]


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size,
        output_seq_length=6,
        hidden_size=64,
        num_layers=2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
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

        output, _ = self.lstm(x)

        # Use last timestep output
        output = output[:, -1, :]

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
    epochs=10
):

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    train_losses, val_losses = [], []

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

    print("\n========== MODEL EVALUATION ==========")

    print("Overall MSE:", mse)
    print("Overall MAE:", mae)
    print("Overall RMSE:", rmse)

    #print("\nPer-step forecast metrics:")
    # for step in range(actuals.shape[1]):
    #     step_mse = mean_squared_error(actuals[:, step], predictions[:, step])
    #     step_mae = mean_absolute_error(actuals[:, step], predictions[:, step])
    #     step_rmse = np.sqrt(step_mse)
    #     print(
    #         f"Step {step + 1}: "
    #         f"MSE={step_mse:.6f}, "
    #         f"MAE={step_mae:.6f}, "
    #         f"RMSE={step_rmse:.6f}"
    #     )

    return predictions, actuals, mse, mae, rmse


def plot_loss_curves(
    train_losses,
    val_losses,
    target=TARGET,
    output_seq_length=None
):

    plt.figure(figsize=(8, 4))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    target_name = TARGET_DISPLAY_NAMES.get(target, target)
    title = f"Train vs Validation Loss - {target_name}"
    if output_seq_length is not None:
        title += f" (Output Length {output_seq_length})"

    plt.title(title)

    plt.legend()

    plt.show()


def plot_predictions(
    actuals,
    predictions,
    forecast_step=1,
    target=TARGET,
    max_points=200
):

    step_index = forecast_step - 1
    if step_index < 0 or step_index >= actuals.shape[1]:
        raise ValueError(
            f"forecast_step must be between 1 and {actuals.shape[1]}"
        )

    target_name = TARGET_DISPLAY_NAMES.get(target, target)

    plt.figure(figsize=(10, 4))

    # Plot one forecast horizon only. Flattening multi-step outputs mixes
    # overlapping windows and creates a misleading zig-zag plot.
    plt.plot(
        actuals[:max_points, step_index],
        label="Actual"
    )

    plt.plot(
        predictions[:max_points, step_index],
        label="Predicted"
    )

    plt.legend()

    plt.title(
        f"Actual vs Predicted {target_name} for LSTM "
        f"(Forecast Horizon {forecast_step})"
    )

    plt.xlabel("Time Steps")
    plt.ylabel(f"Scaled {target_name}")

    plt.show()


def get_forecast_steps_to_plot(output_seq_length):

    forecast_steps = [1]

    if output_seq_length != 1:
        forecast_steps.append(output_seq_length)

    return forecast_steps


def run_lstm_model(
    df,
    epochs=10,
    input_seq_length=24,
    output_seq_length=6,
    show_prediction_plot=True,
    target=TARGET
):

    if target not in BASE_FEATURES:
        raise ValueError(f"Unknown target feature: {target}")

    target_name = TARGET_DISPLAY_NAMES.get(target, target)

    print(
        "\n=================================================="
    )
    print(
        f"RUNNING LSTM EXPERIMENT: Target = {target_name} "
        f"({target}) | Output Length = {output_seq_length}"
    )
    print(
        "=================================================="
    )

    train_loader, val_loader, test_loader, input_size = (
        prepare_lstm_data(
            df,
            input_seq_length,
            output_seq_length,
            target=target
        )
    )

    model = LSTMModel(
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
        val_losses,
        target=target,
        output_seq_length=output_seq_length
    )

    if show_prediction_plot:
        for forecast_step in get_forecast_steps_to_plot(output_seq_length):
            plot_predictions(
                actuals,
                predictions,
                forecast_step=forecast_step,
                target=target
            )

    return {
        "model": model,
        "target": target,
        "output_seq_length": output_seq_length,
        "predictions": predictions,
        "actuals": actuals,
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def run_lstm_experiments(
    df,
    epochs=10,
    input_seq_length=24,
    output_seq_length=6,
    target_features=None,
    show_prediction_plot=True
):

    if target_features is None:
        target_features = TARGET_FEATURES

    results = {}

    for target in target_features:
        results[target] = run_lstm_model(
            df,
            epochs=epochs,
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length,
            target=target,
            show_prediction_plot=show_prediction_plot
        )

    return results
