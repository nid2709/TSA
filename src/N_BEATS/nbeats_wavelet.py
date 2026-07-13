import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from kymatio.scattering1d.frontend.numpy_frontend import ScatteringNumPy1D as Scattering1D
except ImportError as e:
    print("Kymatio 1D import error:", e)
    Scattering1D = None


HISTORY_DAYS = 5
PREDICTION_HOURS = 2
RESAMPLE_FREQ = "15min"

STEPS_PER_HOUR = int(pd.Timedelta(hours=1) / pd.Timedelta(RESAMPLE_FREQ))
input_size = HISTORY_DAYS * 24 * STEPS_PER_HOUR
horizon = PREDICTION_HOURS * STEPS_PER_HOUR

learning_rate = 0.0001
dropout = 0.25
hidden_dim = 64
num_blocks = 2
num_layers = 2
batch_size = 128
num_epochs = 50
weight_decay = 1e-4

# WaveletTransformation
USE_SCATTERING = True
SCATTERING_J = 5
SCATTERING_Q = 8
N_SCATTERING_FEATURES = 24
SCATTERING_SIGNAL_COL = "scd41_co2"


def train_val_test_spliting(feature_df):
    print("\n========== Train, Validation and Test Split ==========")

    train_ratio = 0.70
    val_ratio = 0.15

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

    for station_id, station_data in feature_df.groupby("station_id", sort=True):
        station_train, station_val, station_test = split_station_data(station_data)
        train_parts.append(station_train)
        val_parts.append(station_val)
        test_parts.append(station_test)

    train_df = pd.concat(train_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    total_rows = len(feature_df)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("Train percentage:", len(train_df) / total_rows * 100)
    print("Val percentage:", len(val_df) / total_rows * 100)
    print("Test percentage:", len(test_df) / total_rows * 100)

    split_counts = pd.concat(
        [
            train_df.groupby("station_id").size().rename("train"),
            val_df.groupby("station_id").size().rename("val"),
            test_df.groupby("station_id").size().rename("test"),
        ],
        axis=1,
    ).fillna(0).astype(int)

    print("\nRows per station:")
    print(split_counts)

    return train_df, val_df, test_df


def build_scattering_transform():
    if Scattering1D is None:
        raise ImportError(
            "Kymatio is required for scattering wavelet features. "
            "Install it with: pip install kymatio"
        )

    return Scattering1D(
        J=SCATTERING_J,
        shape=input_size,
        Q=SCATTERING_Q,
    )


def compute_static_scattering_features(signal_window, scattering_transform):
    signal_window = np.asarray(signal_window, dtype=np.float32)

    scattering_coefficients = scattering_transform(signal_window)
    scattering_coefficients = np.asarray(scattering_coefficients)

    if scattering_coefficients.ndim == 2:
        static_vector = scattering_coefficients.mean(axis=1)
    else:
        static_vector = scattering_coefficients.reshape(-1)

    static_vector = static_vector[:N_SCATTERING_FEATURES]

    if len(static_vector) < N_SCATTERING_FEATURES:
        static_vector = np.pad(
            static_vector,
            (0, N_SCATTERING_FEATURES - len(static_vector)),
            mode="constant",
        )

    return static_vector.astype(np.float32)


def min_max_scaler(feature_df, train_df, val_df, test_df):
    print("\n========== Min Max Scaling ==========")

    target_col = "target_co2_15min"

    feature_cols = [
        col for col in feature_df.columns
        if col not in [target_col, "timestamp"]
    ]

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    train_df_scaled = train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()

    train_df_scaled[feature_cols] = x_scaler.fit_transform(train_df[feature_cols])
    val_df_scaled[feature_cols] = x_scaler.transform(val_df[feature_cols])
    test_df_scaled[feature_cols] = x_scaler.transform(test_df[feature_cols])

    train_df_scaled[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])
    val_df_scaled[[target_col]] = y_scaler.transform(val_df[[target_col]])
    test_df_scaled[[target_col]] = y_scaler.transform(test_df[[target_col]])

    print(
        "Scaled X min/max:",
        train_df_scaled[feature_cols].min().min(),
        train_df_scaled[feature_cols].max().max(),
    )
    print(
        "Scaled y min/max:",
        train_df_scaled[target_col].min(),
        train_df_scaled[target_col].max(),
    )

    return (
        target_col,
        feature_cols,
        x_scaler,
        y_scaler,
        train_df_scaled,
        val_df_scaled,
        test_df_scaled,
    )


def create_windows(
    data,
    feature_cols,
    target_col="target_co2_15min",
    scattering_transform=None,
):
    X_windows = []
    static_windows = []
    y_windows = []

    data = data.sort_values(["station_id", "timestamp"])

    signal_idx = feature_cols.index(SCATTERING_SIGNAL_COL)

    for station_id, station_data in data.groupby("station_id"):
        X = station_data[feature_cols].values.astype("float32")
        y = station_data[target_col].values.astype("float32")

        for i in range(input_size, len(station_data) - horizon + 2):
            input_window = X[i - input_size:i]

            X_windows.append(input_window)
            y_windows.append(y[i - 1:i - 1 + horizon])

            if USE_SCATTERING:
                if scattering_transform is None:
                    raise ValueError(
                        "scattering_transform must be provided when "
                        "USE_SCATTERING=True"
                    )

                signal_window = input_window[:, signal_idx]

                static_features = compute_static_scattering_features(
                    signal_window,
                    scattering_transform,
                )
            else:
                static_features = np.zeros(0, dtype=np.float32)

            static_windows.append(static_features)

    return (
        np.array(X_windows, dtype="float32"),
        np.array(static_windows, dtype="float32"),
        np.array(y_windows, dtype="float32"),
    )


def window_creation(train_df_scaled, val_df_scaled, test_df_scaled, feature_cols, target_col):
    print("\n========== Creating Windows ==========")
    print("Use scattering features:", USE_SCATTERING)

    scattering_transform = build_scattering_transform() if USE_SCATTERING else None

    X_train, S_train, y_train = create_windows(
        train_df_scaled,
        feature_cols,
        target_col,
        scattering_transform,
    )

    X_val, S_val, y_val = create_windows(
        val_df_scaled,
        feature_cols,
        target_col,
        scattering_transform,
    )

    X_test, S_test, y_test = create_windows(
        test_df_scaled,
        feature_cols,
        target_col,
        scattering_transform,
    )

    print("X_train:", X_train.shape, "S_train:", S_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "S_val:", S_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "S_test:", S_test.shape, "y_test:", y_test.shape)

    return X_train, S_train, y_train, X_val, S_val, y_val, X_test, S_test, y_test


def tensor_and_dataloader(
    X_train,
    S_train,
    y_train,
    X_val,
    S_val,
    y_val,
    X_test,
    S_test,
    y_test,
):
    print("\n========== Tensor and DataLoader ==========")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    S_train_tensor = torch.tensor(S_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    S_val_tensor = torch.tensor(S_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    S_test_tensor = torch.tensor(S_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, S_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, S_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, S_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
    print("Device:", device)

    return train_loader, val_loader, test_loader, device


class NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        theta_dim,
        num_layers=2,
        dropout=0.25,
        use_layer_norm=False,
        activation_cls=nn.ReLU,
    ):
        super(NBeatsBlock, self).__init__()

        layers = []

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(activation_cls())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.fc = nn.Sequential(*layers)
        self.backcast_layer = nn.Linear(hidden_dim, input_dim)
        self.forecast_layer = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x):
        h = self.fc(x)

        backcast = self.backcast_layer(h)
        forecast = self.forecast_layer(h)

        return backcast, forecast


class NBeats(nn.Module):
    def __init__(
        self,
        input_size,
        num_features,
        hidden_dim=64,
        num_blocks=2,
        num_layers=2,
        horizon=8,
        dropout=0.25,
        static_dim=0,
        block_cls=NBeatsBlock,
        block_kwargs=None,
    ):
        super(NBeats, self).__init__()

        self.input_dim = input_size * num_features + static_dim
        self.static_dim = static_dim
        self.horizon = horizon
        block_kwargs = block_kwargs or {}

        self.blocks = nn.ModuleList(
            [
                block_cls(
                    input_dim=self.input_dim,
                    hidden_dim=hidden_dim,
                    theta_dim=horizon,
                    num_layers=num_layers,
                    dropout=dropout,
                    **block_kwargs,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, static_features=None):
        x = x.reshape(x.size(0), -1)

        if static_features is not None and static_features.numel() > 0:
            x = torch.cat([x, static_features], dim=1)

        residual = x

        forecast = torch.zeros(
            x.size(0),
            self.horizon,
            device=x.device,
        )

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


def build_model(model_class, X_train, S_train, device):
    print("\n========== Model Building ==========")

    num_features = X_train.shape[2]
    static_dim = S_train.shape[1] if USE_SCATTERING else 0

    model = model_class(
        input_size=input_size,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
        static_dim=static_dim,
    )

    model = model.to(device)

    print(model)

    return model


def train_model(model_name, model, train_loader, val_loader, device):
    print(f"\n========== Training {model_name} ==========")

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for X_batch, S_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            y_pred = model(X_batch, S_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, S_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                S_batch = S_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch, S_batch)
                loss = criterion(y_pred, y_batch)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"{model_name} Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)

    print(f"{model_name} Best Val Loss:", best_val_loss)

    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


def graph_training_losses(result, model_name="N-BEATS"):
    print("\n========== Training and Validation Loss Graph ==========")

    plt.figure(figsize=(10, 5))
    plt.plot(result["train_losses"], label=f"{model_name} Train")
    plt.plot(result["val_losses"], linestyle="--", label=f"{model_name} Validation")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def predict_model(model, test_loader, device):
    model.eval()

    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, S_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            S_batch = S_batch.to(device)

            y_pred = model(X_batch, S_batch)

            test_predictions.append(y_pred.cpu().numpy())
            test_actuals.append(y_batch.numpy())

    test_predictions = np.vstack(test_predictions)
    test_actuals = np.vstack(test_actuals)

    return test_predictions, test_actuals


def calculate_metrics(actuals, predictions):
    mae = mean_absolute_error(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, predictions)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def inverse_scale_2d(values, scaler):
    original_shape = values.shape

    values_reshaped = values.reshape(-1, 1)
    values_original = scaler.inverse_transform(values_reshaped)

    return values_original.reshape(original_shape)


def evaluate_model(result, test_loader, device, y_scaler):
    print("\n========== Model Evaluation ==========")

    test_predictions_scaled, test_actuals_scaled = predict_model(
        result["model"],
        test_loader,
        device,
    )

    metrics_scaled = calculate_metrics(
        test_actuals_scaled,
        test_predictions_scaled,
    )

    test_predictions_original = inverse_scale_2d(
        test_predictions_scaled,
        y_scaler,
    )

    test_actuals_original = inverse_scale_2d(
        test_actuals_scaled,
        y_scaler,
    )

    metrics_original = calculate_metrics(
        test_actuals_original,
        test_predictions_original,
    )

    result["test_predictions_scaled"] = test_predictions_scaled
    result["test_actuals_scaled"] = test_actuals_scaled
    result["test_predictions_original"] = test_predictions_original
    result["test_actuals_original"] = test_actuals_original

    print("Metrics WITHOUT inverse scaling")
    print(pd.Series(metrics_scaled))

    print("\nMetrics WITH inverse scaling in original CO2 scale")
    print(pd.Series(metrics_original))

    return result, metrics_scaled, metrics_original


def graph_prediction(result, model_name="N-BEATS"):
    print(f"\n========== {model_name} Prediction Graph ==========")

    plt.figure(figsize=(14, 5))

    plot_points = 100
    forecast_step = 0

    actuals = result["test_actuals_original"]
    predictions = result["test_predictions_original"]

    plt.plot(
        actuals[:plot_points, forecast_step],
        label="Actual CO2",
    )

    plt.plot(
        predictions[:plot_points, forecast_step],
        label=f"{model_name} Predicted CO2",
    )

    plt.xlabel("Test sample")
    plt.ylabel("CO2")
    plt.title(f"{model_name} CO2 Prediction")
    plt.legend()
    plt.grid(True)
    plt.show()


def model_pipeline_nbeats_wavelet(feature_df):
    train_df, val_df, test_df = train_val_test_spliting(feature_df)

    (
        target_col,
        feature_cols,
        x_scaler,
        y_scaler,
        train_df_scaled,
        val_df_scaled,
        test_df_scaled,
    ) = min_max_scaler(
        feature_df,
        train_df,
        val_df,
        test_df,
    )

    X_train, S_train, y_train, X_val, S_val, y_val, X_test, S_test, y_test = window_creation(
        train_df_scaled,
        val_df_scaled,
        test_df_scaled,
        feature_cols,
        target_col,
    )

    train_loader, val_loader, test_loader, device = tensor_and_dataloader(
        X_train,
        S_train,
        y_train,
        X_val,
        S_val,
        y_val,
        X_test,
        S_test,
        y_test,
    )

    model = build_model(NBeats, X_train, S_train, device)

    result = train_model(
        "N-BEATS",
        model,
        train_loader,
        val_loader,
        device,
    )

    graph_training_losses(result, "N-BEATS")

    result, metrics_scaled, metrics_original = evaluate_model(
        result,
        test_loader,
        device,
        y_scaler,
    )

    graph_prediction(result, "N-BEATS")

    return {
        "result": result,
        "metrics_scaled": metrics_scaled,
        "metrics_original": metrics_original,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }