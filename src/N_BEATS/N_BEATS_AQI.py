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


INPUT_SIZE = 60
TARGET_COL = "target_ens160_aqi_15min"


def train_val_test_spliting(feature_df):
    train_ratio = 0.70
    val_ratio = 0.15

    train_parts, val_parts, test_parts = [], [], []

    for station_id, station_data in feature_df.groupby("station_id", sort=True):
        station_data = station_data.sort_values("timestamp")
        n_rows = len(station_data)

        train_end = int(n_rows * train_ratio)
        val_end = train_end + int(n_rows * val_ratio)

        train_parts.append(station_data.iloc[:train_end])
        val_parts.append(station_data.iloc[train_end:val_end])
        test_parts.append(station_data.iloc[val_end:])

    train_df = pd.concat(train_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    val_df = pd.concat(val_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    test_df = pd.concat(test_parts).sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    return train_df, val_df, test_df


def min_max_scaler(feature_df, train_df, val_df, test_df, target_col=TARGET_COL):
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

    return target_col, feature_cols, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled


def create_windows(data, feature_cols, target_col=TARGET_COL, input_size=INPUT_SIZE, horizon=1):
    X_windows, y_windows = [], []
    data = data.sort_values(["station_id", "timestamp"])

    for station_id, station_data in data.groupby("station_id"):
        X = station_data[feature_cols].values.astype("float32")
        y = station_data[target_col].values.astype("float32")

        for i in range(input_size, len(station_data) - horizon + 1):
            X_windows.append(X[i - input_size:i])
            y_windows.append(y[i:i + horizon])

    return np.array(X_windows, dtype="float32"), np.array(y_windows, dtype="float32")


def tensor_and_dataloader(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=64):
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, theta_dim, num_layers=4):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)
        self.backcast_layer = nn.Linear(hidden_dim, input_dim)
        self.forecast_layer = nn.Linear(hidden_dim, theta_dim)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_layer(h)
        forecast = self.forecast_layer(h)
        return backcast, forecast


class NBeats(nn.Module):
    def __init__(self, input_size, num_features, hidden_dim=256, num_blocks=2, num_layers=3, horizon=1):
        super().__init__()

        self.input_dim = input_size * num_features
        self.horizon = horizon

        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_dim=self.input_dim,
                hidden_dim=hidden_dim,
                theta_dim=horizon,
                num_layers=num_layers,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)

        residual = x
        forecast = torch.zeros(x.size(0), self.horizon, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


def train_model(model, train_loader, val_loader, num_epochs=30, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.6f} "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    print("Best Val Loss:", best_val_loss)

    return model, train_losses, val_losses


def evaluate_model(model, test_loader):
    model.eval()

    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            predictions.append(y_pred.numpy())
            actuals.append(y_batch.numpy())

    return np.vstack(predictions), np.vstack(actuals)


def calculate_metrics(actuals, predictions):
    results = []

    for h in range(predictions.shape[1]):
        actual_h = actuals[:, h]
        pred_h = predictions[:, h]

        mse = mean_squared_error(actual_h, pred_h)

        results.append({
            "Horizon": h + 1,
            "MAE": mean_absolute_error(actual_h, pred_h),
            "MSE": mse,
            "RMSE": np.sqrt(mse),
            "R2": r2_score(actual_h, pred_h),
        })

    metrics_df = pd.DataFrame(results)
    best_horizon = metrics_df.loc[metrics_df["MAE"].idxmin()]

    return metrics_df, best_horizon


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_actual_vs_prediction(actuals, predictions, best_horizon):
    best_h = int(best_horizon["Horizon"]) - 1

    plt.figure(figsize=(14, 5))
    plt.plot(actuals[:100, best_h], label="Actual AQI", linewidth=2)
    plt.plot(predictions[:100, best_h], label="Predicted AQI", linewidth=2)
    plt.xlabel("Sample")
    plt.ylabel("AQI")
    plt.title(f"Actual vs Predicted AQI - Horizon {best_h + 1}")
    plt.legend()
    plt.grid(True)
    plt.show()


def model_pipeline_AQI_config(feature_df, horizon=1, input_size=INPUT_SIZE, target_col=TARGET_COL):
    print(f"\n========== AQI N-BEATS Pipeline | Horizon: {horizon} ==========")

    train_df, val_df, test_df = train_val_test_spliting(feature_df)

    target_col, feature_cols, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled = min_max_scaler(
        feature_df,
        train_df,
        val_df,
        test_df,
        target_col,
    )

    X_train, y_train = create_windows(train_df_scaled, feature_cols, target_col, input_size, horizon)
    X_val, y_val = create_windows(val_df_scaled, feature_cols, target_col, input_size, horizon)
    X_test, y_test = create_windows(test_df_scaled, feature_cols, target_col, input_size, horizon)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    train_loader, val_loader, test_loader = tensor_and_dataloader(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    model = NBeats(
        input_size=input_size,
        num_features=X_train.shape[2],
        hidden_dim=256,
        num_blocks=2,
        num_layers=3,
        horizon=horizon,
    )

    print(model)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader)
    plot_losses(train_losses, val_losses)

    test_predictions, test_actuals = evaluate_model(model, test_loader)

    scaled_metrics_df, best_scaled = calculate_metrics(test_actuals, test_predictions)
    print("\nMetrics WITHOUT inverse scaling:")
    print(scaled_metrics_df)
    print("\nBest horizon WITHOUT inverse scaling:")
    print(best_scaled)

    test_predictions_original = y_scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).reshape(test_predictions.shape)

    test_actuals_original = y_scaler.inverse_transform(
        test_actuals.reshape(-1, 1)
    ).reshape(test_actuals.shape)

    original_metrics_df, best_original = calculate_metrics(
        test_actuals_original,
        test_predictions_original,
    )

    print("\nMetrics WITH inverse scaling:")
    print(original_metrics_df)
    print("\nBest horizon WITH inverse scaling:")
    print(best_original)

    plot_actual_vs_prediction(
        test_actuals_original,
        test_predictions_original,
        best_original,
    )

    return model


def model_pipeline_AQI(feature_df):
    return model_pipeline_AQI_config(feature_df, horizon=1)


def model_pipeline_AQI_hor5(feature_df):
    return model_pipeline_AQI_config(feature_df, horizon=5)


def model_pipeline_AQI_hor10(feature_df):
    return model_pipeline_AQI_config(feature_df, horizon=10)