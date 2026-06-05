import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


input_size = 60
horizon = 10


def train_val_test_spliting(feature_df):
    print("\n========== Train, Validation and Test ==========")

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


def min_max_scaler(feature_df, train_df, val_df, test_df):
    print("\n========== Min Max Scaling ==========")

    target_col = "target_ens160_aqi_15min"

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

    print("Scaled X min/max:", train_df_scaled[feature_cols].min().min(), train_df_scaled[feature_cols].max().max())
    print("Scaled y min/max:", train_df_scaled[target_col].min(), train_df_scaled[target_col].max())

    return target_col, feature_cols, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled


def create_windows(data, feature_cols, target_col="target_ens160_aqi_15min"):
    X_windows = []
    y_windows = []

    data = data.sort_values(["station_id", "timestamp"])

    for station_id, station_data in data.groupby("station_id"):
        X = station_data[feature_cols].values.astype("float32")
        y = station_data[target_col].values.astype("float32")

        for i in range(input_size, len(station_data) - horizon + 1):
            X_windows.append(X[i - input_size:i])
            y_windows.append(y[i:i + horizon])

    return np.array(X_windows, dtype="float32"), np.array(y_windows, dtype="float32")


def window_creation(train_df_scaled, val_df_scaled, test_df_scaled, feature_cols, target_col):
    print("\n========== Creating Windows ==========")

    X_train, y_train = create_windows(train_df_scaled, feature_cols, target_col)
    X_val, y_val = create_windows(val_df_scaled, feature_cols, target_col)
    X_test, y_test = create_windows(test_df_scaled, feature_cols, target_col)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def tensor_and_dataloader(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n========== Tensor and DataLoader ==========")

    batch_size = 64

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    return train_loader, val_loader, test_loader


class NBeatsBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, theta_dim, num_layers=4):
        super(NBeatsBlock, self).__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

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
    def __init__(
        self,
        input_size,
        num_features,
        hidden_dim=256,
        num_blocks=2,
        num_layers=3,
        horizon=10,
    ):
        super(NBeats, self).__init__()

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


def print_model(X_train):
    print("\n========== Model Printing ==========")

    num_features = X_train.shape[2]

    model = NBeats(
        input_size=input_size,
        num_features=num_features,
        hidden_dim=256,
        num_blocks=2,
        num_layers=3,
        horizon=horizon,
    )

    print(model)

    return model


def loss_function(model):
    print("\n========== Loss Function ==========")

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5,
    )

    print("Loss function:", criterion)
    print("Optimizer:", optimizer)

    return criterion, optimizer


def print_batch_size_and_minmax_with_batch_loss(train_loader, model, criterion):
    print("\n========== Values Before Training ==========")

    X_batch, y_batch = next(iter(train_loader))

    y_pred = model(X_batch)

    print("X_batch shape:", X_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("y_pred shape:", y_pred.shape)

    print("y_batch min/max:", y_batch.min().item(), y_batch.max().item())
    print("y_pred min/max:", y_pred.min().item(), y_pred.max().item())

    loss = criterion(y_pred, y_batch)

    print("Batch loss:", loss.item())

    return loss


def epochs(model, train_loader, val_loader, optimizer, criterion):
    print("\n========== Epochs Training ==========")

    num_epochs = 30
    patience = 10

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

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


def epochs_graph(train_losses, val_losses):
    print("\n========== Train vs Validation Loss Graph ==========")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def model_evalution(model, test_loader):
    print("\n========== Model Evalution ==========")

    model.eval()

    test_predictions = []
    test_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)

            test_predictions.append(y_pred.numpy())
            test_actuals.append(y_batch.numpy())

    test_predictions = np.vstack(test_predictions)
    test_actuals = np.vstack(test_actuals)

    return test_predictions, test_actuals


def matrix_value_without_inverse_scaling(test_actuals, test_predictions):
    print("\n========== Matrix Value Before Inverse Scaling ==========")

    scaled_results = []

    for h in range(horizon):
        actual_h = test_actuals[:, h]
        pred_h = test_predictions[:, h]

        mae = mean_absolute_error(actual_h, pred_h)
        mse = mean_squared_error(actual_h, pred_h)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_h, pred_h)

        scaled_results.append({
            "Horizon": h + 1,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

    scaled_metrics_df = pd.DataFrame(scaled_results)

    print("WITHOUT inverse scaling metrics:")
    print(scaled_metrics_df)

    best_scaled = scaled_metrics_df.loc[scaled_metrics_df["MAE"].idxmin()]

    print("\nBest horizon WITHOUT inverse scaling based on lowest MAE:")
    print(best_scaled)

    return scaled_metrics_df, best_scaled


def inverse_scale(y_scaler, test_predictions, test_actuals):
    print("\n========== Inverse Scaling ==========")

    test_predictions_original = y_scaler.inverse_transform(
        test_predictions.reshape(-1, 1)
    ).reshape(test_predictions.shape)

    test_actuals_original = y_scaler.inverse_transform(
        test_actuals.reshape(-1, 1)
    ).reshape(test_actuals.shape)

    print("Predictions shape:", test_predictions_original.shape)
    print("Actuals shape:", test_actuals_original.shape)

    return test_predictions_original, test_actuals_original


def matrix_value_with_inverse_scaling(test_actuals_original, test_predictions_original):
    print("\n========== Matrix Value After Inverse Scaling ==========")

    original_results = []

    for h in range(horizon):
        actual_h = test_actuals_original[:, h]
        pred_h = test_predictions_original[:, h]

        mae = mean_absolute_error(actual_h, pred_h)
        mse = mean_squared_error(actual_h, pred_h)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_h, pred_h)

        original_results.append({
            "Horizon": h + 1,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
        })

    original_metrics_df = pd.DataFrame(original_results)

    print("WITH inverse scaling metrics:")
    print(original_metrics_df)

    best_original = original_metrics_df.loc[original_metrics_df["MAE"].idxmin()]

    print("\nBest horizon WITH inverse scaling based on lowest MAE:")
    print(best_original)

    return original_metrics_df, best_original


def graph_for_actual_vs_predection(test_actuals_original, test_predictions_original, best_original):
    print("\n========== Actual vs Prediction Graph ==========")

    best_h = int(best_original["Horizon"]) - 1

    plt.figure(figsize=(14, 5))

    plt.plot(
        test_actuals_original[:100, best_h],
        label="Actual AQI",
        linewidth=2,
    )

    plt.plot(
        test_predictions_original[:100, best_h],
        label="Predicted AQI",
        linewidth=2,
    )

    plt.xlabel("Sample")
    plt.ylabel("AQI")
    plt.title("Actual vs Predicted AQI")
    plt.legend()
    plt.grid(True)
    plt.show()


def graph_for_scatter_actual_vs_prediction(test_actuals_original, test_predictions_original, best_original):
    print("\n========== Scatter Plot ==========")

    best_h = int(best_original["Horizon"]) - 1

    actual_best = test_actuals_original[:, best_h]
    pred_best = test_predictions_original[:, best_h]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(
        actual_best,
        actual_best,
        color="blue",
        alpha=0.5,
    )
    ax[0].set_xlabel("Actual AQI")
    ax[0].set_ylabel("Actual AQI")
    ax[0].set_title("Actual AQI Scatter")
    ax[0].grid(True)

    ax[1].scatter(
        actual_best,
        pred_best,
        color="orange",
        alpha=0.2,
    )
    ax[1].set_xlabel("Actual AQI")
    ax[1].set_ylabel("Predicted AQI")
    ax[1].set_title("Predicted AQI Scatter")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def model_pipeline_AQI_hor10(feature_df):
    train_df, val_df, test_df = train_val_test_spliting(feature_df)

    target_col, feature_cols, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled = min_max_scaler(
        feature_df,
        train_df,
        val_df,
        test_df,
    )

    X_train, y_train, X_val, y_val, X_test, y_test = window_creation(
        train_df_scaled,
        val_df_scaled,
        test_df_scaled,
        feature_cols,
        target_col,
    )

    train_loader, val_loader, test_loader = tensor_and_dataloader(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    model = print_model(X_train)
    criterion, optimizer = loss_function(model)

    print_batch_size_and_minmax_with_batch_loss(
        train_loader,
        model,
        criterion,
    )

    model, train_losses, val_losses = epochs(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
    )

    epochs_graph(train_losses, val_losses)

    test_predictions, test_actuals = model_evalution(
        model,
        test_loader,
    )

    matrix_value_without_inverse_scaling(
        test_actuals,
        test_predictions,
    )

    test_predictions_original, test_actuals_original = inverse_scale(
        y_scaler,
        test_predictions,
        test_actuals,
    )

    original_metrics_df, best_original = matrix_value_with_inverse_scaling(
        test_actuals_original,
        test_predictions_original,
    )

    graph_for_actual_vs_predection(
        test_actuals_original,
        test_predictions_original,
        best_original,
    )

    graph_for_scatter_actual_vs_prediction(
        test_actuals_original,
        test_predictions_original,
        best_original,
    )

    return model