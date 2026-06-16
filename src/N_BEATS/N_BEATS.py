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
from src.N_BEATS.explainability import run_explainability


HISTORY_DAYS = 5
PREDICTION_HOURS = 2
RESAMPLE_FREQ = "15min"

STEPS_PER_HOUR = int(pd.Timedelta(hours=1) / pd.Timedelta(RESAMPLE_FREQ))

input_size = HISTORY_DAYS * 24 * STEPS_PER_HOUR  # 480
horizon = PREDICTION_HOURS * STEPS_PER_HOUR      # 8

learning_rate = 0.0001
dropout = 0.25
hidden_dim = 64
num_blocks = 2
num_layers = 2
batch_size = 128
num_epochs = 50
weight_decay = 1e-4

def train_val_test_spliting(feature_df):
    print("\n========== Train, Validation and Train ==========")
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

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

    print("Scaled X min/max:", train_df_scaled[feature_cols].min().min(), train_df_scaled[feature_cols].max().max())
    print("Scaled y min/max:", train_df_scaled[target_col].min(), train_df_scaled[target_col].max())

    return target_col, feature_cols, x_scaler, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled


def create_windows(data, feature_cols, target_col="target_co2_15min"):
    X_windows = []
    y_windows = []

    data = data.sort_values(["station_id", "timestamp"])
    forecast_offset = horizon  # 8 rows = 2 hours

    for station_id, station_data in data.groupby("station_id"):
        X = station_data[feature_cols].values.astype("float32")
        y = station_data[target_col].values.astype("float32")

        for i in range(input_size, len(station_data) - forecast_offset + 1):
            X_windows.append(X[i - input_size:i])
            y_windows.append(y[i + forecast_offset - 1])

    return np.array(X_windows, dtype="float32"), np.array(y_windows, dtype="float32")


def window_creation(train_df_scaled, val_df_scaled, test_df_scaled, feature_cols, target_col):
    print("\n========== Creating Window ==========")
    X_train, y_train = create_windows(train_df_scaled, feature_cols, target_col)
    X_val, y_val = create_windows(val_df_scaled, feature_cols, target_col)
    X_test, y_test = create_windows(test_df_scaled, feature_cols, target_col)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


def tensor_and_dataloader(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n========== Tensor and DataLoader ==========")
    batch_size = 128

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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
    def __init__(self, input_dim, hidden_dim, theta_dim, num_layers=2, dropout=0.25):
        super(NBeatsBlock, self).__init__()

        layers = []

        for layer_idx in range(num_layers):
            in_dim = input_dim if layer_idx == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())

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
                dropout=dropout,
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
    horizon = 1

    model = NBeats(
        input_size=input_size,
        num_features=num_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_layers=num_layers,
        horizon=horizon,
        dropout=dropout,
    )

    print(model)
    return model


def loss_function(model):
    print("\n========== Loss Function ==========")
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    print("Loss function:", criterion)
    print("Optimizer:", optimizer)

    return criterion, optimizer


def print_batch_size_and_minmax_with_batch_loss(train_loader, model, criterion):
    print("\n========== Printing values before Trainging ==========")
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

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_state = None

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
    mae_scaled = mean_absolute_error(test_actuals, test_predictions)
    mse_scaled = mean_squared_error(test_actuals, test_predictions)
    rmse_scaled = np.sqrt(mse_scaled)
    r2_scaled = r2_score(test_actuals, test_predictions)

    print("Metrics WITHOUT inverse scaling")
    print("MAE:", mae_scaled)
    print("MSE:", mse_scaled)
    print("RMSE:", rmse_scaled)
    print("R2:", r2_scaled)

    return mae_scaled, mse_scaled, rmse_scaled, r2_scaled


def inverse_scale(y_scaler, test_predictions, test_actuals):
    print("\n========== Inverse Scaling ==========")
    test_predictions_original = y_scaler.inverse_transform(test_predictions)
    test_actuals_original = y_scaler.inverse_transform(test_actuals)

    print("Predictions shape:", test_predictions_original.shape)
    print("Actuals shape:", test_actuals_original.shape)

    return test_predictions_original, test_actuals_original


def matrix_value_with_inverse_scaling(test_actuals_original, test_predictions_original):
    print("\n========== Matrix Value After Inverse Scaling ==========")
    mae = mean_absolute_error(test_actuals_original, test_predictions_original)
    mse = mean_squared_error(test_actuals_original, test_predictions_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_actuals_original, test_predictions_original)

    print("Metrics WITH inverse scaling")
    print("Test MAE:", mae)
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test R2:", r2)

    return mae, mse, rmse, r2


# Uncertainity Tecunique
def conformal_uncertainty_quantification(
    model,
    val_loader,
    y_scaler,
    test_predictions_original,
    test_actuals_original,
    alpha=0.10,
):
    print("\n========== Uncertainty Quantification: Split Conformal Prediction ==========")

    model.eval()

    val_predictions = []
    val_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)

            val_predictions.append(y_pred.cpu().numpy())
            val_actuals.append(y_batch.cpu().numpy())

    val_predictions = np.vstack(val_predictions)
    val_actuals = np.vstack(val_actuals)

    val_predictions_original = y_scaler.inverse_transform(val_predictions)
    val_actuals_original = y_scaler.inverse_transform(val_actuals)

    calibration_errors = np.abs(val_actuals_original - val_predictions_original)

    q_hat = np.quantile(calibration_errors, 1 - alpha)

    test_lower = test_predictions_original - q_hat
    test_upper = test_predictions_original + q_hat

    coverage = np.mean(
        (test_actuals_original >= test_lower)
        & (test_actuals_original <= test_upper)
    )

    mean_interval_width = np.mean(test_upper - test_lower)

    print("UQ Technique Used: Split Conformal Prediction")
    print("Confidence Level:", int((1 - alpha) * 100), "%")
    print("Conformal calibration quantile q_hat:", q_hat)
    print("Prediction interval coverage:", coverage)
    print("Mean prediction interval width:", mean_interval_width)

    n_plot = min(300, len(test_predictions_original))

    plt.figure(figsize=(14, 5))
    plt.plot(
        test_actuals_original[:n_plot],
        label="Actual CO2",
        color="black",
        linewidth=1.5,
    )
    plt.plot(
        test_predictions_original[:n_plot],
        label="N-BEATS Prediction",
        color="blue",
        linewidth=1.5,
    )
    plt.fill_between(
        np.arange(n_plot),
        test_lower[:n_plot, 0],
        test_upper[:n_plot, 0],
        color="blue",
        alpha=0.2,
        label="90% Conformal Prediction Interval",
    )

    plt.xlabel("Test Sample")
    plt.ylabel("CO2")
    plt.title("N-BEATS Forecast with 90% Conformal Prediction Interval")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return test_lower, test_upper, q_hat, coverage, mean_interval_width


def graph_for_actual_vs_predection(test_actuals_original, test_predictions_original):
    print("\n========== Actual vs Prediction Graph ==========")
    plt.figure(figsize=(14, 5))
    plt.plot(test_actuals_original[:100], label="Actual CO2")
    plt.plot(test_predictions_original[:100], label="Predicted CO2")
    plt.xlabel("Sample")
    plt.ylabel("CO2")
    plt.title("Actual vs Predicted CO2")
    plt.legend()
    plt.grid(True)
    plt.show()


def graph_for_scatter_actual_vs_prediction(test_actuals_original, test_predictions_original):
    print("\n========== Scatter Plot ==========")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(
        test_actuals_original,
        test_actuals_original,
        color="blue",
        alpha=0.5,
    )
    ax[0].set_xlabel("Actual CO2")
    ax[0].set_ylabel("Actual CO2")
    ax[0].set_title("Actual CO2 Scatter")
    ax[0].grid(True)

    ax[1].scatter(
        test_actuals_original,
        test_predictions_original,
        color="orange",
        alpha=0.2,
    )
    ax[1].set_xlabel("Actual CO2")
    ax[1].set_ylabel("Predicted CO2")
    ax[1].set_title("Predicted CO2 Scatter")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def model_pipeline(feature_df):
    train_df, val_df, test_df = train_val_test_spliting(feature_df)

    target_col, feature_cols, x_scaler, y_scaler, train_df_scaled, val_df_scaled, test_df_scaled = min_max_scaler(
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

    print_batch_size_and_minmax_with_batch_loss(train_loader, model, criterion)

    model, train_losses, val_losses = epochs(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
    )

    epochs_graph(train_losses, val_losses)

    test_predictions, test_actuals = model_evalution(model, test_loader)

    matrix_value_without_inverse_scaling(test_actuals, test_predictions)

    test_predictions_original, test_actuals_original = inverse_scale(
        y_scaler,
        test_predictions,
        test_actuals,
    )

    matrix_value_with_inverse_scaling(
        test_actuals_original,
        test_predictions_original,
    )

    test_lower, test_upper, q_hat, coverage, mean_interval_width = conformal_uncertainty_quantification(
        model=model,
        val_loader=val_loader,
        y_scaler=y_scaler,
        test_predictions_original=test_predictions_original,
        test_actuals_original=test_actuals_original,
        alpha=0.10,
    )

    graph_for_actual_vs_predection(
        test_actuals_original,
        test_predictions_original,
    )

    graph_for_scatter_actual_vs_prediction(
        test_actuals_original,
        test_predictions_original,
    )

    run_explainability(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        feature_cols=feature_cols,
    )

    return model