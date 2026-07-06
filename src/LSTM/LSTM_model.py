import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.LSTM.LSTM_config import (
    DEFAULT_DEVICE,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUTPUT_SEQ_LENGTH,
    DEFAULT_RESTORE_BEST_MODEL,
    DEFAULT_USE_ATTENTION,
    DEFAULT_WEIGHT_DECAY,
)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query_matrix = nn.Linear(hidden_size, hidden_size)
        self.key_matrix = nn.Linear(hidden_size, hidden_size)
        self.value_matrix = nn.Linear(hidden_size, hidden_size)
        self.scale = np.sqrt(hidden_size)

    def forward(self, lstm_outputs, return_attention=False):
        queries = self.query_matrix(lstm_outputs)
        keys = self.key_matrix(lstm_outputs)
        values = self.value_matrix(lstm_outputs)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights, values)

        if return_attention:
            return context, attention_weights

        return context


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT_RATE,
        use_attention=DEFAULT_USE_ATTENTION
    ):
        super().__init__()
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.attention = (
            TemporalAttention(hidden_size)
            if use_attention
            else None
        )
        self.fc = nn.Linear(hidden_size, output_seq_length)

    def forward(self, x):
        predictions, _ = self.forward_with_attention(x)
        return predictions

    def forward_with_attention(self, x):
        output, _ = self.lstm(x)

        attention_weights = None
        if self.use_attention:
            context, attention_weights = self.attention(
                output,
                return_attention=True
            )
            output = context[:, -1, :]
        else:
            output = output[:, -1, :]

        output = self.dropout(output)

        return self.fc(output), attention_weights


def get_training_device(preferred_device=DEFAULT_DEVICE):
    if preferred_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")

    if preferred_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if preferred_device == "mps":
        print("MPS requested but not available. Falling back to CPU.")
    elif preferred_device == "cuda":
        print("CUDA requested but not available. Falling back to CPU.")

    return torch.device("cpu")


def print_batch_sanity_check(model, train_loader):
    print("\n========== BATCH SANITY CHECK ==========")

    model.eval()
    device = next(model.parameters()).device
    X_batch, y_batch = next(iter(train_loader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        y_pred = model(X_batch)

    batch_loss = nn.MSELoss()(y_pred, y_batch)

    print("X_batch shape:", X_batch.shape)
    print("y_batch shape:", y_batch.shape)
    print("y_pred shape:", y_pred.shape)
    print("X_batch min/max:", X_batch.min().item(), X_batch.max().item())
    print("y_batch min/max:", y_batch.min().item(), y_batch.max().item())
    print("y_pred min/max:", y_pred.min().item(), y_pred.max().item())
    print("Batch MSE loss:", batch_loss.item())


def evaluate_loss(model, loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=DEFAULT_EPOCHS,
    patience=5,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    restore_best_model=DEFAULT_RESTORE_BEST_MODEL,
    min_delta=1e-6
):
    device = get_training_device()
    model.to(device)
    print("Training device:", device)

    criterion = nn.MSELoss()
    print("\nUsing standard MSE loss.")

    optimizer = torch.optim.AdamW(
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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
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

        # Early stopping disabled: keep this block commented to run all epochs.
        # if epochs_without_improvement >= patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

    if restore_best_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Restored best validation checkpoint after fixed-epoch training.")

    print("Best Val Loss:", best_val_loss)

    return model, train_losses, val_losses


def calculate_metrics(actuals, predictions):
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals.flatten(), predictions.flatten())

    return mse, mae, rmse, r2


def calculate_horizon_metrics(actuals, predictions):
    horizon_metrics = []

    for step_index in range(actuals.shape[1]):
        forecast_step = step_index + 1
        step_actuals = actuals[:, step_index]
        step_predictions = predictions[:, step_index]
        step_mse = mean_squared_error(step_actuals, step_predictions)
        step_mae = mean_absolute_error(step_actuals, step_predictions)
        step_rmse = np.sqrt(step_mse)
        step_r2 = r2_score(step_actuals, step_predictions)

        horizon_metrics.append({
            "forecast_step": forecast_step,
            "mse": step_mse,
            "mae": step_mae,
            "rmse": step_rmse,
            "r2": step_r2,
        })

    return pd.DataFrame(horizon_metrics)


def save_horizon_metrics(horizon_metrics, results_dir):
    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)
    metrics_path = os.path.join(main_plots_dir, "per_horizon_metrics.csv")
    horizon_metrics.to_csv(metrics_path, index=False)
    return metrics_path


def plot_horizon_error_analysis(horizon_metrics, results_dir):
    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True
    )

    axes[0].plot(
        horizon_metrics["forecast_step"],
        horizon_metrics["mae"],
        marker="o",
        linewidth=1.8,
        label="MAE"
    )
    axes[0].plot(
        horizon_metrics["forecast_step"],
        horizon_metrics["rmse"],
        marker="o",
        linewidth=1.8,
        label="RMSE"
    )
    axes[0].set_ylabel("Error")
    axes[0].set_title("Forecast Error by Horizon")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        horizon_metrics["forecast_step"],
        horizon_metrics["r2"],
        marker="o",
        color="tab:green",
        linewidth=1.8,
        label="R2"
    )
    axes[1].set_xlabel("Forecast step")
    axes[1].set_ylabel("R2 Score")
    axes[1].set_title("Forecast Skill by Horizon")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()

    save_path = os.path.join(
        main_plots_dir,
        "horizon_error_analysis.png"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return save_path


def evaluate_model(model, test_loader, results_dir=None):
    model.eval()
    device = next(model.parameters()).device
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse, mae, rmse, r2 = calculate_metrics(actuals, predictions)

    print("\n========== MODEL EVALUATION ON SCALED VALUES ==========")
    print("Overall MSE:", mse)
    print("Overall MAE:", mae)
    print("Overall RMSE:", rmse)
    print("Overall R2 Score:", r2)

    horizon_metrics = calculate_horizon_metrics(actuals, predictions)

    if results_dir is not None:
        metrics_path = save_horizon_metrics(horizon_metrics, results_dir)
        print("Per-horizon metrics CSV:", metrics_path)
        horizon_plot_path = plot_horizon_error_analysis(
            horizon_metrics,
            results_dir
        )
        print("Horizon error analysis plot:", horizon_plot_path)

    forecast_steps = sorted(
        set([1, max(1, actuals.shape[1] // 2), actuals.shape[1]])
    )

    print("\n========== PER-HORIZON EVALUATION ==========")
    for forecast_step in forecast_steps:
        step_metrics = horizon_metrics.loc[
            horizon_metrics["forecast_step"] == forecast_step
        ].iloc[0]

        print(
            f"Step {forecast_step} -> "
            f"MSE: {step_metrics['mse']:.6f}, "
            f"MAE: {step_metrics['mae']:.6f}, "
            f"RMSE: {step_metrics['rmse']:.6f}, "
            f"R2: {step_metrics['r2']:.6f}"
        )

    return (
        predictions,
        actuals,
        mse,
        mae,
        rmse,
        r2
    )
