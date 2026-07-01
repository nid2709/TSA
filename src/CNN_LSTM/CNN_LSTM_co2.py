import copy
import os
import time

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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.CNN_LSTM.CNN_LSTM_data_processing import (
    add_station_features,
    build_future_target_reference,
    clip_outliers_from_train,
    drop_short_stations_for_windowing,
    fill_missing_dataframe,
    preprocess_data,
    save_future_target_reference,
    scale_data,
    train_val_test_spliting,
)

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
    'scd41_co2',

    'hour_sin',
    'hour_cos',
    'dayofweek_sin',
    'dayofweek_cos',
    'is_weekend',
]

TARGET = 'scd41_co2'
STATION_COLUMN = 'station_id'
SEGMENT_COLUMN = '_continuous_segment_id'

DEFAULT_INPUT_SEQ_LENGTH = 192
DEFAULT_OUTPUT_SEQ_LENGTH = 6
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 15
DEFAULT_LEARNING_RATE = 0.00005
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_RESAMPLE_TIME = '15min'
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_LAYERS = 2

#Extra data/training Safety settings
DEFAULT_MAX_FILL_STEPS = 2
DEFAULT_DROP_SHORT_STATIONS = True
DEFAULT_CLIP_OUTLIERS = True # Only make this false when need to preserve all original sensor peaks
DEFAULT_OUTLIER_CLIP_FACTOR = 1.5
DEFAULT_RESTORE_BEST_MODEL = True
conv_channels = 96

DEFAULT_USE_SCATTERING = False
DEFAULT_SCATTERING_J = 4
DEFAULT_SCATTERING_Q = 8
DEFAULT_N_SCATTERING_FEATURES = 16


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


def format_elapsed_time(seconds):
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"

    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    return f"{remaining_seconds:.2f}s"


def get_cnn_lstm_results_dir(
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
    max_fill_steps=DEFAULT_MAX_FILL_STEPS,
    drop_short_stations=DEFAULT_DROP_SHORT_STATIONS,
    clip_outliers=DEFAULT_CLIP_OUTLIERS,
    restore_best_model=DEFAULT_RESTORE_BEST_MODEL,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
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
        f"HS{hidden_size}_"
        f"RS{resample_time}_"
        f"DR{dropout_rate}_"
        f"WD{weight_decay}_"
        f"NL{num_layers}_"
        f"GF{max_fill_steps}_"
        f"DSS{int(drop_short_stations)}_"
        f"CLP{int(clip_outliers)}_"
        f"RB{int(restore_best_model)}_"
        f"SWT{int(use_scattering)}_"
        f"SWJ{scattering_j if use_scattering else 0}_"
        f"SWQ{scattering_q if use_scattering else 0}_"
        f"SWF{n_scattering_features if use_scattering else 0}"
    )

    return os.path.join(project_root, folder_name)


def get_scattering_feature_names(n_scattering_features):
    return [
        f"scatter_co2_{i + 1}"
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
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_transform=None,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    X, y = [], []
    target_index = model_features.index(TARGET)

    if SEGMENT_COLUMN not in data.columns:
        raise ValueError(
            f"Missing {SEGMENT_COLUMN}. Run fill_missing_dataframe() "
            "before creating sequences."
        )

    data = data.sort_values(
        [STATION_COLUMN, SEGMENT_COLUMN, "timestamp"]
    )

    grouped_segments = data.groupby(
        [STATION_COLUMN, SEGMENT_COLUMN],
        sort=True
    )

    for (station_id, segment_id), segment_data in grouped_segments:
        values = segment_data[model_features].values

        print(
            "\nContinuous segment before sequencing:",
            f"station={station_id}, segment={segment_id}, shape={segment_data.shape}"
        )
        required_length = input_seq_length + output_seq_length

        if len(values) <= required_length:
            print("\nSkipping sequence generation:")
            print(f"Station: {station_id}")
            print(f"Segment: {segment_id}")
            print(f"Available rows: {len(values)}")
            print(f"Required minimum rows: {required_length + 1}")
            continue

        for i in range(
            len(values) - input_seq_length - output_seq_length + 1
        ):
            input_window = values[i:i + input_seq_length]

            if use_scattering:
                if scattering_transform is None:
                    raise ValueError(
                        "scattering_transform must be provided when "
                        "use_scattering=True"
                    )

                co2_window = input_window[:, target_index]
                static_scattering_vector = compute_static_scattering_features(
                    co2_window,
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
    batch_size=DEFAULT_BATCH_SIZE,
    resample_time=DEFAULT_RESAMPLE_TIME,
    max_fill_steps=DEFAULT_MAX_FILL_STEPS,
    drop_short_stations=DEFAULT_DROP_SHORT_STATIONS,
    clip_outliers=DEFAULT_CLIP_OUTLIERS,
    outlier_clip_factor=DEFAULT_OUTLIER_CLIP_FACTOR,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    df = preprocess_data(
        df,
        base_features=BASE_FEATURES,
        station_column=STATION_COLUMN,
        resample_time=resample_time
    )

    if drop_short_stations:
        df = drop_short_stations_for_windowing(
            df,
            station_column=STATION_COLUMN,
            input_seq_length=input_seq_length,
            output_seq_length=output_seq_length
        )

    station_ids = sorted(df[STATION_COLUMN].unique().tolist())

    train_df, val_df, test_df = train_val_test_spliting(
        df,
        station_column=STATION_COLUMN
    )

    print("\n========== DATA GAP CONFIGURATION ==========")
    print("Expected timestamp interval:", resample_time)
    print("Maximum feature fill steps:", max_fill_steps)
    print(
        "Maximum feature fill duration:",
        pd.Timedelta(resample_time) * max_fill_steps
    )
    print("Sequences crossing detected timestamp gaps: disabled")

    train_df = fill_missing_dataframe(
        train_df,
        base_features=BASE_FEATURES,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN,
        resample_time=resample_time,
        max_fill_steps=max_fill_steps
    )
    val_df = fill_missing_dataframe(
        val_df,
        base_features=BASE_FEATURES,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN,
        resample_time=resample_time,
        max_fill_steps=max_fill_steps
    )
    test_df = fill_missing_dataframe(
        test_df,
        base_features=BASE_FEATURES,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN,
        resample_time=resample_time,
        max_fill_steps=max_fill_steps
    )

    if clip_outliers:
        train_df, val_df, test_df = clip_outliers_from_train(
            train_df,
            val_df,
            test_df,
            BASE_FEATURES,
            clip_factor=outlier_clip_factor
        )
    else:
        print("\n========== OUTLIER CLIPPING DISABLED ==========")

    print("\n========== FUTURE TARGET REFERENCE ==========")
    print(
        "Creating ahead target columns for analysis only:",
        f"step 1 to step {output_seq_length}"
    )
    print("These columns are not used as CNN-LSTM input features.")
    future_target_reference = build_future_target_reference(
        train_df,
        val_df,
        test_df,
        output_seq_length=output_seq_length,
        target_column=TARGET,
        station_column=STATION_COLUMN,
        segment_column=SEGMENT_COLUMN
    )
    print("Future target reference shape:", future_target_reference.shape)

    train_df = add_station_features(
        train_df,
        station_ids,
        station_column=STATION_COLUMN
    )
    val_df = add_station_features(
        val_df,
        station_ids,
        station_column=STATION_COLUMN
    )
    test_df = add_station_features(
        test_df,
        station_ids,
        station_column=STATION_COLUMN
    )

    station_features = [
        f"station_{station_id}"
        for station_id in station_ids
    ]

    # Model features include station one-hot columns for Explainability plots.
    # station_id itself is still only used for splitting/window creation.
    model_features = BASE_FEATURES + station_features

    train_df, val_df, test_df, scaler = scale_data(
        train_df,
        val_df,
        test_df,
        model_features,
        target_column=TARGET
    )

    scattering_transform = None
    scattering_feature_names = []

    print("\n========== FEATURE CONFIGURATION ==========")
    print("Base dynamic feature count:", len(model_features))
    print("Use scattering features:", use_scattering)

    if use_scattering:
        print("\n========== SCATTERING WAVELET FEATURES ==========")
        print("Scattering source signal: scaled scd41_co2 input window")
        print("Scattering J:", scattering_j)
        print("Scattering Q:", scattering_q)
        print("Static scattering features:", n_scattering_features)

        scattering_transform = build_scattering_transform(
            input_seq_length=input_seq_length,
            scattering_j=scattering_j,
            scattering_q=scattering_q
        )
        scattering_feature_names = get_scattering_feature_names(
            n_scattering_features
        )
    else:
        print("Static scattering features: 0")

    X_train, y_train = create_sequences(
        train_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )
    X_val, y_val = create_sequences(
        val_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )
    X_test, y_test = create_sequences(
        test_df,
        model_features,
        input_seq_length,
        output_seq_length,
        use_scattering=use_scattering,
        scattering_transform=scattering_transform,
        n_scattering_features=n_scattering_features
    )

    model_features = model_features + scattering_feature_names

    print("\n========== FINAL DATA SHAPES ==========")
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val:", X_val.shape, "y_val:", y_val.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("Final model feature count:", len(model_features))
    if scattering_feature_names:
        print("Scattering feature names:", scattering_feature_names)

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
        model_features,
        future_target_reference
    )


class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        num_layers=DEFAULT_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT_RATE
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(conv_channels, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_seq_length)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        output = output[:, -1, :]
        return self.fc(output)


# Optional alternative loss for peak-weighted experiments.
# Currently unused because train_model uses MSELoss.
# def weighted_mse_loss(predictions, targets):
#     weights = 1.0 + 5.0 * targets
#     return torch.mean(weights * (predictions - targets) ** 2)


def print_batch_sanity_check(model, train_loader):
    print("\n========== BATCH SANITY CHECK ==========")

    model.eval()
    X_batch, y_batch = next(iter(train_loader))

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
    patience=5,
    learning_rate=DEFAULT_LEARNING_RATE,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    restore_best_model=DEFAULT_RESTORE_BEST_MODEL,
    min_delta=1e-6
):
    criterion = nn.MSELoss()

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
    predictions, actuals = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.numpy())
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

    #plt.show()
    plt.close()


def plot_scattering_wavelet_features(
    X_train,
    model_features,
    n_scattering_features,
    scattering_j,
    scattering_q,
    results_dir
):
    if n_scattering_features <= 0 or len(X_train) == 0:
        return None

    target_index = model_features.index(TARGET)
    scattering_feature_names = get_scattering_feature_names(
        n_scattering_features
    )
    scattering_indices = [
        model_features.index(feature_name)
        for feature_name in scattering_feature_names
    ]

    sample_window = X_train[0]
    co2_signal = sample_window[:, target_index]

    # Scattering features are static within a sequence, so the first timestep
    # contains the same coefficient values supplied at every timestep.
    scattering_values = sample_window[0, scattering_indices]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]}
    )

    axes[0].plot(
        np.arange(len(co2_signal)),
        co2_signal,
        color="tab:blue",
        linewidth=1.8
    )
    axes[0].set_title("Representative Input Window")
    axes[0].set_xlabel("Input timestep")
    axes[0].set_ylabel("Scaled CO2")
    axes[0].grid(alpha=0.25)

    feature_labels = [
        f"S{i + 1}"
        for i in range(n_scattering_features)
    ]
    axes[1].bar(
        feature_labels,
        scattering_values,
        color="tab:orange"
    )
    axes[1].set_title(
        f"Static Scattering Features Used by CNN-LSTM "
        f"(J={scattering_j}, Q={scattering_q})"
    )
    axes[1].set_xlabel("Selected scattering coefficient")
    axes[1].set_ylabel("Mean coefficient value")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Scattering Wavelet Transform Feature Example",
        fontsize=14
    )
    fig.tight_layout()

    main_plots_dir = os.path.join(results_dir, "main_plots")
    os.makedirs(main_plots_dir, exist_ok=True)
    save_path = os.path.join(
        main_plots_dir,
        "scattering_wavelet_features.png"
    )
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved scattering wavelet plot:", save_path)

    return save_path


def plot_predictions(
    actuals,
    predictions,
    forecast_step=1,
    max_plot_points=5000,
    results_dir=None
):
    step_index = forecast_step - 1

    if forecast_step < 1 or forecast_step > actuals.shape[1]:
        raise ValueError(f"forecast_step must be between 1 and {actuals.shape[1]}")

    x_values = np.arange(len(actuals))
    actual_values = actuals[:, step_index]
    predicted_values = predictions[:, step_index]

    if max_plot_points is not None and  len(x_values) > max_plot_points:
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

    #plt.show()
    plt.close()


def plot_actual_vs_predicted_scatter(
    actuals,
    predictions,
    max_points=5000,
    results_dir=None
):
    actual_values = actuals.flatten()
    predicted_values = predictions.flatten()

    if max_points is not None and len(actual_values) > max_points:
        actual_values = actual_values[:max_points]
        predicted_values = predicted_values[:max_points]

    min_value = min(actual_values.min(), predicted_values.min())
    max_value = max(actual_values.max(), predicted_values.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        actual_values,
        predicted_values,
        alpha=0.25,
        s=12,
        color="tab:orange"
    )
    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        color="tab:blue",
        linewidth=1.5,
        label="Perfect prediction"
    )
    ax.set_xlabel("Actual scaled CO2")
    ax.set_ylabel("Predicted scaled CO2")
    ax.set_title("Actual vs Predicted Scatter for CNN-LSTM")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if results_dir is not None:
        os.makedirs(os.path.join(results_dir, "main_plots"), exist_ok=True)
        save_path = os.path.join(
            results_dir,
            "main_plots",
            "actual_vs_predicted_scatter.png"
        )
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print("Saved scatter plot:", save_path)

    plt.close(fig)


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

    plot_actual_vs_predicted_scatter(
        actuals,
        predictions,
        results_dir=results_dir
    )


def run_cnn_lstm_model(
    df,
    epochs=DEFAULT_EPOCHS,
    input_seq_length=DEFAULT_INPUT_SEQ_LENGTH,
    output_seq_length=DEFAULT_OUTPUT_SEQ_LENGTH,
    batch_size=DEFAULT_BATCH_SIZE,
    learning_rate=DEFAULT_LEARNING_RATE,
    hidden_size=DEFAULT_HIDDEN_SIZE,
    dropout_rate=DEFAULT_DROPOUT_RATE,
    resample_time=DEFAULT_RESAMPLE_TIME,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    num_layers=DEFAULT_NUM_LAYERS,
    max_fill_steps=DEFAULT_MAX_FILL_STEPS,
    drop_short_stations=DEFAULT_DROP_SHORT_STATIONS,
    clip_outliers=DEFAULT_CLIP_OUTLIERS,
    outlier_clip_factor=DEFAULT_OUTLIER_CLIP_FACTOR,
    restore_best_model=DEFAULT_RESTORE_BEST_MODEL,
    show_prediction_plot=True,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES
):
    run_start_time = time.perf_counter()

    print("\n========== CNN-LSTM RUN CONFIGURATION ==========")
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
    print("Maximum feature fill steps:", max_fill_steps)
    print("Drop short stations:", drop_short_stations)
    print("Clip outliers:", clip_outliers)
    print("Outlier clip factor:", outlier_clip_factor)
    print("Restore best validation checkpoint:", restore_best_model)
    print("Gap-aware sequence generation:", True)
    print("Convolution channels:", conv_channels)
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
        model_features,
        future_target_reference
    ) = prepare_cnn_lstm_data(
        df,
        input_seq_length=input_seq_length,
        output_seq_length=output_seq_length,
        batch_size=batch_size,
        resample_time=resample_time,
        max_fill_steps=max_fill_steps,
        drop_short_stations=drop_short_stations,
        clip_outliers=clip_outliers,
        outlier_clip_factor=outlier_clip_factor,
        use_scattering=use_scattering,
        scattering_j=scattering_j,
        scattering_q=scattering_q,
        n_scattering_features=n_scattering_features
    )

    model = CNNLSTMModel(
        input_size=input_size,
        output_seq_length=output_seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout_rate
    )

    results_dir = get_cnn_lstm_results_dir(
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
        max_fill_steps=max_fill_steps,
        drop_short_stations=drop_short_stations,
        clip_outliers=clip_outliers,
        restore_best_model=restore_best_model,
        use_scattering=use_scattering,
        scattering_j=scattering_j,
        scattering_q=scattering_q,
        n_scattering_features=n_scattering_features
    )
    horizon_metrics_path = os.path.join(
        results_dir,
        "main_plots",
        "per_horizon_metrics.csv"
    )

    print_batch_sanity_check(
        model,
        train_loader
    )

    model, train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        restore_best_model=restore_best_model
    )
    (
        predictions,
        actuals,
        mse,
        mae,
        rmse,
        r2
    ) = evaluate_model(
        model,
        test_loader,
        results_dir=results_dir
    )

    future_target_reference_path = save_future_target_reference(
        future_target_reference,
        results_dir
    )

    plot_loss_curves(
        train_losses,
        val_losses,
        results_dir=results_dir
    )

    scattering_plot_path = None
    if use_scattering:
        scattering_plot_path = plot_scattering_wavelet_features(
            X_train=X_train,
            model_features=model_features,
            n_scattering_features=n_scattering_features,
            scattering_j=scattering_j,
            scattering_q=scattering_q,
            results_dir=results_dir
        )

    if show_prediction_plot:
        plot_forecast_comparison(
            actuals,
            predictions,
            results_dir=results_dir
        )

    run_elapsed_time = time.perf_counter() - run_start_time
    print("\n========== CNN-LSTM RUNTIME ==========")
    print("CNN-LSTM runtime seconds:", run_elapsed_time)
    print("CNN-LSTM runtime:", format_elapsed_time(run_elapsed_time))

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
        "scattering_plot_path": scattering_plot_path,
        "horizon_metrics_path": horizon_metrics_path,
        "future_target_reference_path": future_target_reference_path,
        "training_runtime_seconds": run_elapsed_time,
        "training_runtime_formatted": format_elapsed_time(run_elapsed_time),

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
        "max_fill_steps": max_fill_steps,
        "drop_short_stations": drop_short_stations,
        "clip_outliers": clip_outliers,
        "outlier_clip_factor": outlier_clip_factor,
        "restore_best_model": restore_best_model,
        "gap_aware_sequences": True,
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
