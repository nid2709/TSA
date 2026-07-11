import os
import time

from src.CNN_LSTM.CNN_LSTM_config import (
    BASE_FEATURES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIP_OUTLIERS,
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_DROP_SHORT_STATIONS,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_INPUT_SEQ_LENGTH,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_FILL_STEPS,
    DEFAULT_N_SCATTERING_FEATURES,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUTLIER_CLIP_FACTOR,
    DEFAULT_OUTPUT_SEQ_LENGTH,
    DEFAULT_RESAMPLE_TIME,
    DEFAULT_RESTORE_BEST_MODEL,
    DEFAULT_SCATTERING_J,
    DEFAULT_SCATTERING_Q,
    DEFAULT_USE_ATTENTION,
    DEFAULT_USE_GAP_AWARE_SEGMENTS,
    DEFAULT_USE_STATION_ONE_HOT,
    DEFAULT_USE_SCATTERING,
    DEFAULT_WEIGHT_DECAY,
    SEGMENT_COLUMN,
    STATION_COLUMN,
    TARGET,
    format_elapsed_time,
    get_cnn_lstm_results_dir,
    get_target_label,
)
from src.CNN_LSTM.CNN_LSTM_model import (
    CNNLSTMModel,
    TemporalAttention,
    calculate_horizon_metrics,
    calculate_metrics,
    evaluate_loss,
    evaluate_model,
    plot_horizon_error_analysis,
    print_batch_sanity_check,
    save_horizon_metrics,
    train_model,
)
from src.CNN_LSTM.CNN_LSTM_dataset import prepare_cnn_lstm_data
from src.CNN_LSTM.CNN_LSTM_plots import (
    plot_attention_weights,
    plot_forecast_comparison,
    plot_loss_curves,
    plot_scattering_wavelet_features,
)
from src.CNN_LSTM.CNN_LSTM_data_processing import save_future_target_reference

# Re-export commonly used helpers for older imports.
from src.CNN_LSTM.CNN_LSTM_dataset import create_loader, create_sequences
from src.CNN_LSTM.CNN_LSTM_plots import (
    plot_actual_vs_predicted_scatter,
    plot_predictions,
)
from src.CNN_LSTM.CNN_LSTM_scattering import (
    build_scattering_transform,
    compute_static_scattering_features,
    get_scattering_feature_names,
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
    use_gap_aware_segments=DEFAULT_USE_GAP_AWARE_SEGMENTS,
    use_station_one_hot=DEFAULT_USE_STATION_ONE_HOT,
    show_prediction_plot=True,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES,
    use_attention=DEFAULT_USE_ATTENTION
):
    # Trains, evaluates, plots, and returns reusable outputs for the CNN-LSTM CO2 model.
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
    print("Gap-aware sequence generation:", use_gap_aware_segments)
    print("Use station one-hot encoding:", use_station_one_hot)
    print("Convolution channels:", DEFAULT_CONV_CHANNELS)
    print("Use scattering:", use_scattering)
    print("Use attention:", use_attention)
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
        use_gap_aware_segments=use_gap_aware_segments,
        use_station_one_hot=use_station_one_hot,
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
        dropout=dropout_rate,
        use_attention=use_attention
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
        use_gap_aware_segments=use_gap_aware_segments,
        use_station_one_hot=use_station_one_hot,
        use_scattering=use_scattering,
        scattering_j=scattering_j,
        scattering_q=scattering_q,
        n_scattering_features=n_scattering_features,
        use_attention=use_attention
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

    attention_plot_paths = None
    if use_attention:
        attention_plot_paths = plot_attention_weights(
            model=model,
            test_loader=test_loader,
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
        "attention_plot_paths": attention_plot_paths,
        "horizon_metrics_path": horizon_metrics_path,
        "future_target_reference_path": future_target_reference_path,
        "training_runtime_seconds": run_elapsed_time,
        "training_runtime_formatted": format_elapsed_time(run_elapsed_time),

        # Objects reused by optional explainability and uncertainty stages.
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "X_train": X_train,
        "X_test": X_test,
        "model_features": model_features,
        "output_seq_length": output_seq_length,
        "resample_time": resample_time,
        "epochs": epochs,
        "batch_size": batch_size,
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
        "gap_aware_sequences": use_gap_aware_segments,
        "use_station_one_hot": use_station_one_hot,
        "use_scattering": use_scattering,
        "scattering_j": scattering_j,
        "scattering_q": scattering_q,
        "n_scattering_features": n_scattering_features,
        "use_attention": use_attention,
        "input_size": input_size,
    }
