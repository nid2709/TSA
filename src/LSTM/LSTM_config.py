import os

#To prevent temporary config/cache files
MPL_CONFIG_DIR = os.path.join(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
    ".matplotlib"
)
os.makedirs(MPL_CONFIG_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CONFIG_DIR)

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
DEFAULT_OUTPUT_SEQ_LENGTH = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.00003
DEFAULT_HIDDEN_SIZE = 128
DEFAULT_RESAMPLE_TIME = '15min'
DEFAULT_DROPOUT_RATE = 0.2
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_LAYERS = 2

DEFAULT_MAX_FILL_STEPS = 2
DEFAULT_DROP_SHORT_STATIONS = True
DEFAULT_CLIP_OUTLIERS = True
DEFAULT_OUTLIER_CLIP_FACTOR = 1.5
DEFAULT_RESTORE_BEST_MODEL = True
DEFAULT_DEVICE = "mps"
DEFAULT_USE_GAP_AWARE_SEGMENTS = False
DEFAULT_USE_STATION_ONE_HOT = True

DEFAULT_USE_SCATTERING = False
DEFAULT_SCATTERING_J = 4
DEFAULT_SCATTERING_Q = 8
DEFAULT_N_SCATTERING_FEATURES = 16
DEFAULT_USE_ATTENTION = False


def get_target_label(target_column):
    # Converts internal target column names into readable labels for plots and logs.
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
    # Converts runtime seconds into a readable hours/minutes/seconds string.
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"

    if minutes >= 1:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"

    return f"{remaining_seconds:.2f}s"


def get_lstm_results_dir(
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
    use_gap_aware_segments=DEFAULT_USE_GAP_AWARE_SEGMENTS,
    use_station_one_hot=DEFAULT_USE_STATION_ONE_HOT,
    use_scattering=DEFAULT_USE_SCATTERING,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q,
    n_scattering_features=DEFAULT_N_SCATTERING_FEATURES,
    use_attention=DEFAULT_USE_ATTENTION
):
    # Builds a unique results folder name from the active LSTM experiment settings.
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )

    folder_name = (
        f"LSTM_results_"
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
        f"GAP{int(use_gap_aware_segments)}_"
        f"SOH{int(use_station_one_hot)}_"
        f"SWT{int(use_scattering)}_"
        f"SWJ{scattering_j if use_scattering else 0}_"
        f"SWQ{scattering_q if use_scattering else 0}_"
        f"SWF{n_scattering_features if use_scattering else 0}_"
        f"ATT{int(use_attention)}"
    )

    return os.path.join(project_root, folder_name)
