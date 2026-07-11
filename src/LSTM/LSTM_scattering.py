import numpy as np

from src.LSTM.LSTM_config import (
    DEFAULT_N_SCATTERING_FEATURES,
    DEFAULT_SCATTERING_J,
    DEFAULT_SCATTERING_Q,
)

try:
    from kymatio.numpy import Scattering1D
except ImportError:
    Scattering1D = None


def get_scattering_feature_names(n_scattering_features):
    # Creates stable names for appended scattering wavelet feature columns.
    return [
        f"scatter_co2_{i + 1}"
        for i in range(n_scattering_features)
    ]

def build_scattering_transform(
    input_seq_length,
    scattering_j=DEFAULT_SCATTERING_J,
    scattering_q=DEFAULT_SCATTERING_Q
):
    # Builds the Kymatio 1D scattering transform used for CO2 signal features.
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
    # Converts one CO2 input window into a fixed-length scattering feature vector.
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
