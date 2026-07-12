# Indoor Air Quality Time-Series Forecasting (TSA)

This project develops and evaluates deep-learning models for multivariate indoor-air-quality time-series forecasting. It uses readings collected from multiple monitoring stations to predict variables such as **CO2**, **air quality index (AQI)**, and **temperature** over short and longer forecast horizons.

The repository contains reusable PyTorch pipelines for LSTM, CNN-LSTM, and N-BEATS models, plus a broad collection of Jupyter experiments for Transformer, wavelet-scattering, explainability, and uncertainty-quantification methods.

## Project capabilities

- Multi-station time-series preprocessing and chronological train/validation/test splitting
- LSTM and CNN-LSTM forecasting pipelines
- N-BEATS variants for CO2, AQI, and temperature prediction
- Transformer experiments for horizons from 15 minutes to 24 hours
- Optional wavelet-scattering features using Kymatio
- Optional temporal attention in the LSTM and CNN-LSTM models
- Explainable AI using SHAP, permutation feature importance, and Integrated Gradients
- Predictive uncertainty using Monte Carlo dropout and deep ensembles
- Evaluation with MSE, MAE, RMSE, R-squared, and per-horizon metrics
- Automated saving of plots, metrics, predictions, and trained model checkpoints

## Dataset

The main dataset is:

```text
data/indoorAir2.csv
```

It contains approximately **1.9 million observations** from multiple indoor monitoring stations. Timestamps are stored as Unix seconds and converted to datetimes during loading.

### Available sensor columns

| Sensor family | Measurements |
|---|---|
| SCD41 | CO2, temperature, humidity |
| ENS160 | equivalent CO2, TVOC, AQI |
| SVM41 | temperature, humidity, NOx index, VOC index |
| BME688 | temperature, humidity, pressure, gas resistance |
| SFA30 | temperature, humidity, formaldehyde (HCO) |
| Metadata | record ID, station ID, station name, timestamp |

The file `data/AIQStationsDocumentation 2.pdf` contains supporting station documentation.

### Default model features

The current LSTM and CNN-LSTM CO2 pipelines use:

- `ens160_aqi`
- `ens160_tvoc`
- `bme688_gas_resistance`
- `bme688_pressure`
- `scd41_temperature`
- `scd41_humidity`
- `scd41_co2`
- cyclical hour and weekday features
- a weekend indicator
- optional one-hot station features
- optional learned wavelet-scattering features

The default target is `scd41_co2`.

## Repository structure

```text
TSA/
├── data/
│   ├── indoorAir2.csv
│   └── AIQStationsDocumentation 2.pdf
├── src/
│   ├── LSTM/                 # Modular LSTM pipeline
│   ├── CNN_LSTM/             # Modular CNN-LSTM pipeline
│   └── N-BEATS/              # N-BEATS model variants
├── notebooks/
│   ├── EDA.ipynb
│   ├── LSTM and CNN_LSTM Notebook/
│   ├── N-BEATS/
│   ├── Transformer/
│   └── Transformer_new/
├── Transformer_Results/      # Saved Transformer artifacts
├── dataPipeline_LSTM_co2.py
├── dataPipeline_CNN_LSTM_co2.py
├── data_pipeline_NBEATS.py
├── dataPipeline_LSTM_old.py
├── dataPipeline_CNN_LSTM_old.py
├── requirement.yml
└── README.md
```

Files containing `_old` are retained legacy pipelines. For current LSTM and CNN-LSTM work, use the `_co2.py` entry points.

## Installation

The project uses Python 3.11 and the Conda environment name `torch-env`.

### Create the environment

From the project directory:

```bash
cd "/Volumes/Hackintosh - Data/Users/Manvi/TSA"
conda env create -f requirement.yml
conda activate torch-env
```

### Update an existing environment

If `torch-env` already exists:

```bash
conda env update -n torch-env -f requirement.yml --prune
conda activate torch-env
```

### Register the Jupyter kernel

```bash
python -m ipykernel install --user --name torch-env --display-name "Python (torch-env)"
```

Then select **Python (torch-env)** when opening a notebook.

### Verify the installation

```bash
python -c "import torch, tensorflow, pandas, sklearn; print('Dependencies are available')"
```

The environment includes PyTorch, TensorFlow, NeuralForecast, NumPy, pandas, SciPy, scikit-learn, statsmodels, Matplotlib, Seaborn, PyWavelets, Kymatio, SHAP, Captum, and Jupyter.

## Running the project

Always run commands from the repository root so local `src` imports and data paths resolve correctly.

### LSTM CO2 pipeline

```bash
conda activate torch-env
python dataPipeline_LSTM_co2.py
```

This entry point performs data loading, preprocessing, LSTM training, evaluation, and—by default—SHAP/PFI/Integrated Gradients, Monte Carlo dropout, and deep-ensemble analysis.

### CNN-LSTM CO2 pipeline

```bash
conda activate torch-env
python dataPipeline_CNN_LSTM_co2.py
```

This runs the equivalent workflow using a convolutional front end followed by an LSTM model.

### N-BEATS pipeline

```bash
python data_pipeline_NBEATS.py
```

The active section currently runs short-term CO2 prediction. Other variants in the file can be enabled for long-term forecasts, syN-BEATS, wavelet N-BEATS, fixed horizons, explainability/uncertainty, AQI, and temperature.

> **Current N-BEATS path note:** `data_pipeline_NBEATS.py` imports `src.N_BEATS`, while the repository directory is named `src/N-BEATS`. Python package names cannot contain a hyphen. Rename that directory to `src/N_BEATS` (and add `__init__.py` files if needed) before using this entry point, or update its import strategy.

### Jupyter notebooks

Start JupyterLab with:

```bash
jupyter lab
```

The notebooks include:

- exploratory data analysis
- step-by-step LSTM and CNN-LSTM development
- N-BEATS forecasting and uncertainty experiments
- SHAP, Integrated Gradients, and permutation feature importance
- Transformer forecasts at 15-minute, 2-hour, 4-hour, 6-hour, 12-hour, and 24-hour horizons
- wavelet-scattering and Transformer/BiLSTM experiments

Some notebooks contain saved outputs or historical experiment paths. Check each notebook's data path and selected Jupyter kernel before running all cells.

## Preprocessing workflow

The modular LSTM and CNN-LSTM pipelines currently:

1. Load `data/indoorAir2.csv`.
2. Convert Unix timestamps to pandas datetimes.
3. Sort readings by station and time.
4. Exclude station 6 to match the N-BEATS preprocessing convention.
5. Resample each station to 15-minute intervals by default.
6. Interpolate and fill selected missing sensor values within each station.
7. Add cyclical hour and weekday features plus a weekend flag.
8. Split each station chronologically into 70% training, 15% validation, and 15% test data.
9. Fit scaling from training data and create sliding input/forecast windows.
10. Optionally clip outliers, preserve gap-aware segments, encode station identity, and add scattering features.

Chronological splitting is important: future observations are not randomly mixed into the training set.

## Default LSTM and CNN-LSTM settings

| Setting | Default |
|---|---:|
| Input sequence length | 192 time steps |
| Output sequence length | 1 time step |
| Sampling interval | 15 minutes |
| Batch size | 32 |
| Epochs | 10 |
| Learning rate | 0.00003 |
| Hidden size | 128 |
| Recurrent layers | 2 |
| Dropout | 0.2 |
| Weight decay | 0.0001 |
| Restore best validation model | Yes |
| Station one-hot encoding | Yes |
| Gap-aware sequences | No |
| Wavelet scattering | No |
| Attention | No |

With a 15-minute sampling interval, 192 input steps represent 48 hours of history. Increase `output_seq_length` to produce multi-step forecasts.

Configuration is defined in:

```text
src/LSTM/LSTM_config.py
src/CNN_LSTM/CNN_LSTM_config.py
```

Model functions also accept keyword arguments, allowing experiment-specific overrides without changing every module.

## Optional pipeline stages

At the top of `dataPipeline_LSTM_co2.py` and `dataPipeline_CNN_LSTM_co2.py`, these switches control additional work:

```python
RUN_EXPLAINABILITY = True
RUN_MC_DROPOUT = True
RUN_DEEP_ENSEMBLE = True
SAVE_EDA_PLOTS = False
```

Explainability and ensemble stages can be computationally expensive. Set unwanted stages to `False` for a faster model-only run. Set `SAVE_EDA_PLOTS = True` to save time-series, correlation-heatmap, and PCA plots.

## Explainability

The LSTM and CNN-LSTM modules support:

- **SHAP:** estimates feature contributions to model predictions
- **Permutation feature importance:** measures performance degradation after shuffling a feature
- **Integrated Gradients:** attributes predictions to input features and time steps
- **Attention plots:** visualizes temporal attention when attention is enabled

Explainability artifacts are stored inside experiment-specific `shap/`, `pfi/`, and `integrated_gradients/` directories.

## Uncertainty quantification

Two approaches are implemented:

- **Monte Carlo dropout:** performs repeated stochastic forward passes and summarizes prediction mean, standard deviation, and intervals
- **Deep ensembles:** trains multiple independently initialized models and measures variation across their predictions

These outputs are saved under `mc_dropout/` and `deep_ensemble/` within each results directory.

## Evaluation and results

Forecasting performance is evaluated with:

- mean squared error (MSE)
- mean absolute error (MAE)
- root mean squared error (RMSE)
- coefficient of determination (R-squared)
- per-horizon metrics for multi-step forecasting

LSTM and CNN-LSTM runs create a uniquely named directory in the project root. The name records major hyperparameters such as input/output length, batch size, epochs, learning rate, hidden size, resampling interval, dropout, gap handling, station encoding, scattering, and attention.

A typical results directory contains:

```text
<model>_results_<experiment-settings>/
├── main_plots/               # Loss, predictions, horizon metrics, attention
├── eda_plots/                # Optional EDA outputs
├── shap/                     # SHAP artifacts
├── pfi/                      # Permutation importance
├── integrated_gradients/     # Captum attribution outputs
├── mc_dropout/               # MC-dropout uncertainty
└── deep_ensemble/            # Ensemble uncertainty
```

`Transformer_Results/` contains saved Transformer metrics, predictions, uncertainty arrays, feature-importance outputs, plots, and `.pth` checkpoints from existing experiments.

## Hardware acceleration

The modular PyTorch code prefers Apple's Metal Performance Shaders (`mps`) on supported Macs. If MPS is unavailable, it falls back to CPU. CUDA can be selected in the model configuration on a compatible NVIDIA system.

Check PyTorch device availability with:

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available()); print('CUDA:', torch.cuda.is_available())"
```

The full dataset and optional explainability/uncertainty stages can require substantial memory and runtime. Begin with fewer epochs or disabled optional stages when validating a new environment.

## Reproducible experiment workflow

1. Activate `torch-env`.
2. Confirm the dataset path and target column.
3. Record configuration changes before running.
4. Keep the chronological split to avoid leakage.
5. Run the base model first.
6. Enable explainability or uncertainty stages as needed.
7. Compare both overall and per-horizon metrics.
8. Retain the generated parameterized results directory with the corresponding code revision.

## Troubleshooting

### `ModuleNotFoundError`

Activate the correct environment and run from the project root:

```bash
conda activate torch-env
conda env update -n torch-env -f requirement.yml
```

For the N-BEATS-specific `src.N_BEATS` error, see the directory-name note in the N-BEATS section.

### Jupyter uses the wrong Python environment

Register the kernel and select **Python (torch-env)**:

```bash
python -m ipykernel install --user --name torch-env --display-name "Python (torch-env)"
```

### MPS operation is unsupported

Set `DEFAULT_DEVICE = "cpu"` in the relevant configuration file. Training will be slower but more broadly compatible.

### Training takes too long

- reduce `DEFAULT_EPOCHS`
- shorten the input sequence
- disable SHAP, MC dropout, and deep ensembles
- test on a smaller dataset sample before a full run

### Out-of-memory errors

- reduce batch size
- disable optional analysis stages
- shorten input/output sequences
- run explainability on fewer samples

## Notes

- The dataset is large and is part of the repository; avoid accidentally committing additional generated copies.
- Model checkpoints (`.pth`) and result arrays (`.npy`) can be large.
- `.ipynb_checkpoints`, `.DS_Store`, Matplotlib caches, and generated result directories are development artifacts rather than source code.
- Review privacy, consent, and data-sharing requirements before publishing raw station data.

## License and citation

No explicit license or citation information is currently included in this repository. Add a `LICENSE` file before public distribution, and add the project authors, institution, dataset source, and preferred citation here when available.
