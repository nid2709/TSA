# Time Series Analysis with PyTorch

This repository contains code and notebooks for time series analysis and forecasting using PyTorch. The project includes data preprocessing, model implementations (e.g., LSTM, Transformer‑style models), and evaluation scripts.

## 🧰 Prerequisites

You need:
- Python 3.11 (via Anaconda)
- PyTorch with `torchvision` and `torchaudio`
- Basic experience with Jupyter notebooks or Python scripts

---

## 🛠 Installation & Setup

1. **Install Anaconda**  
   Download and install Anaconda from:  
   [https://www.anaconda.com/download](https://www.anaconda.com/download)

2. **Open Terminal (Mac) or Anaconda Prompt (Windows)**

3. **Create a Conda environment named `torch-env` with Python 3.11**  
   ```bash
   conda create -n torch-env python=3.11
   ```

4. **Activate the environment**  
   ```bash
   conda activate torch-env
   ```

5. **Install PyTorch and associated libraries**  
   ```bash
   python -m pip install torch torchvision torchaudio
   ```

6. **Verify the installation**  
   Run this command to check the PyTorch version:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```
   If you see a version number (e.g., `2.0.1`), the installation was successful.

---

## 📂 Repository structure
- data/ #Raw and processed datasets
- notebooks/ #Jupyter notebooks
- src/ #Python scripts
   - modeling.py #model
   - train.py #Training pipeline
   - utils.py 3Helper functions
- results/ #output, plots and prediction
- requirements.txt #dependencies
- README.md


---

## 🚀 Getting started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/time-series-analysis.git
   cd time-series-analysis
   ```

2. **Ensure your `torch-env` environment is activated**
   ```bash
   conda activate torch-env
   ```

3. **Install any additional dependencies (optional)**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run a training example**
   ```bash
   python src/train.py
   ```
   (Adjust arguments and paths according to your project.)

5. **Explore analysis notebooks**
   ```bash
   jupyter notebook notebooks/eda_forecasting.ipynb
   ```

---

## 📊 Example (PyTorch usage)

Inside your scripts or notebooks, you can use PyTorch like:

```python
import torch
import torch.nn as nn

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.LSTM(input_size=1, hidden_size=50, num_layers=1)
```

---

## 📜 License

This project is open source and available under the MIT License (you can modify this as needed).