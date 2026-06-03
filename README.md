# Battery Prediction Models — Project Documentation

## 📁 Directory Structure

```
battery/
├── Battery_dataset (1).csv        ← Source data (680 samples, 3 batteries)
├── Battery (1).pdf                ← Reference architecture PDF
│
├── analytical_model/              ← Statistical / physics-based models
│   ├── main.py                    ← Run this to train everything
│   ├── data_loader.py             ← Load & clean dataset
│   ├── feature_engineering.py    ← Derived feature computation
│   ├── eda.py                     ← Exploratory Data Analysis plots
│   ├── visualization.py           ← Prediction & comparison plots
│   ├── models/
│   │   ├── linear_model.py        ← Linear / Ridge / Lasso regression
│   │   ├── polynomial_model.py    ← Polynomial regression (deg 2/3)
│   │   └── degradation_model.py   ← Exponential decay curve fitting
│   ├── evaluation/
│   │   └── metrics.py             ← MAE, RMSE, R², MAPE, MaxError
│   ├── results/                   ← Generated plots & CSV (auto-created)
│   └── saved_models/              ← .pkl pipeline files (auto-created)
│
└── ann_model/                     ← Deep ANN model (PyTorch)
    ├── main.py                    ← Run this to train everything
    ├── data_loader.py             ← Load, engineer features, split
    ├── preprocessor.py            ← StandardScaler + MinMaxScaler
    ├── trainer.py                 ← Training history & visualisation
    ├── models/
    │   └── ann.py                 ← StandardANN + MultiTaskANN + BatteryANN
    ├── evaluation/
    │   └── metrics.py             ← MAE, RMSE, R², MAPE, MaxError
    ├── results/                   ← Generated plots & CSV (auto-created)
    └── saved_models/              ← .pt model files + scalers (auto-created)
```

---

## 🔋 Dataset

| Column | Description | Unit |
|--------|-------------|------|
| `battery_id` | Battery identifier (B5, B6, B7) | — |
| `cycle` | Charge/discharge cycle number | — |
| `chI` | Charging current | A |
| `chV` | Charging voltage | V |
| `chT` | Charging temperature | °C |
| `disI` | Discharging current | A |
| `disV` | Discharging voltage | V |
| `disT` | Discharging time | min |
| `BCt` | Battery capacity | Ah |
| **`SOH`** | **State of Health** ← target | **%** |
| **`RUL`** | **Remaining Useful Life** ← target | **cycles** |

---

## 📊 Analytical Model

### Models Trained

| Model | Description |
|-------|-------------|
| Linear Regression | OLS baseline |
| Ridge Regression (α=1.0) | L2 regularised |
| Lasso Regression (α=0.01) | L1 regularised (feature selection) |
| Polynomial deg=2 (Ridge) | Nonlinear capacity fade capture |
| Polynomial deg=3 (Ridge) | Higher-order degradation |
| Exponential Decay Model | Physics-based: SOH = a·exp(−b·n) + c |

### Engineered Features (16 total)
`cycle, chI, chV, chT, disI, disV, disT, BCt,`
`charge_energy, discharge_energy, energy_efficiency,`
`temp_diff, current_ratio, voltage_drop, capacity_fade_rate, cycle_normalized`

### Results (on test set)

| Model | SOH R² | SOH RMSE | RUL R² | RUL RMSE |
|-------|--------|----------|--------|----------|
| Linear Regression | 1.0000 | ~0.00 | 0.9827 | 8.06 |
| Ridge Regression | 0.9999 | 0.1269 | 0.9827 | 8.06 |
| Lasso Regression | 1.0000 | 0.0808 | 0.9833 | 7.93 |
| Polynomial (deg=2) | 0.9999 | 0.1902 | **0.9896** | **6.26** |
| Polynomial (deg=3) | 0.9971 | 0.8629 | 0.9675 | 11.06 |
| Exponential Model | 0.9890 | 1.7464 | — | — |

### How to Run
```powershell
cd analytical_model
$env:PYTHONUTF8=1; python main.py
```

---

## 🧠 ANN Model

### Architecture

**Standard ANN:**
```
Input(16) → Dense(256)→BN→ReLU→Drop(0.2)
          → Dense(128)→BN→ReLU→Drop(0.2)
          → Dense(64) →BN→ReLU→Drop(0.2)
          → Dense(32) →BN→ReLU→Drop(0.2)
          → Dense(2)  → [SOH, RUL]
```

**Multi-Task ANN:**
```
Input(16) → Shared Dense(256)→BN→ReLU→Drop(0.2)
          → Shared Dense(128)→BN→ReLU→Drop(0.2)
          ↙                              ↘
SOH Head [Dense(64)→Dense(32)→Linear(1)]  RUL Head [Dense(64)→Dense(32)→Linear(1)]
          ↘                              ↙
              Concatenate → output [SOH, RUL]
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Weight decay (L2) | 1e-4 |
| Initial LR | 1e-3 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=12) |
| Early stopping | patience=30, restore best |
| Max epochs | 300 |
| Batch size | 32 |
| Feature scaler | StandardScaler |
| Target scaler | MinMaxScaler [0,1] |

### ANN Results (Standard architecture)

| Target | MAE | RMSE | R² |
|--------|-----|------|----|
| SOH | 1.785 | 2.384 | 0.978 |
| RUL | 12.24 | 14.72 | 0.946 |

### How to Run
```powershell
cd ann_model
$env:PYTHONUTF8=1; python main.py
```

---

## 🧬 Hybrid Physics-Informed ANN

### Architecture
Implements the end-to-end framework merging data-driven and physics-based models:
1. **Physics Engine**: Calculates degradation rates, yielding `SOH_phys`, `RUL_phys_norm`, `PDI`, thermal stress, and current stress.
2. **Feature Fusion**: Concatenates the 14 data-driven features and 5 physics outputs into a 19-dimensional input vector.
3. **Multi-task ANN**: Shared trunk [256→128→64] with dual heads [32→16] for SOH and RUL.
4. **Physics-Informed Loss**: `Loss = λ_data * MSE + λ_phys * Physics_Adherence + λ_mono * Monotonicity_Penalty`.
5. **Physics Validation Layer**: Computes standard metrics plus **PAS** (Physics Adherence Score) and **DTCS** (Degradation Trajectory Consistency Score).

### Hybrid Results (Test Set)

| Target | MAE | RMSE | R² | PAS | DTCS |
|--------|-----|------|----|-----|------|
| **SOH** | 2.83 | 3.50 | 0.953 | 0.966 | 0.747 |
| **RUL** | 16.95 | 19.28 | 0.908 | 0.966 | 0.747 |

> **PAS (0.966)** indicates the model predictions align very strongly with physical degradation laws.
> **DTCS (0.747)** confirms a high degree of monotonic physical degradation consistency.

### How to Run
```powershell
cd hybrid_model
$env:PYTHONUTF8=1; python main.py
```

---

## 🚀 Quick Start

```powershell
# Install dependencies (already installed)
pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib torch

# Run Analytical Model
cd c:\Users\avijn_th5xjtu\Desktop\code\battery\analytical_model
$env:PYTHONUTF8=1; python main.py

# Run ANN Model
cd c:\Users\avijn_th5xjtu\Desktop\code\battery\ann_model
$env:PYTHONUTF8=1; python main.py
```

> **Note:** The `$env:PYTHONUTF8=1` prefix is needed on Windows PowerShell to handle UTF-8 emoji in console output.
