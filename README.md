# MarkovStates: Weather Regime Analysis using Hidden Markov Models

**A machine learning pipeline for identifying and analyzing distinct weather regimes using Hidden Markov Models**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Option 1: Conda (Recommended)](#option-1-conda-recommended)
  - [Option 2: Manual Installation](#option-2-manual-installation)
- [Environment Setup](#environment-setup)
- [Running the Project](#running-the-project)
- [User Inputs & Parameters](#user-inputs--parameters)
- [Output & Interpretation](#output--interpretation)
- [Project Architecture](#project-architecture)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Overview

**MarkovStates** is an end-to-end machine learning pipeline that uses **Hidden Markov Models (HMM)** to identify and analyze distinct weather regimes. Rather than treating weather as a continuous phenomenon, this project models weather as a sequence of discrete states (regimes) that transition probabilistically from one to another.

Using historical weather data from Madrid, Spain (via the Open-Meteo API), the project:
- Identifies 5 distinct weather regimes (Warm/Humid, Cool/Moderate, Warm/Dry, Cold/Dry, Hot/Peak Summer)
- Computes transition probabilities between regimes
- Calculates steady-state probabilities (long-run regime distribution)
- Provides regime characteristics (mean temperature, pressure, dew point per regime)
- Generates professional visualizations and statistical analysis

This approach is useful for climate modeling, weather pattern analysis, and understanding seasonal weather transitions.

---
#### Pipeline Steps

```
Historical Weather Data
    ↓
[Data Collection] → Fetch from Open-Meteo API (hourly data)
    ↓
[Preprocessing] → Resample to daily, interpolate missing values
    ↓
[Feature Selection] → Factor analysis to select 3 key variables
    ↓
[Scaling] → Standardize features (mean=0, std=1)
    ↓
[Model Training] → Fit Gaussian HMM (5 states, 50 seeds)
    ↓
[Regime Prediction] → Assign regime label to each day
    ↓
[Analysis] → Transition matrix, steady-state, characteristics
    ↓
[Visualization] → Generate multi-panel plots
```

---

## Project Structure

```
markovstates/
├── main.py                    # Entry point - end-to-end pipeline
├── README.md                  # This file
├── environment.yml            # Conda environment specification
├── pyproject.toml             # Project metadata and dependencies
│
├── markovstates/              # Main package
│   ├── __init__.py
│   ├── data_collect.py        # Fetch data from Open-Meteo API
│   ├── preprocessing.py       # Data cleaning and resampling
│   ├── factor_analysis.py     # Feature selection via factor analysis
│   ├── models.py              # HMM model classes (abstract + concrete)
│   ├── utils.py               # Central import hub (avoids circular deps)
│   ├── cli.py                 # CLI argument parsing
│   └── scrap.py               # Experimental code
│
├── models/                    # Pre-trained models
│   ├── hmm_final.pkl          # Trained Gaussian HMM (5 states)
│   └── scaler.pkl             # Feature scaler
│
├── notebooks/                 # Jupyter analysis notebooks
│   ├── data_explore.ipynb     # Data exploration and statistics
│   ├── factor_explore.ipynb   # Feature importance analysis
│   ├── models_explore.ipynb   # Model evaluation and diagnostics
│   └── viz.ipynb              # Visualization and interpretation
│
└── testing/                   # Unit tests
    ├── test_data_collect.py
    └── test_preprocessing.py
```

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Package Manager**: Conda (miniforge/anaconda) or pip
- **Internet Connection**: Required for downloading weather data from Open-Meteo

### Option 1: Conda (Recommended)

**Step 1: Clone or navigate to the project directory**
```bash
cd /path/to/markovstates
```

**Step 2: Create and activate the conda environment**
```bash
conda env create -f environment.yml
conda activate markovstates
```

This automatically installs all dependencies including:
- numpy, pandas, scipy, scikit-learn
- hmmlearn (Hidden Markov Model library)
- matplotlib, seaborn (visualization)
- openmeteo-requests, requests-cache (data fetching)

**Step 3: Verify installation**
```bash
python -c "import markovstates; print('Installation successful!')"
```

### Option 2: Manual Installation

**Step 1: Create a virtual environment (if not using conda)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 2: Upgrade pip**
```bash
pip install --upgrade pip
```

**Step 3: Install dependencies from pyproject.toml**
```bash
pip install -e .
```

Or install manually:
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn pytest hmmlearn openmeteo-requests requests-cache retry-requests
```

**Step 4: Install the package in development mode**
```bash
pip install -e .
```

---

## Environment Setup

### Using Conda (Recommended)

The `environment.yml` file specifies all dependencies:

```yaml
name: markovstates
channels: 
  - conda-forge
dependencies:
  - python >= 3.10
  - numpy
  - pandas
  - hmmlearn
  - matplotlib
  - seaborn
  - scipy
  - scikit-learn
  - pip
  - pip:
    - openmeteo_requests
    - requests_cache 
    - retry_requests
```

**To set up:**
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate markovstates

# Verify all packages
conda list
```

### Using pip with Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install from pyproject.toml
pip install -e .
```

### Troubleshooting Environment Issues

**Issue**: `ModuleNotFoundError: No module named 'hmmlearn'`
```bash
conda install -y hmmlearn
# or
pip install hmmlearn
```

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
```bash
conda install -y scikit-learn
# or
pip install scikit-learn
```

**Issue**: `ModuleNotFoundError: No module named 'seaborn'`
```bash
conda install -y seaborn
# or
pip install seaborn
```

---

## Running the Project

### Quick Start

From the project root directory, run:

```bash
python main.py
```

This runs the pipeline with default settings (start date: 2023-04-10, end date: today, using pre-trained model).

### Command-Line Arguments

Customize the pipeline with command-line arguments:

```bash
# Use custom date range
python main.py --start 2024-01-01 --end 2024-12-31

# Train a new model
python main.py --retrain

# Combine options
python main.py -s 2023-06-01 -e 2024-06-01 --retrain
```

**Available Arguments:**

- `--start, -s DATE`: Start date in YYYY-MM-DD format (default: 2023-04-10)
- `--end, -e DATE`: End date in YYYY-MM-DD format (default: today's date)
- `--retrain`: Train a new model instead of using the pre-trained one (flag, no value needed)
---

## User Inputs & Parameters

### Command-Line Arguments

| Argument | Shorthand | Type | Default | Description |
|----------|-----------|------|---------|-------------|
| `--start` | `-s` | str (YYYY-MM-DD) | 2023-04-10 | Begin date for weather data |
| `--end` | `-e` | str (YYYY-MM-DD) | Today's date | End date for weather data |
| `--retrain` | — | flag | false | Train a new model (no value needed) |

### Configurable Model Parameters

Edit `main.py` to modify:

```python
# In train_or_load_model() function:
hmm = HMMWeatherModel(
    n_components=5,      # Number of weather regimes
    covar_type='diag',   # Covariance type (diag, tied, full, spherical)
    n_restarts=50        # Number of random seeds to test
)
```

### Fixed Weather Features

The model uses these 3 features selected via factor analysis:

```python
FINAL_FEATURES = [
    "temperature_2m",      # 2-meter air temperature (°C)
    "surface_pressure",    # Sea-level pressure (hPa)
    "dew_point_2m"         # Dew point temperature (°C)
]
```

### Regime Labels

The 5 identified weather regimes are:

| ID | Name | Characteristics |
|:--:|------|-----------------|
| 0 | Warm/Humid | High temperature, high humidity |
| 1 | Cool/Moderate | Moderate temperature, moderate conditions |
| 2 | Warm/Dry | Warm temperature, low humidity |
| 3 | Cold/Dry | Low temperature, dry conditions |
| 4 | Hot/Peak Summer | Highest temperature, very dry |

---

### Module Descriptions

#### `data_collect.py`
- Connects to Open-Meteo Archive API
- Fetches hourly historical weather data
- Preprocesses into pandas DataFrame
- **Exports**: `hourly_dataframe`, `response`

#### `preprocessing.py`
- **`Preprocess`** class: Resampling, missing value handling, scaling
- **`FeatMat`** class: Feature matrix construction pipeline
- Methods: `resample()`, `handle_missing()`, `fit_scaler()`, `apply_scaler()`

#### `factor_analysis.py`
- Applies Factor Analysis for feature reduction
- Identifies 3 most informative weather variables
- **Exports**: `FINAL_FEATURES` (temperature, pressure, dew point)

#### `models.py`
- **`WeatherModel`** (ABC): Abstract base class defining interface
- **`HMMWeatherModel`** (concrete): Gaussian HMM implementation
- Methods: `fit()`, `predict()`, `score()`, `transition_mat()`, `save()`, `load()`
- Features: Multi-restart training for robustness

#### `utils.py`
- Central import hub to prevent circular dependencies
- Exports: `Preprocess`, `FeatMat`, `hourly_dataframe`, `WeatherModel`, `HMMWeatherModel`

#### `main.py`
- End-to-end pipeline orchestration
- User interaction and CLI
- 8-stage execution flow
- Professional output formatting

---
