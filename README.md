# Prophet Forecasting for Portfolio Optimisation

## Project Overview
An end-to-end machine learning project that forecasts stock and asset prices using Facebook/Meta Prophet time series forecasting model, then applies Markowitz portfolio optimisation to rebalance portfolios based on these forecasts.

_(Needless to say it's for illustrative purposes and not financial advice)._

## Components

### 1. Prophet (Time Series Forecasting)

**What is Prophet?**

Prophet is Facebook's open-source time series forecasting tool designed for business forecasting. It handles trends, seasonality, and holidays automatically, making it robust and easy to use for forecasting time series data.

**How It Works in This Project:**

- Input: Historical price time series with datetime index
- Model: Prophet fits additive components (trend, seasonality, holidays)
- Output: Forecasted prices for each asset in the portfolio for the next trading day
- Training: The model fits to historical price data and generates one-step-ahead forecasts

### 2. Markowitz Portfolio Optimisation

**What is Markowitz Portfolio Optimisation?**

Markowitz portfolio optimisation, also known as Modern Portfolio Theory (MPT), is a mathematical framework for constructing optimal portfolios. Developed by Harry Markowitz in 1952, it balances the trade-off between expected returns and risk.

**Key Concepts:**

- **Expected Return**: The weighted average of expected returns of individual assets
- **Risk (Volatility)**: Measured as the standard deviation of portfolio returns
- **Correlation**: How assets move relative to each other
- **Efficient Frontier**: The set of optimal portfolios offering the highest expected return for a given level of risk

**The Optimisation Problem:**


```

Maximize: μᵀw - λ(wᵀΣw)

Subject to:

* Σwᵢ = 1 (weights sum to 1)
* wᵢ ≥ 0 (long-only portfolio, optional)
* Additional constraints (sector limits, etc.)

```

Where:
- `μ` = vector of expected returns (from Prophet price forecasts)
- `Σ` = covariance matrix of asset returns
- `w` = portfolio weights
- `λ` = risk aversion parameter (configurable in `src/settings.py`)

**How It Works in This Project:**

1. **Input**: Forecasted returns (derived from Prophet price predictions) for each asset
2. **Risk Estimation**: Historical covariance matrix calculated from asset returns
3. **Optimisation**: Solves for optimal weights that maximise risk-adjusted returns using SciPy's SLSQP solver
4. **Output**: Recommended portfolio allocation (weights for each asset)
5. **Rebalancing**: Portfolio is rebalanced based on these optimal weights

## Project Workflow


```

Historical Data Extraction
↓
Data Preprocessing
↓
Prophet Model Training
↓
Price Forecasting
↓
Markowitz Optimisation
↓
Optimal Portfolio Weights
↓
Results Saved to Supabase
↓
Streamlit Dashboard Hosted on Hostinger VPS

```

## Installation

### Prerequisites

- **Python 3.12+**
- **Poetry** (for dependency management)
- **Supabase Account** (for storing results)
- **CircleCI Account** (optional, for CI/CD)

### 1. Environment Setup

#### Option A: Linux & macOS
If you are on Linux or macOS, you can install Python and Poetry directly. I recommend using `pyenv` to manage Python versions.

```bash
# Install Pyenv (if not installed)
curl [https://pyenv.run](https://pyenv.run) | bash

# Install Python 3.12
pyenv install 3.12
pyenv global 3.12

# Install Poetry
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -

```

#### Option B: Windows (via WSL)

**This project is optimized for Linux environments.** If you are using Windows, it is highly recommended to use **WSL2 (Windows Subsystem for Linux)**. This gives you a real Ubuntu Linux environment inside Windows.

1. **Install WSL:**
Open PowerShell as Administrator and run:
```powershell
wsl --install

```


*Restart your computer if prompted.*
2. **Open Ubuntu:**
Search for "Ubuntu" in your Start menu and open it. You are now in a Linux terminal!
3. **Install Prerequisites (inside Ubuntu):**
```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Python build dependencies
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

# Install Pyenv
curl [https://pyenv.run](https://pyenv.run) | bash

# (Follow the on-screen instructions to add pyenv to your shell profile)

# Install Python 3.12
pyenv install 3.12
pyenv global 3.12

# Install Poetry
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -

```



### 2. Install Project Dependencies

Once your environment (Linux, Mac, or WSL) is ready, clone the repo and install dependencies:

```bash
# Clone the repository
git clone [https://github.com/yourusername/Prophet-Forecasting-For-Portfolio-Optimisation.git](https://github.com/yourusername/Prophet-Forecasting-For-Portfolio-Optimisation.git)
cd Prophet-Forecasting-For-Portfolio-Optimisation

# Install dependencies
make install-dev
# OR manually:
poetry install

```

### 3. Configure Environment Variables

You must provide your Supabase credentials for the project to run.

**Method 1: Using a `.env` file (Recommended for Local Dev)**
Create a `.env` file in the root directory:

```bash
cp .env.example .env  # If you have an example file
# OR create it manually
nano .env

```

Add your credentials inside `.env`:

```ini
SUPABASE_URL="your_supabase_url"
SUPABASE_KEY="your_supabase_anon_key"

```

**Method 2: Using Terminal Exports (Temporary)**

```bash
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_anon_key"

```

## Usage

### Basic Usage

```bash
poetry run python -m src.main

```

Or using the Makefile:

```bash
make run

```

### Configuration

Edit `src/settings.py` to customise:

* **Portfolio Tickers**: Modify `PORTFOLIO_TICKERS` list
* **Risk Aversion**: Adjust `RISK_AVERSION` (higher = more risk averse)
* **Minimum Allocation**: Change `MINIMUM_ALLOCATION` (minimum weight per asset)
* **Date Range**: Update `START_DATE` and `END_DATE` for historical data

Example:

```python
# src/settings.py
PORTFOLIO_TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
RISK_AVERSION = 3 
MINIMUM_ALLOCATION = 0.05 
START_DATE = "2024-01-01"

```

### Programmatic Usage

```python
from src.main import run_optimisation

result = run_optimisation(
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

print(f"Optimal Weights: {result['weights']}")
print(f"Predicted Returns: {result['predicted_returns']}")
print(f"Current Prices: {result['current_prices']}")
print(f"Prediction Date: {result['prediction_date']}")

```

### Running the Streamlit Dashboard

The Streamlit dashboard reads from Supabase to display historical predictions, portfolio weights, and performance metrics:

```bash
poetry run streamlit run src/streamlit_app.py

```

Or using the Makefile:

```bash
make dashboard

```

The dashboard allows you to:

* View portfolio weights and predictions for any date
* Analyze individual stock performance over time
* Compare predicted vs actual prices
* Track prediction accuracy metrics

**Note:** The dashboard requires Supabase to be configured and populated with data from previous optimization runs.

```

```
