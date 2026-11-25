# Risk Engine

A Python-based quantitative risk analysis engine that computes portfolio risk metrics including volatility, Value at Risk (VaR), Sharpe ratio, Conditional VaR (ES), correlation matrix, beta, and portfolio volatility using historical returns from LSEG.

## Metrics

Metrics used to calculate portfolio risk.

```
Expected Return: Annualizes the arithmetic mean of daily returns
Volatility: Annualizes the standard deviation of daily returns
Sharpe Ratio: Measures risk-adjusted returns
Historical VaR: Estimates a one-day loss threshold from the historical return quantile
CVaR (ES): Averages historical losses that exceed the VaR threshold
Correlation Matrix: How assets move together in relation to each other
Beta: Measures the sensitivity of an asset relative to market movement
Portfolio Expected Return: Calculates weighted average of each asset's expected return
Portfolio Volatility: Calculates how much the overall portfolio fluctuates
```

## Installation

```bash
git clone https://github.com/dvidsalaz/risk-engine.git
cd risk-engine
pip install -e .
```

This installs the package and the dependencies for the cached example (`pandas`, `numpy`, and `matplotlib`). For live LSEG data, install the optional dependency with `pip install -e ".[lseg]"` and open an authorized session with `lseg.data.open_session()`.

## Usage

Run this example from the repository root. Cached mode loads every date and ticker in `data/sample_returns.csv`. The ticker and date arguments apply to live requests.

```python
# Calculating Non-Parametric VaR and cVaR (ES) using Real Market Data

from risk_engine.data_loader import get_market_data
from risk_engine.risk_metrics import hist_var, cvar
from risk_engine.visualization import plot_var_cvar

tickers = ['SPY', 'TLT.O', 'GLD', 'TSLA.O']
returns = get_market_data(tickers, '2020-01-02', '2024-12-31', use_cache=True)

print(hist_var(returns))  # 95% VaR
print(cvar(returns))  # Expected Shortfall

# Visualization

var = hist_var(returns['SPY'])
spy_cvar = cvar(returns['SPY'])

tsla_var = hist_var(returns['TSLA.O'])
tsla_cvar = cvar(returns['TSLA.O'])

plot_var_cvar(returns=returns['SPY'], var=var, cvar=spy_cvar)
plot_var_cvar(returns=returns['TSLA.O'], var=tsla_var, cvar=tsla_cvar)
```

> **Note:** Tickers use LSEG RIC (Reuters Instrument Code) conventions. Suffixes like `.O` denote the listing exchange (e.g. NASDAQ) where applicable — this is why `TLT.O` and `TSLA.O` have suffixes while `SPY` and `GLD` don't.

## Sample Output

In the sample output, the daily returns cover **1,258 trading days** from January 2, 2020, through December 31, 2024. The 95% one-day historical VaR was **1.92% for SPY** and **6.28% for TSLA**. Each asset had 63 days when losses exceeded its VaR threshold. Across those days, CVaR (Expected Shortfall) was **3.19% for SPY** and **9.09% for TSLA**.

## Limitations

- Historical VaR and ES use the observed distribution of one-day returns. Results depend on the sample window and may miss stress scenarios absent from the data. The example reports in-sample estimates, and the repository does not include an out-of-sample backtest.
- Portfolio VaR and ES use weighted daily returns with fixed weights, which assumes daily rebalancing. They reflect the assets’ observed daily co-movement. Portfolio volatility uses one covariance matrix estimated across the sample, so it does not show how correlations change over time.
- The Sharpe ratio uses a 4% annual risk-free rate by default. The annualized historical mean is a sample estimate and can vary substantially across windows.
