# Risk Engine
A Python-based quantitative risk analysis engine that computes portfolio risk metrics including volatility, Value at Risk (VaR), Sharpe ratio, Conditional VaR (ES), correlation matrix, beta, and portfolio volatility.  

## Installation

```bash
git clone https://github.com/dvidsalaz/risk-engine.git
cd risk-engine
pip install -r requirements.txt
```

## Usage

```python
# Calculating Volatility using Real Market Data

from risk_engine.data_loader import get_market_data

from risk_engine.risk_metrics import volatility

tickers = ['SPY', 'TLT', 'GLD']
prices, returns = get_market_data(tickers, '2020-01-01', '2025-01-01')

print(volatility(returns))
```

## Metrics
Metrics used to calculate portfolio risk.
```
Expected Return: Calculates the annual expected return
Volatility: Calculates an assets annualized volatility
Sharpe Ratio: Measures risk-adjusted returns
Historical VaR: Estimates the maximum expected loss at a given confidence level
CVaR (ES): Average of all returns that are worse than your VaR threshold
Correlation Matrix: How assets move together in relation to each other
Beta: Measures the sensitivity of an asset relative to market movement
Portfolio Expected Return: Calculates weighted average of each asset's expected return
Portfolio Volatility: Calculates how much the overall portfolio fluctuates
```
