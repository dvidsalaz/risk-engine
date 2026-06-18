# Risk Engine
A Python-based quantitative risk analysis engine that computes portfolio risk metrics including volatility, Value at Risk (VaR), Sharpe ratio, Conditional VaR (ES), correlation matrix, beta, and portfolio volatility.  

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

## Installation

```bash
git clone https://github.com/dvidsalaz/risk-engine.git
cd risk-engine
pip install -r requirements.txt
```

## Usage

```python
# Calculating Non-Parametric VaR and cVaR (ES) using Real Market Data

from risk_engine.data_loader import get_market_data
from risk_engine.risk_metrics import hist_var, cvar
from risk_engine.visualization import plot_var_cvar

tickers = ['SPY', 'TLT', 'GLD', 'TSLA']
prices, returns = get_market_data(tickers, '2020-01-01', '2025-01-01')

print(hist_var(returns))  # 95% VaR
print(cvar(returns))  # Expected Shortfall

# Visualization

var = hist_var(returns['SPY']) 
spy_cvar = cvar(returns['SPY'])

tsla_var = hist_var(returns['TSLA'])
tsla_cvar = cvar(returns['TSLA'])

plot_var_cvar(returns=returns['SPY'], var=var, cvar=spy_cvar)
plot_var_cvar(returns=returns['TSLA'], var=tsla_var, cvar=tsla_cvar)s


```

## Sample Output
<img width="500" height="343" alt="image" src="https://github.com/user-attachments/assets/16872cdd-7c76-4dc5-9af8-cca649aeee66" />
<img width="500" height="343" alt="image" src="https://github.com/user-attachments/assets/d55cb189-8497-414d-ba26-ad8b6278da21" />

> [!NOTE]
> This project uses historical VaR rather than parametric VaR.
>  Historical data captures skewness and excess kurtosis that a normal distribution assumption would fail to capture.

In the data provided, Figure 1.1, captures the VaR and Conditional Value at Risk (CVaR/ES) for SPY. In the graph, the red line represents VaR at a 95% confidence level. The daily losses are not expected to exceed **1.93%**. The green dashed line represents CVaR. On days where losses breach the threshold, the average loss is **3.20%**. In the shaded band between VaR and CVaR, it highlights the severity of tail events. The wider the band, the more dangerous the tail.

In the TSLA graph, Figure 1.2, at a 95% confidence level, daily losses are not expected to exceed **6.28%**. When daily losses breach the VaR threshold, the average loss is **9.09%**. Compared to the figure 1.1, TSLA's band is significantly wider. Indicating a fatter tail and a higher susceptibility to extreme loss events. 

When comparing the two, SPY's tighter band reflects the diversification benefit of holding an index compared to a single volatile equity. 


