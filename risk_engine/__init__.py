from risk_engine.data_loader import get_market_data
from risk_engine.risk_metrics import (
    hist_var,
    cvar,
    volatility,
    expected_return,
    sharpe_ratio,
    beta,
    correlation_matrix,
    portfolio_expected_return,
    portfolio_volatility,
    portfolio_returns,
    portfolio_var,
    portfolio_cvar,
    stress_test,
)
from risk_engine.visualization import plot_var_cvar