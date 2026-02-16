import numpy as np

# annual_expected_returns


def expected_return(asset_returns):
    """
    Calculated annual expected return

    Parameters
    ----------
    asset_returns : pd.Series or pd.DataFrame
        Daily returns for asset(s)

    Returns
    -------
    float or pd.Series
        Annualized expected return
    """

    annual_returns = asset_returns.mean() * 252
    return annual_returns

# annual_volatility


def volatility(asset_returns):
    """
    Calculates annualized volatility

    Parameters
    ----------
    asset_returns : pd.Series or pd.DataFrame
        Daily returns for asset(s)

    Returns
    -------
    float or pd.Series
        Annualized volatility
    """

    vol = asset_returns.std() * np.sqrt(252)
    return vol


# sharpe_ratio


def sharpe_ratio(asset_returns, risk_free_rate=0.04):
    """
    Calculates the Sharpe Ratio

    Parameters
    ----------

    asset_returns : pd.Series or pd.DataFrame
        Daily returns for asset(s)
    risk_free_rate : float, default=0.04
        Risk-free rate for Sharpe ratio calculation

    Returns
    -------
    float or pd.Series
        Sharpe Ratio
    """
    sr = (expected_return(asset_returns) - risk_free_rate) / volatility(asset_returns)
    return sr

# value at risk (VaR)


def hist_var(asset_returns, confidence_level=0.95):
    """
    Calculates VaR using historical or empirical data

    Parameters
    ----------

    asset_returns : pd.Series or pd.DataFrame
        Daily returns for asset(s)
    confidence_level : float, default=0.95
        Confidence level for VaR

    Returns
    -------
    float or pd.Series
        Historical VaR

    """

    var = asset_returns.quantile(1-confidence_level)
    return var

# Conditional Value at Risk (CVaR)


def cvar(asset_returns, confidence_level=0.95):
    """
    Calculation for Conditional VaR
        The average of all returns that are worse than your VaR threshold

    Parameters
    ----------
    asset_returns : pd.Series or pd.DataFrame
        Daily returns for asset(s)
    confidence_level : float, default=0.95
        Confidence level for CVaR

    Returns
    -------
    float or pd.Series
        Conditional VaR

    """

    var_threshold = hist_var(asset_returns, confidence_level)
    bad_returns = asset_returns[asset_returns < var_threshold]
    return bad_returns.mean()

# Correlation Matrix


def correlation_matrix(asset_returns):
    """
    Correlation Matrix
        How assets move together

    Parameters
    ----------

    asset_returns : pd.Series or pd.DataFrame
        Daily Returns of asset(s)

    Returns
    -------

    pd.DataFrame or 2D table
        Correlation matrix shows relationship between assets

    """
    return asset_returns.corr()

# beta


def beta(asset_returns, market_returns):
    """
    Measures the sensitivity of an asset relative to market movement
    
    Parameters
    ----------

    asset_returns: pd.Series or pd.DataFrame
        Daily returns of asset(s)
    market_returns: pd.Series
        Daily returns of market benchmark (e.g. SPY)

    Returns
    -------

    float or pd.Series
        Beta
    """

    covariance = asset_returns.cov(market_returns)
    m_var = market_returns.var()
    return covariance / m_var

# portfolio expected returns


def portfolio_expected_return(asset_returns, weights):
    """
    Calculates weighted portfolio expected return

    Parameters
    ----------
    
    asset_returns : pd.Series or pd.DataFrame
        Daily return of asset(s)
    weights : np.ndarray
        Portfolio weights for each asset (must sum to 1.0)

    Returns
    -------

    float or pd.Series
        Portfolio expected return
    """

    individual_returns = expected_return(asset_returns)
    portfolio_returns = (individual_returns * weights).sum()
    return portfolio_returns

# portfolio volatility


def portfolio_volatility(asset_returns, weights):
    """
    Calculates portfolio volatility using covariance matrix
    
    Parameters
    ----------

    asset_returns : pd.Series or pd.DataFrame
        Daily returns of asset(s)
    weights : np.ndarray
        Portfolio weights for each asset (must sum to 1.0)

    Returns
    -------

    float
        portfolio volatility

    """
    cov_matrix = asset_returns.cov() * 252
    return np.sqrt(weights.T @ cov_matrix @ weights)
