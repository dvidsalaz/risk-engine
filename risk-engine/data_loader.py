import yfinance as yf
import pandas as pd


def get_market_data(tickers, start_date, end_date):
    """
    Docstring for data_loader

    Downloads Adjusted Closing Prices and computes Daily Returns.

    Args
    ----------

    tickers::list[str]
        List of  Ticker Symbols (e.g. '['SPY', 'TLT', 'GLD']).
    start_date::str
        Start date in 'YYYY-MM-DD' format
    end_date::str
        End date in 'YYYY-MM-DD' format.

    Returns
    ----------
    prices : DataFrame
        Adjusted close prices for each ticker
    returns : dataFrame
    Daily returns (decimal form) for each ticker

    """

    data = yf.download(tickers, start=start_date, end=end_date , auto_adjust=False)

    prices = data['Adj Close']

    returns = prices.pct_change().dropna()

    return(prices, returns)

