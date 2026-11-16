import pandas as pd
import lseg.data as ld


def get_market_data(tickers, start_date, end_date, use_cache=False, cache_path='data/sample_returns.csv'):
    """
    
    **Before running any data, open an LSEG session**
    
    Docstring for get_market_data

    Computes Daily Returns by pulling TR.TotalReturn directly from LSEG (already dividend-adjusted, already a return) and converts units to decimal form.

    Args
    ----------

    tickers : list[str]
        List of Ticker Symbols (e.g. ['SPY', 'TLT.O', 'GLD']).
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    ----------
    
    returns : DataFrame
        Daily returns (decimal form) for each ticker

    """
    if use_cache:
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    df = ld.get_data(
        universe=tickers,
        fields=['TR.TotalReturn.date', 'TR.TotalReturn'],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            "Frq": "D"
        }
    )
    df['Decimal Return'] = df['Total Return'] / 100
    wide_df = df.pivot(index='Date', columns='Instrument', values='Decimal Return')
    return wide_df