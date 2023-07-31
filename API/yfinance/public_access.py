# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:04:57 2023

@author: ManuMan
"""

import yfinance
from datetime import datetime, timedelta
from kiwano_portfolio.model.model_utils import set_lookback

def date_round(date):
    return datetime(year=date.year,
                    month=date.month,
                    day=date.day)


def get_candlestick(symbol, timeframe, lookback, end_date=None, original_lookback=None):
    
    # Set lookback
    print(timeframe, lookback, end_date)
    lookback_int = set_lookback(lookback, timeframe, 'int')
    if original_lookback is None:
        original_lookback = lookback_int
    
    
    if 'S&P500' in symbol:
        symbol = "^GSPC"
    elif 'BTC' in symbol:
        symbol = 'BTC-USD'
        # lookback_int += 1

        
    if end_date in [None, 'now', 'Now', 'None']:
        
        # Format lookback
        if 'd' in lookback:
            lookback = lookback.split('day')[0]
            unit_lookback = 'd'
        elif 'h' in lookback:
            lookback = lookback.split('hour')[0]
            unit_lookback = 'h'
        elif 'm' in lookback:
            lookback = lookback.split('min')[0]
            unit_lookback = 'm'
        else:
            raise NotImplementedError
        lookback = str(lookback_int) + unit_lookback

        df = yfinance.download(tickers=symbol, interval=timeframe,
                               period=lookback)
    else:
        end = datetime(*[int(s) for s in end_date.split('-')])
        utc_delta = date_round(datetime.utcnow()) - date_round(datetime.now())
        end = end + utc_delta
        if 'd' in lookback:
            unit_lookback = 'd'
            start = end - timedelta(days=lookback_int)
        elif 'h' in lookback:
            unit_lookback = 'h'
            start = end - timedelta(hours=lookback_int) 
        elif 'm' in lookback:
            unit_lookback = 'm'
            start = end - timedelta(minutes=lookback_int) 
        else:
            raise NotImplementedError
    
        _format = '%Y-%m-%d'
        end = date_round(end)
        end = end.strftime(_format)
        start = date_round(start)
        start = start.strftime(_format)

        df = yfinance.download(tickers = symbol, start = start, 
                               end = end, interval = timeframe)
        if len(df) == 0:
            raise Exception('Symbol not found.')
    
    if len(df) != original_lookback:
        '''
        In order to get the exact same number of data points as indicated in 
        the lookback period, it is needed to call the function more than once,
        since sometimes there is no data in a given day.
        ''' 
        missing_element = original_lookback - len(df)
        # print(original_lookback, missing_element)
        new_lookback = lookback_int + missing_element
        new_lookback = set_lookback(new_lookback, timeframe, output_type='str', unit_lookback=unit_lookback)
        # print(new_lookback)
        df = get_candlestick(symbol, timeframe, new_lookback, end_date=end_date, original_lookback = original_lookback)
    else:
        # df = df.reset_index()
        timestamps = [date.timestamp() for date in df.index]
        df.insert(0, "TimeStamp", timestamps, False)
        df = df.reset_index()
        
    return df

# timeframe = '1d'
# lookback = '20days'
# data = get_candlestick('BTC-CUSD', timeframe, lookback, end_date='2022-07-12')
