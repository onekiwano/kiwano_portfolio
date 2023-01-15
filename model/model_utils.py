# Copyright 2023 The Kiwano Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import pandas as pd
import numpy as np

import keyboard
from datetime import datetime, timedelta
import time

'''
This file contains all the functions needed for the strategy and portfolio classes.
'''

###############################################################################
# %% utility, timing dictionary and pandas
###############################################################################

def timestamp_str(timestamp):
    date = datetime.fromtimestamp(timestamp)
    time = datetime(year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute)
    time_str = str(time)
    date = time_str.split(' ')[0]
    timing = time_str.split(' ')[1]
    timing = timing.replace(':', '-')
    return date+'-'+timing
    
def previous_delta_date(timestamp, delta_t):
    date = datetime.fromtimestamp(timestamp)
    time = datetime(year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute)
    delta_time = timedelta(minutes=delta_t / 60)

    return datetime.timestamp(time - delta_time), delta_time

def next_delta_date(timestamp, delta_t):
    date = datetime.fromtimestamp(timestamp)
    time = datetime(year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour,
                    minute=date.minute)
    delta_time = timedelta(minutes=delta_t / 60, seconds=1)

    return datetime.timestamp(time + delta_time), delta_time

def smart_pause(timestamp, delta_t, extra_wait=0, step=0.06, key=None, force_sell=None):
    current_timestamp = datetime.timestamp(datetime.now())  # Need to remove one minute to actually fit data timstamps
    print('--------------------------------------------------')
    text = "Wait until next timestep... "
    if key is not None:
        text += f" (press {key} to quit)"
    print(text)
    print('--------------------------------------------------')
    # Quick fix, to account for different timings in binance and Cypto.com
    next_timestamp, _ = next_delta_date(timestamp, delta_t)
    if current_timestamp >= next_timestamp:
        delta_t = 2*delta_t # Empirical
    next_timestamp, _ = next_delta_date(timestamp, delta_t+extra_wait)
    
    while True:
        # Check timestamps
        current_timestamp = datetime.timestamp(datetime.now())
        if current_timestamp >= next_timestamp:
            return True
        time.sleep(step)
        
        # Check escape key
        if key is not None:
            if keyboard.is_pressed(key):
                print(f"{key} pressed, ending loop")
                return False
        if force_sell is not None:
            if keyboard.is_pressed(force_sell):
                print(f"{key} Order 66: ending loop and selling assets")
                return 66

class timing_context():
    """Context manager to measure how much time was spent in the target scope."""

    def __init__(self, name='', allow_print=True):
        self.t0 = None
        self.dt = None
        self.allow_print = allow_print
        self.set_name(name)
        
    def set_name(self, name):
        self.name = name
    
    def reset(self):
        # Force reset before end of the loop
        self.__exit__()
        self.t0 = time.perf_counter()
    
    def __enter__(self):
        self.t0 = time.perf_counter()

    def __exit__(self, type=None, value=None, traceback=None):
        self.dt = (time.perf_counter() - self.t0) # Store the desired value.
        if self.allow_print is True:
            print(f"Scope {self.name} took {self.dt*1000: 0.1f} milliseconds.")
    
def timing(function , *args, **kwargs):
    start = time.time()
    output = function(*args, **kwargs)
    elapsed_time = time.time() - start
    return output, elapsed_time

def set_lookback(lookback, timeframe, output_type='int'):
    '''
    Parameters
    ----------
    loockback : int or str
        the time to look back in the past, if it is an int, it
        will be considered to be the output. If it is an int, it will
        be converted into a number of index.
    timeframe : str
        The timeframe of the candlestick.

    Returns
    -------
    loockback: int 
        the number of time steps in the past to look back.

    '''
    if isinstance(lookback, int) or isinstance(lookback, np.float64):
        if output_type == 'int':
            return int(lookback)
    if isinstance(lookback, str):
        if output_type == 'int':
            timeframe_second = timeframe_to_seconds(timeframe)
            lookback_second = timeframe_to_seconds(lookback)
            return int(lookback_second / timeframe_second)
    else:
        raise Exception(f"In 'set_lookback': lookback argument can't be of type {type(lookback)}. Must be int or str.")

def timeStructured(timing=None):
    
    if timing is None:
        named_tuple = time.localtime()  # get struct_time
    else:
        named_tuple = timing
    time_string = time.strftime("%Y-%m-%d--%H-%M-%S-", named_tuple)
    random_string = ''.join([str(r) for r in np.random.choice(10, 4)])
    time_string += '-' + random_string
    return time_string


def timeframe_to_str(lookback, timeframe):
    if not isinstance(lookback, str):
        tf_index = timeframe_to_seconds(timeframe)

        lookback = lookback * tf_index / 60

        return str(f'{int(lookback)}m')
    else:
        return lookback

def timeframe_to_seconds(timeframe):
    if 'm' in timeframe.lower():
        unit_conversion = 60
    if 'h' in timeframe.lower():
        unit_conversion = 60 * 60
    if 'd' in timeframe.lower():
        unit_conversion = 24 * 60 * 60

    value_str = ''
    for c in timeframe:
        if c.isdigit():
            value_str += c

    value_int = int(value_str)
    return value_int * unit_conversion  # seconds


def list2df(crypto_list):
    dic = {'Date': [datetime.fromtimestamp(data[0] / 1000) for data in crypto_list],
           'TimeStamp': [np.float64(data[0]) for data in crypto_list],
           'Open': [np.float64(data[1]) for data in crypto_list],
           'High': [np.float64(data[2]) for data in crypto_list],
           'Low': [np.float64(data[3]) for data in crypto_list],
           'Close': [np.float64(data[4]) for data in crypto_list],
           'Volume': [np.float64(data[5]) for data in crypto_list]}

    return pd.DataFrame(dic)

def dic2df(crypto_dic):
    newDic = {'Date': [datetime.fromtimestamp(dic['t'] / 1000) for dic in crypto_dic],
              'TimeStamp': [dic['t'] for dic in crypto_dic],
              'Open': [dic['o'] for dic in crypto_dic],
              'High': [dic['h'] for dic in crypto_dic],
              'Low': [dic['l'] for dic in crypto_dic],
              'Close': [dic['c'] for dic in crypto_dic],
              'Volume': [dic['v'] for dic in crypto_dic]}
    return pd.DataFrame(newDic)



###############################################################################
# %% functions for model
###############################################################################

    
def check_name(self, crypto_pair):
    if type(crypto_pair) is str:
        self.crypto_pairs = [crypto_pair]
    elif type(crypto_pair) is list:
        self.crypto_pairs = crypto_pair
    else:
        raise Exception('in check_name: crypto_pair must be str or list of str')
       
def prepare_data(strategy, reset=True, **kwargs):
    # Prepare data and check eventual change of parameters
    crypto_pair = kwargs.get('crypto_pair', None)
    last_value = kwargs.get('last_value', False)
    if crypto_pair is not None:
        strategy.crypto_pairs = crypto_pair
    timeframe = kwargs.get('timeframe', None)
    if timeframe is not None:
        strategy.timeframe = timeframe
    lookback = kwargs.get('lookback', None)
    if lookback is not None:
        strategy.lookback = set_lookback(lookback, strategy.timeframe)
    if reset:
        strategy.reset_data()
        strategy.reset_portfolio()
    if kwargs.get('end_date', None) is not None:
        strategy.end_date = kwargs.get('end_date', None)
    # print('prepare', strategy.end_date, strategy.lookback)
    strategy.update_data(strategy.crypto_pairs, timeframe=strategy.timeframe, 
                         lookback=strategy.lookback, end_date=strategy.end_date)
    strategy.update_portfolio(strategy.lookback, last_value=last_value)  # Done into update_data

def evaluate_strategy(Portfolio, _print=True, error=False):
    # Get all transactions
    buys_sells_portfolio = Portfolio.portfolio.loc[
        (Portfolio.portfolio[Portfolio.crypto_name() + '(transaction)'] != 0) |
        (Portfolio.portfolio[Portfolio.fiat_currency + '(transaction)'] != 0)]
    nb_transaction = len(buys_sells_portfolio)
    usd_trans = buys_sells_portfolio[Portfolio.fiat_currency + '(transaction)']
    earnings = [usd_trans.values[i] + usd_trans.values[i + 1] for i in range(len(usd_trans) - 1)]
    earnings = np.array(earnings)
    win = earnings[earnings > 0]
    lost = earnings[earnings < 0]
    total = len(win) + len(lost)

    winrate = len(win) / total if not total == 0 else 0

    ## Load portfolio
    portfolio = Portfolio.portfolio
    data = Portfolio.data.copy()
    data = data[Portfolio.crypto_output]
    try:
        data = data.drop('Adjusted Close', axis=1)
    except:
        pass
    data = data.dropna()

    fiat_currency = portfolio[Portfolio.fiat_currency].values
    crypto_output = portfolio[Portfolio.crypto_name()].values

    ## Total earning
    # Last time step possession
    fiat_last_asset = fiat_currency[-1]
    crypto_last_asset = crypto_output[-1]
    total_last_fiat = fiat_last_asset + crypto_last_asset * data['average0'].values[-1]

    # First time step possession
    fiat_first_asset = fiat_currency[0]
    crypto_first_asset = crypto_output[0]
    total_first_fiat = fiat_first_asset + crypto_first_asset * data['average0'].values[0]

    # Compute total earnings or loss
    total_earning_fiat = total_last_fiat - total_first_fiat
    crypto_hold = fiat_first_asset / data['average0'].values[0]
    earning_hold = crypto_hold * data['average0'].values[-1]
    trade_versus_hold = np.round(total_last_fiat - earning_hold, 1)
    cumret = total_last_fiat / total_first_fiat
    
    # Update portfolio
    portfolio[f'cumret {Portfolio.crypto_output}'].loc[len(portfolio)-1] = cumret
    
    # Dates
    date0 = data['Date'].values[0]
    date1 = data['Date'].values[-1]

    # Summary text
    text = '##################################################### \n'
    text += f'#      Summary of strategy ({Portfolio.crypto_output})   \n'
    text += '#################################################### \n'
    text += f'Starting date: {pd.Timestamp(date0)} \n'
    text += f'Time stopped:  {pd.Timestamp(date1)} \n'
    text += f"Total number of transaction: {nb_transaction} \n"
    text += f"Cumulative return: {cumret:.3f}; Winrate: {winrate:.3f} \n"
    text += '---------------------------------------------------- \n'
    text += f"Current wealth (fiat + crypto): {total_last_fiat:.3f} in fiat {Portfolio.fiat_currency} \n"
    text += f"Total earning: {total_earning_fiat:.3f} in fiat {Portfolio.fiat_currency} \n"
    text += '---------------------------------------------------- \n'
    text += "             Holding versus trading \n"
    text += '---------------------------------------------------- \n'
    text += f"At times t=0, buying {crypto_hold:.3f} worth of {Portfolio.crypto_output}, \n"
    text += f"For a price of {crypto_hold * data['average0'].values[0]:.3f} in fiat {Portfolio.fiat_currency} \n"
    text += f"Holding it until the last time step, it has the value: {crypto_hold * data['average0'].values[-1]:.3f} {Portfolio.fiat_currency} \n"
    text += f"=> Trade versus hold: {trade_versus_hold:.3f} in fiat {Portfolio.fiat_currency} \n"
    if error:
        text += '---------------------------------------------------- \n'
        text += f"Error API| update:{len(Portfolio.error_manager['update'])}|order:{len(Portfolio.error_manager['order'])} \n"

    if _print:
        print(text)
    
    return text
