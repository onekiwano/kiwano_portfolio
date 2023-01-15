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

import numpy as np
import pandas as pd
from ta.trend import ema_indicator
from ta.momentum import rsi

'''
All functions to compute metric on the price curves.
You can either use one of the predetermined metric here, or your own 
metric by using the function: compute_generic_metric()
'''


#####################################################################################
# %% Function for loading metric 
#####################################################################################


def compute_metric_selector(self, metric_name, window, lookback, **kwargs):
    data_name = kwargs.get('data_name', 'None')
    if metric_name == 'average':
        option_average = kwargs.get('option_average', 'oc')
        # print(len(self.data), lookback, option_average)
        data = average_candlestick(self.data, lookback, option_average=option_average, **kwargs)

    elif metric_name == 'log_average':
        option_average = kwargs.get('option_average', 'oc')
        data = average_candlestick(self.data, lookback,
                                   option_average=option_average,
                                   log_option=True, **kwargs)

    elif metric_name == 'moving_average':
        if window > lookback:
            lookback = window + 1

        # Compute moving average
        data = moving_average_candlestick(self.data, lookback, window=window, **kwargs)

    elif metric_name == 'EMA':
        if window > lookback:
            lookback = window + 1
        if kwargs.get('data_name', None) is not None:
            kwargs.pop('data_name')
        data = ema_metric(self.data, lookback,
                          data_name=data_name, window=window, **kwargs)

    elif metric_name == 'rsi':
        if window > lookback:
            lookback = window + 1
        data = rsi_candlestick(self.data, lookback,
                               data_name=data_name, window=window, **kwargs)

    elif metric_name == 'bollinger_bands':
        if window > lookback:
            lookback = window + 1
        coeff = kwargs.get('coeff', 2)
        option_average = kwargs.get('option_average', 'c')
        data = moving_average_candlestick(self.data, lookback, window=window)
        data = bollinger_bands(data, lookback,
                               window=window,
                               coeff=coeff,
                               option_average=option_average,
                               **kwargs)

    elif metric_name == 'smooth':
        if window > lookback:
            lookback = window + 1
        data_name = kwargs.get('data_name', 'average0')
        filter_name = kwargs.get('filter_name', 'LMS')

        if filter_name == 'LMS':
            if self.W_lms is not None:
                kwargs = {'W_lms': self.W_lms}
            else:
                kwargs = {}
            data, self.W_lms = smooth_data(self.data, lookback, data_name=data_name,
                                           filter_name=filter_name, window=window,
                                           **kwargs)
        else:
            data = smooth_data(self.data, lookback, data_name=data_name,
                               filter_name=filter_name, window=window, **kwargs)

    elif metric_name == 'cross_ma':
        if window > lookback:
            lookback = window + 1
        data = cross_moving_average(self.data, lookback, window=window,
                                    metric_names=data_name, **kwargs)

    return data


#####################################################################################
# %% Utility for metric computation
#####################################################################################

# Decorator
def split_data(func):
    def inner(full_datas, lookback, **kwargs):
        # Selected column for computation
        crypto_pairs = [col for col in full_datas.keys()]
        crypto_output = kwargs.get('crypto_output', None)
        if crypto_output is None:
            crypto_outputs = crypto_pairs
        else:
            crypto_outputs = crypto_output
            kwargs.pop('crypto_output')
        if isinstance(crypto_outputs, str):
            crypto_outputs = [crypto_outputs]

        # Shrink data to keep lookback
        shrink_datas = {}
        for crypto_pair, full_data in full_datas.items():
            size = len(full_data)
            full_data = full_data.iloc[np.arange(size - lookback, size)]
            shrink_datas.update({crypto_pair: full_data})

        # Compute function and update data
        if len(crypto_pairs) == 1:  # Unique input crypto, unique output
            shrink_data = shrink_datas[crypto_outputs[0]]
            full_data = full_datas[crypto_outputs[0]]
            full_data = func(shrink_data, full_data, **kwargs)
            full_datas[crypto_outputs[0]] = full_data
        else:  # Multiple input crypto, multiple output
            for crypto_output in crypto_outputs:
                shrink_data = shrink_datas[crypto_output]
                full_data = full_datas[crypto_output]
                full_datas[crypto_output] = func(shrink_data, full_data, **kwargs)

        return full_datas

    return inner


def update_dataframe(new_data, full_data, column_name, live=False):
    if isinstance(new_data, dict):  # Check if multiple crypto in data
        crypto_pairs = [crypto for crypto in new_data.keys()]
        crypto = crypto_pairs[0]
        full_data = update_dataframe(new_data[crypto], full_data[crypto], column_name, live)
        return full_data
    elif isinstance(full_data, dict):  # Check if multiple crypto in data
        crypto_pairs = [crypto for crypto in full_data.keys()]
        full = full_data[crypto_pairs[0]].copy()
        full_data = update_dataframe(new_data, full, column_name, live)
        return full_data
    if isinstance(new_data, pd.core.frame.DataFrame):
        new_data = pd.DataFrame(new_data, columns=[column_name])
    elif isinstance(new_data, pd.core.series.Series):
        new_data = pd.DataFrame(new_data)

    # Update dataframe
    if column_name not in full_data.columns:
        full_data[column_name] = np.nan
        # full_data = pd.concat([full_data, new_data], axis=1, join='outer')

    # Add new values to datraframe 
    new_data = new_data.values if isinstance(new_data, pd.core.frame.DataFrame) else new_data
    new_data = list(filter(lambda x: np.isnan(x) == False, new_data))
    if live:  # If live mode
        new_data = new_data[-1]  # If so, only the last element will be added
        try:
            len(new_data)
        except:
            new_data = [new_data]

    # print(len(full_data)-len(new_data), len(full_data), len(new_data), column_name)
    full_data.loc[len(full_data) - len(new_data):len(full_data), column_name] = new_data

    return full_data


@split_data
def compute_generic_metric(data, full_data, metric_name, func, **kwargs):
    # Compute metric with given function
    new_metric = func(data, **kwargs)

    return full_data


#####################################################################################
# %% Predefined metrics
#####################################################################################


@split_data
def cross_moving_average(data, full_data, metric_names, window=0, **kwargs):
    # Compute metric with given function
    data['trend'] = np.nan
    data.loc[data[metric_names['slow']] > data[metric_names['fast']], 'trend'] = 1  # Up trend
    data.loc[data[metric_names['slow']] < data[metric_names['fast']], 'trend'] = 0  # Down trend
    data.loc[1:, 'change_trend'] = np.diff(data['trend'])

    # Update dataframe
    full_data = update_dataframe(data['trend'], full_data, f'cross_ma{window}')
    full_data = update_dataframe(data['change_trend'], full_data, f'change_trend')

    return full_data


@split_data
def average_candlestick(data, full_data, option_average='oc', log_option=False, **kwargs):
    # Average of candlestick
    if option_average == 'ohlc':
        average = data['Open'] + data['High'] + data['Low'] + data['Close']
        average /= 4
    elif option_average == 'oc':
        average = data['Open'] + data['Close']
        average /= 2
    elif option_average == 'hl':
        average = data['High'] + data['Low']
        average /= 2
    elif option_average == 'c':
        average = data['Close']
    elif option_average == 'o':
        average = data['Open']

    if log_option:
        average = np.log10(average)

    # Update dataframe
    full_data = update_dataframe(average, full_data, 'average0')

    return full_data


@split_data
def bollinger_bands(data, full_data, window=20, coeff=2.5, option_average='c'):
    # Compute standard deviation
    std = data['Close'].rolling(window).std()

    # Compute bollinger bands
    Upper = data[f'moving_average{window}'] + coeff * std
    Lower = data[f'moving_average{window}'] - coeff * std

    # Update data
    full_data = update_dataframe(Upper, full_data, f'upper{window}')
    full_data = update_dataframe(Lower, full_data, f'lower{window}')

    return full_data


@split_data
def rsi_candlestick(data, full_data, data_name='Close', window=2):
    # Compute RSI
    rsi_data = rsi(data[data_name], window)

    # Update data
    full_data = update_dataframe(rsi_data, full_data, f'rsi{window}')

    return full_data


@split_data
def moving_average_candlestick(data, full_data, window, data_name='average0'):
    # Compute moving average on average0
    average = data[data_name]
    mvg_averages = average.rolling(window=window).mean()

    # Update dataframe
    full_data = update_dataframe(mvg_averages, full_data, f'moving_average{window}')

    return full_data


@split_data
def ema_metric(data, full_data, data_name, window):
    # Compute EMA with given window on data
    ema = ema_indicator(data[data_name], window=window)

    # Update dataframe
    full_data = update_dataframe(ema, full_data, f'EMA{window}')

    return full_data


@split_data
def differentiate_metric(data, full_data, data_name, order=1):
    # Compute derivative on data
    derivative = np.diff(np.log10(data[data_name].values), order)

    # Update dataframe
    full_data = update_dataframe(derivative, full_data, f'diff{order}_{data_name}')

    return full_data


@split_data
def smooth_data(data, full_data, data_name, filter_name='LMS', window=10, **kwargs):
    # Compute filter on data
    if filter_name == 'savitzky_golay':
        smoothed_data = savitzky_golay(data[data_name].values, window, **kwargs)
    if filter_name == 'LMS':
        W_lms = kwargs.get('W_lms', np.zeros(window))
        lms = LMS(W_lms)
        smoothed_data = lms.filter_data(data[data_name].values, window)

    # Update dataframe
    full_data = update_dataframe(smoothed_data, full_data, 'smooth_' + data_name)

    if filter_name == 'LMS':
        return full_data, lms.Wt
    else:
        return full_data


####################################################################################
# %% Filters
####################################################################################

def savitzky_golay(y, window_size, order=3, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the lookback of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


class LMS:
    """ lms = LMS( Wt, damp=.5 )  Least mean squares adaptive filter
    in:
        Wt: initial weights, e.g. np.zeros( 33 )
        damp: a damping factor for swings in Wt
 
    # for t in range(1000):
 
    yest = lms.est( X, y [verbose=] )
    in: X: a vector of the same lookback as Wt
        y: signal + noise, a scalar
        optional verbose > 0: prints a line like "LMS: yest y c"
    out: yest = Wt.dot( X )
        lms.Wt updated
 
    How it works:
    on each call of est( X, y ) / each timestep,
    increment Wt with a multiple of this X:
        Wt += c X
    What c would give error 0 for *this* X, y ?
 
        y = (Wt + c X) . X
        =>
        c = (y  -  Wt . X)
            --------------
               X . X
 
    Swings in Wt are damped a bit with a damping factor a.k.a. mu in 0 .. 1:
        Wt += damp * c * X
 
    Notes:
        X s are often cut from a long sequence of scalars, but can be anything:
        samples at different time scales, seconds minutes hours,
        or for images, cones in 2d or 3d x time.
 
"""

    # See also:
    #     http://en.wikipedia.org/wiki/Least_mean_squares_filter
    #     Mahmood et al. Tuning-free step-size adaptation, 2012, 4p
    # todo: y vec, X (Wtlen,ylen)

    # ...............................................................................
    def __init__(self, Wt, damp=.5):
        self.Wt = np.squeeze(getattr(Wt, "A", Wt))  # matrix -> array
        self.damp = damp

    def estimate(self, X, y, verbose=0):
        X = np.squeeze(getattr(X, "A", X))
        yest = self.Wt.dot(X)
        c = (y - yest) / X.dot(X)
        # clip to cmax ?
        self.Wt += self.damp * c * X
        if verbose:
            print("LMS: yest %-6.3g   y %-6.3g   err %-5.2g   c %.2g" % (
                yest, y, yest - y, c))
        return yest

    def filter_data(self, data, window_size):
        yests = [np.nan] * window_size
        for i, t in enumerate(range(window_size, len(data))):
            X = data[t - window_size:t]
            y = data[t]  # predict
            yest = self.estimate(X, y, verbose=(t % 10 == 0))
            if i > 10:
                yests += [yest]
            else:
                yests += [np.nan]
        return yests
