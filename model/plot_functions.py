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
import pylab as plb
import matplotlib.pyplot as plt
import matplotlib.dates as mpl_dates
import matplotlib.ticker as ticker
from mpl_finance import candlestick_ohlc
import pandas as pd

from datetime import datetime

'''
This code contain all function needed to plot the price, 
strategy, and portfolio.
'''

###############################################################################
# %% Plot functions
###############################################################################

def prepate_options(option_metrics):
    metrics = np.copy(option_metrics['metrics'])
    windows = np.copy(option_metrics['windows'])

    if 'bollinger_bands' in metrics:
        idx = np.where(metrics == 'bollinger_bands')[0][0]
        metrics = list(metrics)
        windows = list(windows) 
        metrics.pop(idx)
        metrics.append('lower')
        metrics.append('upper')
        windows.append(windows[idx])
        windows.append(windows[idx])
        windows.pop(idx)

    return metrics, windows


def prepare_live_plot(option_metrics, order=False):
    metrics, windows = prepate_options(option_metrics)

    fig = plb.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    lines = {}
    for i, (metric, window) in enumerate(zip(metrics, windows)):
        # print(i, metric, window)
        if metric == 'average':
            line = ax.plot([], '.-', lw=3, label=metric + str(window))[0]
        elif metric == 'diff':
            continue
        else:
            line = ax.plot([], '--', lw=2, label=metric + str(window))[0]
        lines.update({metric + str(window): line})
    if order:
        fig, ax, lines = prepare_live_order(fig, ax, lines)
        
    ax.set_ylabel('Price ($)')
    ax.set_xlabel('TimeStamps')
    fig.legend(loc='upper right')
    fig.canvas.draw()
    plt.show(block=False)

    return fig, ax, lines


def prepare_live_portfolio(Portfolio):
    fig = plb.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)

    ## Plot
    lines = []
    lines.append(ax.plot([], '.-', color='green', linewidth=2)[0])
    ax.set_ylabel(f'{Portfolio.fiat_currency}', color="green", fontsize=15)
    ax.set_xlabel('TimeStamp', fontsize=15)
    ax1 = ax.twinx()
    lines.append(ax1.plot([], '--', color='orange', linewidth=2)[0])
    ax1.set_ylabel(f'{Portfolio.crypto_name()}', color="orange", fontsize=15)

    fig.canvas.draw()
    plt.show(block=False)
    axs = (ax, ax1)

    return fig, axs, lines


def prepare_live_order(fig, ax, lines, size_marker=20, cancel=False):
    buy = ax.plot([], marker='v', color='green', markersize=size_marker, linestyle="None")[0]  # Buy
    sell = ax.plot([], marker='^', color='red', markersize=size_marker, linestyle="None")[0]  # Sell
    lines.update({'buy': buy})
    lines.update({'sell': sell})
    if cancel:
        cancel_buy = ax.plot([], marker='v', color='gray', markersize=size_marker, linestyle="None")[0]  # Canceled buy
        cancel_sell = ax.plot([], marker='^', color='gray', markersize=size_marker, linestyle="None")[
            0]  # Canceled sell
        lines.update({'cancel_buy': cancel_buy})
        lines.update({'cancel_sell': cancel_sell})

    return fig, ax, lines


def plot_live_data(data, option_metrics, lines, fig, ax, length=60, order=False):
    metrics, windows = prepate_options(option_metrics)

    data['datetime'] = pd.to_datetime(data['Date'],
                                    format="%d.%m.%Y %H:%M:%S.%f")
    data.set_index('datetime', inplace=True)
    
    for i, (metric, window) in enumerate(zip(metrics, windows)):
        if metric == 'smooth':
            metric_name = metric + '_average0'  # Quick Fix
        elif metric == 'diff':
            continue
        else:
            metric_name = metric + str(window)
        lines[metric_name].set_data(data.index, data[metric_name])        

    if order:
        lines = plot_live_orders(data, lines)

    dstart = datetime.fromtimestamp(int(data['TimeStamp'].values[0]/1000))
    dend = datetime.fromtimestamp(int(data['TimeStamp'].values[-1]/1000))
    ax.set_xlim(dstart, dend)
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylim(0.999 * min(data['average0']), 1.001 * max(data['average0']))
    fig.autofmt_xdate()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.tight_layout()
    
    return fig, ax, lines


def plot_live_portfolio(Portfolio, fig, axs, lines):
    # Plot
    portfolio = Portfolio.portfolio  # Get the dataframe
    fiat_currency = portfolio[Portfolio.fiat_currency].values
    crypto_output = portfolio[Portfolio.crypto_name()].values
    
    portfolio['datetime'] = pd.to_datetime(portfolio['Date'],
                                           format="%d.%m.%Y %H:%M:%S.%f")
    portfolio.set_index('datetime', inplace=True)
    
    lines[0].set_data(portfolio.index, fiat_currency)
    lines[1].set_data(portfolio.index, crypto_output)

    dstart = datetime.fromtimestamp(int(portfolio['TimeStamp'].values[0]/1000))
    dend = datetime.fromtimestamp(int(portfolio['TimeStamp'].values[-1]/1000))
    axs[0].set_xlim(dstart, dend)
    axs[0].tick_params(axis='x', rotation=35)
    axs[0].set_ylim(0.99 * min(fiat_currency), 1.01 * max(fiat_currency))
    axs[1].set_ylim(min(crypto_output) - 0.01, 1.1 * max(crypto_output))

    fig.canvas.draw()
    fig.canvas.flush_events()

    return fig, axs, lines


def plot_live_orders(data, lines, cancel=False):
    # Load datas
    data['datetime'] = pd.to_datetime(data['Date'],
                                      format="%d.%m.%Y %H:%M:%S.%f")
    data.set_index('datetime', inplace=True)
    
    metric = data['average0'].values

    # Convert Orders into plots
    Orders = data['Orders']
    Buy_index = Orders == 1
    Sell_index = Orders == -1
    Cancel_sell = Orders == -2
    Cancel_buy = Orders == 2

    # Set data in plot
    lines['buy'].set_data(data.index[Buy_index], metric[Buy_index])
    lines['sell'].set_data(data.index[Sell_index], metric[Sell_index])
    if cancel:
        lines['cancel_buy'].set_data(data.index[Cancel_buy], metric[Cancel_buy])
        lines['cancel_sell'].set_data(data.index[Cancel_sell], metric[Cancel_sell])

    return lines


def plot_candlestick(data, y_axis='TimeStamp', options_dic=None, ax=None):
    ohlc = data.loc[:, [y_axis, 'Open', 'High', 'Low', 'Close']]
    if y_axis == 'Date':
        ohlc[y_axis] = pd.to_datetime(ohlc[y_axis])
        ohlc[y_axis] = ohlc[y_axis].apply(mpl_dates.date2num)
    ohlc = ohlc.astype(float)

    # Plot candlestick
    if ax is None:
        fig, ax = plt.subplots()

    candlestick_ohlc(ax, ohlc.values, width=1, colorup='green', colordown='red', alpha=0.8)

    # Setting labels & titles
    ax.set_xlabel(y_axis)
    ax.set_ylabel('Price')

    if options_dic is not None:
        metrics = options_dic['metrics']
        values = options_dic['values']
        for metric, value in zip(metrics, values):
            if metric == 'average':
                data_to_plot = data[metric + str(value)].values
            else:
                data_to_plot = data[metric + str(value)].values[value:]
            ax.plot(data['TimeStamp'].values[value:], data_to_plot,
                    '--', linewidth=2, label=metric + str(value))
        ax.legend(loc='upper left')

    return ax