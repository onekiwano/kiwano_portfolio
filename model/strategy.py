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
from pathlib import Path
import sys

from kiwano_portfolio.model.model_utils import (list2df, set_lookback, timeframe_to_str,
                                                check_name)
from kiwano_portfolio.strategy.run_strategy import run_strategy, fast_backtesting, layer_strategy_selector
from kiwano_portfolio.model.metrics import compute_generic_metric, compute_metric_selector

# Get path for saving files 
PATH_ROOT = Path(__file__).resolve().parents[1]  # Adjust the number if needed


class Strategy:
    def __init__(self, crypto_pair, timeframe, lookback=None):

        # Add data
        check_name(self, crypto_pair)
        # NB: Currently, strategies accept multiple inputs
        # /!\ But support only a single output!
        self.crypto_output = self.crypto_pairs[0]
        # By default, the first crypto_pair is used for output
        # You can change that by providing the argument: crypto_output, 
        # both in compute_metric, and select_strategy

        self.data = {crypto_pair: pd.DataFrame() for crypto_pair in self.crypto_pairs}
        self.size = 0

        # Parameter for the trade
        self.strategy_names = []
        self.timeframe = timeframe
        self.lookback = lookback
        self.W_lms = None
        self.frac_portfolio = {}
        self.error_manager = {'update': [],
                              'order': []}

    def update_data(self, crypto_pair=None, timeframe=None, lookback=None, end_date=None):
        '''
        Update data taken from candlestick. The dataframe 'data'
        will be appended by the last 'lookback' time steps in the past.
        If lookback is 1, then only the time step at current time is added.

        Parameters
        ----------
        crypto_pair : str
            Pair name of the asset CRYPTO_FIAT.
        timeframe : str
            Timeframe to consider to compute the candlestick.
        lookback : int
            Lookback in unit of candlestick timeframe.
        end_date: str
            Format is given like this: '2022-01-01-20-20-20' for year-month-day-hour-minute-second.
        '''
        # Get parameters
        if crypto_pair is not None:
            check_name(self, crypto_pair)
        if timeframe is not None:
            self.timeframe = timeframe
        if lookback is not None:
            self.lookback = lookback

        if end_date is not None:
            self.end_date = end_date
        else:
            self.end_date = None

        # Add data for all crypto_pairs
        for crypto_pair in self.crypto_pairs:
            current_data = self.data[crypto_pair]
            # Update data
            sys.path.append(self.api_location)
            from public_access import get_candlestick
            if self.api == 'crypto.com':
                # NB: we implemented it, but it is not yet fully functional hence the comment.
                # crypto_dic, _ = get_candlestick(self.crypto_pair, self.timeframe)
                # data = crypto_dic['result']['data']
                # lookback = set_lookback(self.lookback, self.timeframe)
                # size = len(data)
                # data = list(data)[size - lookback:size]
                # data = dic2df(data)
                raise NotImplementedError
            elif self.api == 'bybit':
                raise NotImplementedError
            elif self.api == 'binance':
                lookback = timeframe_to_str(self.lookback, self.timeframe)
                updated = False
                count = 0
                # ToDo: make a sleep of 5s, and a condition to exit on time
                while count < 5:  # Loop for overcoming potential connexion errors
                    try:  # Sometime you will get requests.exceptions.ReadTimeout     
                        data = get_candlestick(crypto_pair, self.timeframe, lookback, client=self.client,
                                               end_date=self.end_date)
                        updated = True
                        print(f"{crypto_pair} Data updated, ...")
                        break
                    except:
                        print(f"({count}) Order failed: {sys.exc_info()[0]} occured.")
                        print(sys.exc_info()[1])
                        self.error_manager['update'].append(sys.exc_info()[1])
                        count += 1
                data = list2df(data)

            if updated:
                # Avoid adding duplicate TimeStamps
                if len(self.data[crypto_pair]) > 0:
                    data = data.loc[data['TimeStamp'] > current_data['TimeStamp'].iloc[-1]]
                self.data[crypto_pair] = current_data.append(data, ignore_index=True)
                self.size = len(self.data[crypto_pair])

    def compute_metric(self, metric_name, lookback, window=0, func=None, **kwargs):
        '''
        Compute the metric given on 'data' from last time step 'lookback', 
        and update the dataframe 'data' with newly computed metric.

        Parameters
        ----------
        metric_name : str
            The name of metric to be computed. 
            See "compute_metric_selector" function in model_utils 
            for more info on the options:            
                - 'average'
                - 'log_average'
                - 'moving_average'
                - 'EMA'
                - 'rsi'
                - 'bollinger_bands'
                - 'smooth'
        lookback : int
            lookback in time, unit of candlestick timeframe, for wich the metric
            will be computed.
        window : int
            the integration window to compute some of the metrics. Default is 0.
        
        **kwargs:
            - 'option_average'
            - 'data_name': to specify on which data input to compute the metric
            - 'filter_name'
            - 'coeff'
            - 'crypto_output': to spcify which crypto to use for the output
        '''

        crypto_output = kwargs.get('crypto_output', None)
        # Select the output cryptos on which metrics are computed
        # NB: by default all crypto pairs are selected for computation
        if crypto_output is None:
            kwargs.update(dict(crypto_output=self.crypto_pairs))

        # Provide your own metric function 
        if func is not None:
            self.data = compute_generic_metric(self.data, lookback, metric_name, **kwargs)
        # Or utilizes a predefined metric
        elif func is None:
            # Check parameters
            lookback = set_lookback(lookback, self.timeframe)
            self.data = compute_metric_selector(self, metric_name, window, lookback, **kwargs)

    def compute_multiple_metric(self, **option_metrics):
        '''
        Compute multiple metrics with given windows, lookback and required arguments

        Parameters
        ----------
        option_metrics : dict
            must contan two keys:
                - 'metrics': list of str | list of metric names.
                - 'windows' : list of int | list of window sizes
                              if metric computation do not need a window, 
                              put it to 0.
                - lookback : int | See compute_metric.
        '''
        metrics = option_metrics['metrics']
        option_metrics.pop('metrics')
        windows = option_metrics['windows']
        option_metrics.pop('windows')
        data_names = option_metrics.get('data_names', None)
        lookbacks = option_metrics.get('lookbacks', None)

        # Check lookback argument
        if lookbacks is None:
            lookbacks = self.size * np.ones(len(metrics))
        else:
            option_metrics.pop('lookbacks')
            if len(lookbacks) == 1 and len(metrics) > 1:
                lookbacks = set_lookback(lookbacks, self.timeframe)
                lookbacks = lookbacks * np.ones(len(metrics))
        if len(windows) == 1 and len(metrics) > 1:
            windows = windows * np.ones(len(windows))
        if data_names is None:
            data_names = ['None'] * len(metrics)
        else:
            option_metrics.pop('data_names')
            if len(data_names) == 1 and len(metrics) > 1:
                data_names = data_names * len(metrics)

        # Compute metrics
        for i, (metric, window) in enumerate(zip(metrics, windows)):
            lookback = set_lookback(lookbacks[i], self.timeframe)
            self.compute_metric(metric, lookback, window, data_name=data_names[i], **option_metrics)

    def select_strategy(self, option_metrics=None, **strategy_kwargs):
        '''
        Select the strategy to be run, these strategies can
        be layered with several strategies applied one
        after the other.

        Parameters
        ----------
        option_metrics : TYPE, optional
            DESCRIPTION. The default is None.
        **strategy_kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.
        '''

        crypto_output = strategy_kwargs.get('crypto_output', None)
        if crypto_output is not None:
            self.crypto_output = crypto_output
            strategy_kwargs.pop('crypto_output')

        if option_metrics is not None:
            # Metric to compute
            if 'smooth' in option_metrics['metrics']:
                option_metrics.update(dict(smooth=True))
            self.option_metrics = option_metrics
            smooth = option_metrics.get('smooth', False)

            # Strategy to apply
            self.layered_strategy = []
            for layer, layer_kwargs in strategy_kwargs.items():
                layer_strategy_selector(self, layer_kwargs, smooth)
        else:
            self.option_metric = None

    def run_strategy(self, mode,
                     count_stop=np.inf,
                     plot_data=False,
                     plot_portfolio=False,
                     save=True,
                     timeframe_to_save=None,
                     reset=False,
                     key_stop=None,
                     force_sell=None,
                     debug=False,
                     **kwargs):
        '''
        Run the strategy with the given mode.

        Parameters
        ----------
        mode : str
            Can be either:
                - 'backtesting'
                - 'fast_backtesting'
                - 'livetesting'
                - 'livetrading.
            => These three modes are generating data at
            each time frame, and applying the strategy on the current timeframe. 
            As such it provides animated backtesting, and live modes.
                - 'fast_backtesting'
            => The fast version of the backtesting generates the whole data set 
            first, and then compute the strategy on it. To be able to run it, 
            your strategy needs to already loop on all of the timings. 
            If it only run on one timeframe, use 'backtesting' instead.
        count_stop : int, optional
            Counter stop for the live modes. The default is np.inf.
        plot_data : bool, optional
            If set to true, price data along with metric calculation
            are displayed over time. 
            The default is False.
        plot_portfolio : bool, optional
            If set to trye, portfolio wealth is displayed over time. 
            The default is False.
        save : bool, optional
            If true, save a csv for historical data and orders, 
            and save the portfolio historic. 
            The default is True.
        timeframe_to_save: int
            If 'save' is set to True, the number of timeframe to be saved.
            Default is None, which means all the dataframe is saved.
        reset : bool, optional
            If set to true, previous crypto data are reset. 
            The default is False.
        key_stop : str, optional
            The keayboard macro to stop the live (not working in back).
            The default is 'ctrl+q'.
        force_sell : str, optional
            The control key to force sell and quit the program.
            The default is 'ctrl+6'.
        debug : bool, optional
            If set to true, in livetesting, print the error between
            price and adjusted price. 
            The default is False.

        '''
        # Number of timeframe to save
        if timeframe_to_save is not None:
            self.timeframe_to_save = timeframe_to_save
        else:
            self.timeframe_to_save = self.size

        # Run the strategy
        if mode == 'fast_backtesting':
            fast_backtesting(self, plot_data, plot_portfolio, reset,
                             save, **kwargs)
        else:
            run_strategy(self, mode, count_stop, plot_data, plot_portfolio,
                         save, reset, key_stop, force_sell,
                         debug, **kwargs)

    def reset_data(self):
        '''
        Reset the data dictionnary storing the DataFrames of crypto and strategies
        '''
        self.data = {crypto_pair: pd.DataFrame() for crypto_pair in self.crypto_pairs}
