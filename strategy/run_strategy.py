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

from datetime import datetime

from kiwano_portfolio.API.binance.public_access import get_realtime_price
from kiwano_portfolio.model.model_utils import (smart_pause, set_lookback, prepare_data,
                                                timeframe_to_seconds, timestamp_str, next_delta_date,
                                                previous_delta_date, timing_context,
                                                )
from kiwano_portfolio.model.plot_functions import (prepare_live_plot, plot_live_data,
                                    prepare_live_portfolio, plot_live_portfolio)

from kiwano_portfolio.strategy.generic_strategies import (singlecrypto_strategy, multicrypto_strategy,
                                                          no_filter_portfolio, order66)

###############################################################################
# %% functions for loading the strategy
###############################################################################

# Dict. containing the function and name to be called inside the layer_strategy_selector
# NB: as of 2022-12-17 the model accept multilayer strategies, two types are possible:
# - hidden: the functions computing the Orders;
# -> Which receive only self.data as input
# - readout: the functions reading the Orders, filtering or making effective the buy and sell signals,
# -> Which receive the whole self as input
# When adding a new strategy here, you must provide the _type in supplementary argument 
# -> See layer_strategy_selector() for more details...
dic_strategy_function = {'generic_single': (singlecrypto_strategy, dict(_type='hidden')),
                         'generic_multiple': (multicrypto_strategy, dict(_type='hidden')),
                         'no_filter': (no_filter_portfolio, dict(_type='readout')),
                         }


# NB: If there are any other custom arguments you'd like to add, add them directly
# in the dictionary inside the tuple of the associated strategy name.

def layer_strategy_selector(self, layer_kwargs=None, smooth=False):
    if layer_kwargs is None:
        layer_kwargs = {}
    strategy_name = layer_kwargs.get('name', 'None')
    if strategy_name != 'None':
        layer_kwargs.pop('name')
        self.strategy_names.append(strategy_name)

    # Strategy first layer ----------------------------------------------------
    if strategy_name == 'generic':
        if len(self.crypto_pairs) == 1:
            strategy_name += '_single'
        else:
            strategy_name += '_multiple'

    # Get function for strategy
    strategy = dic_strategy_function.get(strategy_name, None)

    if strategy is not None:
        strategy_function = strategy[0]
        supplementary_arguments = strategy[1]  # Must be a dictionary
        layer_kwargs.update(supplementary_arguments)

        if layer_kwargs['_type'] == 'hidden':
            layer_kwargs.update(dict(smooth=smooth))
            layer_kwargs.update(dict(crypto_pairs=self.crypto_pairs))
            layer_kwargs.update(dict(crypto_output=self.crypto_output))

        self.layered_strategy.append((strategy_function, layer_kwargs))
    else:
        raise Exception(f"In method 'select_strategy':  argument {strategy_name} not supported")


###############################################################################
# %% functions for running the strategy
###############################################################################


def strategy_args_selector(self, l, mode, strategy_kwargs, counter=None):
    # If last layer
    strat_type = strategy_kwargs.get('_type', None)
    if strat_type == 'readout':
        # Check what price to use for the orders
        if 'live' in mode:
            self.data[self.crypto_output]['Adjusted Close'].iloc[-1] = float(get_realtime_price(self.crypto_output))
            price_label = 'Adjusted Close'
        else:
            price_label = 'Close'
        strategy_kwargs.update({'price_label': price_label})
        arg = self
    elif strat_type == 'hidden':
        strategy_kwargs.update({'count': counter})
        arg = self.data
    else:
        raise Exception(f'strat_type {strat_type} not provided, or not supported.')

    return arg, strategy_kwargs


def multi_layer_strategy(self, mode, counter=None, timer=None):
    # Engine of the run_strategy and fast_backtesting functions.

    for layer, (strategy_layer, strategy_kwargs) in enumerate(self.layered_strategy, start=1):

        # Argument selector
        arg, strategy_kwargs = strategy_args_selector(self, layer, mode,
                                                      strategy_kwargs,
                                                      counter)
        # Run strategy
        # If called from fast_backtesting()
        if 'fast' in mode:
            strategy_layer(arg, self.size, **strategy_kwargs)
        # If called from run_strategy()
        else:
            timer.set_name(f'strategy-l{layer}')

            strategy_layer(arg, **strategy_kwargs)

            # Update timer
            if timer is not None:
                if layer < len(self.layered_strategy):
                    self.timings['strategy'].append(timer.dt)
                    timer.reset()
            # ToDo: uniformize by adding timer in fast_backtesting() 
            # + directly adding lookback=self.size in arg.


def run_strategy(self, mode,
                 count_stop=np.inf,
                 plot_data=False,
                 plot_portfolio=False,
                 save=True,
                 reset=False,
                 key_stop=None,
                 force_sell=None,
                 debug=False,
                 **kwargs):
    # mode selector
    if mode == 'backtesting':
        # The delta t between each update of data
        delta_t_pause = 0  # (seconds) | No pause in the backtesting

        # Starting date and end date
        format = "%Y-%m-%d-%H-%M-%S"
        end_date = datetime.strptime(self.end_date, format)
        end_timestamp = datetime.timestamp(end_date)
        delta_date = timeframe_to_seconds(self.lookback)
        current_timestamp, _ = previous_delta_date(end_timestamp, delta_date)
        count_stop = set_lookback(self.lookback, self.timeframe)

    elif 'live' in mode:
        # The delta t between each update of data
        delta_t_pause = timeframe_to_seconds(self.timeframe)  # (seconds)

        # The delta t between each update of data
        if mode == 'livetrading':
            if not self.synchronize_wallet:
                err = f"In run_strategy: the mode {mode} must be used in pair with self.synchronize_wallet to True \n"
                err += "If you want to use livetrading, set synchronize_wallet to True at the initialization of the portfolio class \n"
                raise Exception(err)
        elif mode == 'livetesting':
            pass
        else:
            raise Exception(f"In run_strategy: the mode {mode} is not supported.")

        # Start date
        current_timestamp = datetime.timestamp(datetime.now())

    # Add lookback into the arguments of strategy
    for _, strategy_kwargs in self.layered_strategy:
        if strategy_kwargs.get('lookback', None) is None:
            strategy_kwargs.update({'lookback': 1})
    # NB: If your strategy needs a specific lookback, set it directly in the 
    # parameters, otherwise it is set to one by default.
    # lookback for readout layer must be kept to one to insure proper functioning,
    # It is advised to not change it otherwise orders can be made before the actual starting date.

    print('##########################################################')
    print(f'                       {mode}')
    print('##########################################################\n')

    # The delta t between each update of data
    delta_t = timeframe_to_seconds(self.timeframe)  # (seconds)

    # Figure
    if plot_data:
        order = kwargs.get('order', True)
        fig, ax, lines = prepare_live_plot(self.option_metrics, order=order)

    # Figure for portfolio
    if plot_portfolio:
        fig1, axs, lines1 = prepare_live_portfolio(self)

    # Reset and initialize data
    print('1. Prepare data')
    # Get the previous timestamp, as the current is not yet finished and will
    # produce an incomplete candle
    previous_timestamp, _ = previous_delta_date(current_timestamp, delta_t)
    end_date = timestamp_str(previous_timestamp)
    kwargs.update({'end_date': end_date})

    prepare_data(self, reset=reset, **kwargs)

    # Add columns
    self.data[self.crypto_output].insert(loc=6, column='Adjusted Close', value=np.nan)  # For real price value
    nbCol = len(self.portfolio.columns)
    self.portfolio.insert(loc=nbCol, column=f'cumret {self.crypto_output}', value=np.nan)

    # Compute metrics
    print('2. Compute metrics')
    self.compute_multiple_metric(**self.option_metrics)

    # Plot data            
    if plot_data:
        fig, ax, lines = plot_live_data(self.data[self.crypto_output],
                                        self.option_metrics,
                                        lines=lines, fig=fig, ax=ax,
                                        order=False)
    # Plot Portfolio
    if plot_portfolio:
        fig1, axs, lines1 = plot_live_portfolio(self, fig1, axs, lines1)

    ########################################################################################
    # Livestream loop        
    ########################################################################################
    timer = timing_context()  # Timer context manager
    self.timings = {'request': [],
                    'strategy': [],
                    'order': []}
    counter = 0
    condition_run = True
    while condition_run:

        ##############################################################
        # Wait until next iteration     
        ##############################################################
        current_timestamp = self.data[self.crypto_output]['TimeStamp'].values[-1] / 1000
        next_timestamp, _ = next_delta_date(current_timestamp, delta_t)
        condition_run = smart_pause(next_timestamp, delta_t=delta_t_pause,
                                    key=key_stop, force_sell=force_sell)

        # Order 66 kills all actions
        if condition_run == 66:
            self.layered_strategy = order66()

        ##############################################################
        # Update dataframes
        ##############################################################
        print(f't:{counter + 1}. Update data')
        end_date = timestamp_str(next_timestamp)
        timer.set_name('request')

        with timer:
            # Update data
            self.update_data(self.crypto_pairs, self.timeframe, lookback=1, end_date=end_date)

            # Update portfolio
            self.update_portfolio(lookback=1, last_value=True)
            self.compute_multiple_metric(**self.option_metrics)

        self.timings['request'].append(timer.dt)
        ##############################################################
        # Apply strategy
        ##############################################################
        if (condition_run is True) or (condition_run == 66):
            multi_layer_strategy(self, mode, counter, timer)

        # If there is an order, save the timing
        if self.data[self.crypto_output]['Orders'].values[-1] != 0:
            self.timings['order'].append(timer.dt)

        ##############################################################
        # Plot and save
        ##############################################################

        # Plot data            
        if plot_data:
            fig, ax, lines = plot_live_data(self.data[self.crypto_output],
                                            self.option_metrics,
                                            lines=lines, fig=fig, ax=ax,
                                            order=True)
        # Plot Portfolio
        if plot_portfolio:
            fig1, axs, lines1 = plot_live_portfolio(self, fig1, axs, lines1)

        # Display live stream
        print('------------------------------------------------')
        print(self.data[self.crypto_output].iloc[-1])
        print('------------------------------------------------')

        # Increment counter
        counter += 1

        # Conditions to terminate the loop 
        if counter >= count_stop:
            condition_run = False
        if condition_run == 66:
            condition_run = False

        # Save portfolio
        if save:
            self.save_portfolio(name=mode)

    # Compute average of timings
    for label, value in self.timings.items():
        self.timings[label] = [np.mean(value), np.std(value)]

    # Compute the difference between close and adjusted close
    close = self.data[self.crypto_output].loc[:, ['Close', 'Adjusted Close']]
    close = close.dropna()
    error = close['Close'].values - close['Adjusted Close'].values
    self.error_close = [np.mean(error), np.std(error)]
    if debug:
        print('Error between close and adjusted close:')
        print(self.error_close)


def fast_backtesting(self, plot_data=False, plot_portfolio=False, reset=True,
                     save=True, **kwargs):
    mode = 'fast_backtesting'

    fig, fig1 = None, None
    # Figure for data
    if plot_data:
        fig, ax, lines = prepare_live_plot(self.option_metrics, order=True)
        # Figure for portfolio
    if plot_portfolio:
        fig1, axs, lines1 = prepare_live_portfolio(self)

    # Reset and initialize data
    print('1. Prepare data')
    prepare_data(self, reset=reset, **kwargs)

    # Add columns        
    nbCol = len(self.portfolio.columns)
    self.portfolio.insert(loc=nbCol, column=f'cumret {self.crypto_output}', value=np.nan)

    # Compute metrics
    print('2. Compute metrics')
    self.compute_multiple_metric(**self.option_metrics)

    # Apply strategy
    print('3. Apply strategy')
    multi_layer_strategy(self, mode)

    # Plot data            
    if plot_data:
        fig, ax, lines = plot_live_data(self.data[self.crypto_output],
                                        self.option_metrics,
                                        lines, fig, ax,
                                        order=True)
    # Plot Portfolio
    if plot_portfolio:
        fig1, axs, lines1 = plot_live_portfolio(self, fig1, axs, lines1)

    # Save portfolio
    if save:
        strat_name = ''
        for name in self.strategy_names:
            strat_name += '_' + name

        save_figs = {}
        if not fig is None:
            save_figs.update(data=fig)
        if not fig1 is None:
            save_figs.update(portfolio=fig1)
        self.save_portfolio(name=mode + strat_name + '_', save_figs=save_figs)
