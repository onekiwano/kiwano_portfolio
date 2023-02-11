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



import argparse, sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from kiwano_portfolio.model import Portfolio

'''
Run this file the first time to test that the model works
'''

# =============================================================================
# Dummy strategies
# =============================================================================


def buy_wait(data, **kwargs):
    count = kwargs['count']
    if count % 2:
        return np.ones(len(data))
    else:
        return np.zeros(len(data))

def sell_wait(data, **kwargs):
    count = kwargs['count']
    if count % 2:
        return -np.ones(len(data))
    else:
        return np.zeros(len(data))

def buy_sell_wait(data, **kwargs):
    print(kwargs)
    count = kwargs['count']
    if count % 2:
        return np.ones(len(data))
    elif count % 3:
        return -np.ones(len(data))
    else:
        return np.zeros(len(data))

    
def select_strategy(option):
    if option=='buy_wait':
        return buy_wait
    elif option=='sell_wait':
        return sell_wait
    elif option=='buy_sell_wait':
        return buy_sell_wait

crypto_names = lambda crypto_pairs,fiat: [crypto_pair.split(fiat)[0] for crypto_pair in crypto_pairs]

# =============================================================================
# Test functions
# =============================================================================

def test_backtesting(option):
    
    timeframe = '1m'
    lookback = '1 h'
    option_metrics = {'metrics': ['average'],
                      'windows': [0],
                      'data_names': ['None']}
    
    strategy_kwargs = {'layer1':{'name':'generic',
                                 'func_order':select_strategy(option),
                                 '_type':'hidden',
                                 },
                       'layer2':dict(name='no_filter',
                                     prop_to_sell=1.0,
                                     prop_to_buy=1.0, 
                                     fees=0.0,),
                       }

    ## Create portfolio
    my_portfolio = Portfolio(fiat_currency='USDT',
                             budgets_simulation={'USDT': 100000,},
                             crypto_currencies=crypto_names(crypto_pair,'USDT'),
                             api='binance', synchronize_wallet=False)

    ## Strategy on BTC
    my_portfolio.add_strategy(crypto_pair, timeframe=timeframe,
                              lookback=lookback)
    my_portfolio.update_data(end_date=end_date)
    # Backtesting
    my_portfolio.select_strategy(option_metrics, **strategy_kwargs)
    my_portfolio.run_strategy(method='backtesting', plot_data=True, plot_portfolio=True, 
                              reset=False, save=True, timeframe_to_save=30,)
    portfolio = my_portfolio.portfolio
    crypto_df = my_portfolio.data
    
    return crypto_df, portfolio
    
def test_livetesting(option):
    
    timeframe = '1m'
    lookback = '1 h'
    crypto_name = 'SOLUSDT'
    option_metrics = {'metrics': ['average'],
                      'windows': [0],
                      'lookbacks': [lookback],
                      'data_names': ['None']}
    
    strategy_kwargs = {'layer1':{'name':'generic',
                                 'func_order':select_strategy(option),
                                 '_type':'hidden',
                                 'lookback':3,  # Below, Orders will be Nan's+
                                 },
                       'layer2': dict(name='no_filter',
                                      lookback=1,
                                      prop_to_buy=1,
                                      prop_to_sell=1,
                                      fees=0.0, )
                       }
    
    
    ## Create portfolio
    my_portfolio = Portfolio(fiat_currency='USDT', 
                             budgets_simulation={'USDT': 100000},
                             crypto_currencies=crypto_names(crypto_pair,'USDT'),
                             api='binance', synchronize_wallet=False)

    ## Strategy on BTC
    my_portfolio.add_strategy(crypto_pair, timeframe=timeframe,
                              lookback=lookback)
    my_portfolio.update_data()
    # Backtesting
    my_portfolio.select_strategy(option_metrics, **strategy_kwargs)
    my_portfolio.compute_metric('average', lookback=lookback, window=0, func=None)
    my_portfolio.run_strategy(method='livetesting', plot_data=True, plot_portfolio=True, reset=False,
                              key_stop='ctrl+m', force_sell='ctrl+6')
    portfolio = my_portfolio.portfolio
    crypto_df = my_portfolio.data
    
    return crypto_df, portfolio
    
def test_livetrading(option):
     # livetrading with class and different option of dummy strategies
    
    timeframe = '1m'
    lookback = '1 h'
    option_metrics = {'metrics': ['average'],
                      'windows': [0],
                      'lookbacks': [lookback],
                      'data_names': ['None']}
    
    strategy_kwargs = {'layer1': {'name':'generic',
                                  'func_order':select_strategy(option),
                                  'lookback':3,
                                  '_type':'hidden',
                                  },  # Below, Orders will be Nan's+
                       'layer2': dict(name='no_filter',
                                      lookback=1,
                                      prop_to_buy=1,
                                      prop_to_sell=1,
                                      fees=0.0, )
                       }
    
    # Portfolio
    my_portfolio = Portfolio(fiat_currency='USDT',
                             api='binance', 
                             synchronize_wallet=True)
    
    my_portfolio.add_strategy(crypto_pair=crypto_pair, timeframe=timeframe,
                              lookback=lookback)
    my_portfolio.update_data(crypto_pair, timeframe, lookback)
    
    # Strategy
    my_portfolio.select_strategy(option_metrics, **strategy_kwargs)
    my_portfolio.run_strategy(method='livetrading', count_stop=5, 
                              plot_data=True, plot_portfolio=True,
                              key_stop='ctrl+h', force_sell='ctrl+6')
    
    portfolio = my_portfolio.portfolio
    crypto_df = my_portfolio.data
    
    return crypto_df, portfolio

crypto_pair = ['BTCUSDT', 'ETHUSDT']
end_date = '2019-06-08-01-01-01'  
def main():
    args = parse_args()
    
    if args.test == 'all':
        tests = ['backtesting', 'livetesting', 'livetrading']
    else:
        tests = [args.test]
    
    error_count = 0
    for test in tests:
        print('-----------------------------')
        print(f'Test function: {test}')
        print('-----------------------------')
        if test == 'backtesting':
            try:
                test_backtesting(args.strategy)
                print('Done')
            except:
                print(f'Error in {test}:', sys.exc_info()[0], 'occured.')
                print('>', sys.exc_info()[1])
                error_count += 1
        elif test == 'livetesting':
            try:
                test_livetesting(args.strategy)
                print('Done')
            except:
                print(f'Error in {test}:', sys.exc_info()[0], 'occured.')
                print('>', sys.exc_info()[1])
                error_count += 1
        elif test == 'livetrading':
            try:
                test_livetrading(args.strategy)
                print('Done')
            except:
                print(f'Error in {test}:', sys.exc_info()[0], 'occured.')
                print('>', sys.exc_info()[1])
                error_count += 1
    print('===================================')           
    print('Test file completed')
    print('with a total number of error:', error_count)
    print('===================================')

def parse_args():
    """Parse args."""
    # Initialize the command line parser
    parser = argparse.ArgumentParser()
    
    # Read command line argument
    parser.add_argument('--strategy', default='buy_sell_wait', type=str, help='Code modality',
                        choices=['buy_sell_wait', 'buy_wait', 'sell_wait'])
    parser.add_argument('--test', default='livetrading', type=str, help='Code modality',
                        choices=['all', 'backtesting', 'livetesting', 'livetrading'])
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main()
