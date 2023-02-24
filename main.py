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

import argparse, json
import warnings
from kiwano_portfolio import Portfolio

warnings.filterwarnings("ignore")


crypto_pair = ['SOLUSDT', 'BTCUSDT', 'ETHUSDT', 'FTMUSDT', 'ADAUSDT']


def dummy_strategy(dfs, crypto_output='BTCUSDT', **kwargs):
    dfs = {'input_prices': list(dfs.values()), 'output_prices': [dfs[crypto_output]]}

    # Create an algorithm that processes several crypto price movements and proposes a trade on one of them

    # The strategy should give back a +1 for a buy, a -1 for a sell and a 0 for no action
    import numpy as np
    order = np.random.choice([-1, 0, 1])

    return [order]


def get_portfolio():
    args = parse_args()
    live = args.mode in ['livetrading', 'livetesting']

    lookbacks = {} if live else {'lookbacks': [args.lookback]}
    option_metrics = {'metrics': ['average'], 'windows': [0], 'data_names': ['None'], **lookbacks}

    fees = args.fees if not args.mode == 'livetrading' else 0.0
    l1_lookback, l2_lookback = ({}, {}) if not live else ({'lookback': 7000}, {'lookback': 1})
    strategy_kwargs = {
        'layer1': dict(name='generic', func_order=dummy_strategy, args=args, **l1_lookback),
        'layer2': dict(name='no_filter', prop_to_buy=1, prop_to_sell=1, fees=fees, **l2_lookback),
        'crypto_output': 'BTCUSDT'
    }

    if args.mode == 'livetrading':
        pkwargs = {'synchronize_wallet': True}
    else:
        pkwargs = {
            'budgets_simulation': {args.fiat_name: args.initial_fiat, args.crypto_pair[:3]: args.initial_crypto}}

    # Portfolio
    my_portfolio = Portfolio(
        fiat_currency=args.fiat_name, api=args.api, api_keys_location=args.api_keys_location, **pkwargs
    )
    my_portfolio.add_strategy(crypto_pair=crypto_pair, timeframe=args.timeframe, lookback=args.lookback)

    if args.mode in ['backtesting', 'fast_backtesting']:
        my_portfolio.update_data(end_date=args.end_date)
    else:
        my_portfolio.update_data(crypto_pair, args.timeframe, args.lookback)

    # Strategy
    my_portfolio.select_strategy(option_metrics, **strategy_kwargs)
    my_portfolio.run_strategy(mode=args.mode, plot_data=args.plot, plot_portfolio=args.plot, fees=fees)

    return my_portfolio


def parse_args():
    """Parse args."""
    # Initialize the command line parser
    parser = argparse.ArgumentParser()
    # Read command line argument

    parser.add_argument('--timeframe', default='1m', type=str, help='Minimal time resolution')
    parser.add_argument('--end_date', default='2021-01-12-00-00-00', type=str,
                        help='End time for backtesting. E.g. 2022-01-01-20-20-20 or now')  # None
    # parser.add_argument('--end_date', default='now', type=str, help='Start time for backtesting')
    parser.add_argument('--lookback', default='5 days', type=str, help='Backtesting period')  # 5
    parser.add_argument('--api', default='binance', type=str, help='API')
    parser.add_argument('--api_keys_location', default='', type=str, help='API keys location')
    parser.add_argument('--comments', default='',
                        type=str, help='Extra variable arguments to pass to the trading function')
    parser.add_argument('--crypto_pair', default='BTCUSDT', type=str, help='Crypto to use')
    parser.add_argument('--initial_fiat', default=10000, type=int, help='Initial deposit in fiat')
    parser.add_argument('--fiat_name', default='USDT', type=str, help='Fiat to use')
    parser.add_argument('--initial_crypto', default=0, type=int, help='Initial deposit in crypto')
    parser.add_argument('--fees', default=0.0, type=float, help='Fees applied if back or live testing')
    parser.add_argument('--plot', action='store_true', help='Plot performance')
    parser.add_argument('--mode', default='livetesting', type=str, help='Code modality',
                        choices=['fast_backtesting', 'backtesting', 'livetesting', 'livetrading'])
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))

    return args


if __name__ == '__main__':
    my_portfolio = get_portfolio()
    portfolio = my_portfolio.portfolio
    crypto_df = my_portfolio.data
