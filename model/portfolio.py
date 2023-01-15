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
import os
from pathlib import Path
import sys
import time
import importlib.util

from kiwano_portfolio.model.strategy import Strategy
from kiwano_portfolio.model.model_utils import set_lookback, evaluate_strategy, timeStructured

# Get path for files (#ToDo : make a module)
PATH_ROOT = Path(__file__).resolve().parents[1]  # Adjust the number if needed


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_last_quantity(Portfolio, currency, last_value):
    if last_value:
        if Portfolio.synchronize_wallet:
            dic_balance = Portfolio.synchronize_portfolio()
            last_quantity = dic_balance[currency]
        else:
            last_quantity = Portfolio.portfolio[currency].values[-1]
    else:
        last_quantity = Portfolio.budgets.get(currency)
    return last_quantity


class Portfolio(Strategy):
    def __init__(self, budgets_simulation=dict(),
                 fiat_currency='USDT', crypto_currencies=None, api=None,
                 synchronize_wallet=False, api_keys_location=''):
        '''
        This class manages the simulated and real portfolio, storing the history of transactions for all cryptos
        during strategy run time.

        Parameters
        ----------
        budgets_simulation : dict
            Amount of money you want to simulate your strategy with. Let's say you want an algorithm that trades between
            USDT and ETH, then you have to provide the amounts you start trading with in the simulation, such as
            budgets_simulation={'USDT': 1000, 'ETH': 1.2}. The default is dict().
        fiat_currency : str
            The crypto currency where you want to perform the buy and sell actions. The default is 'USDT'.
        crypto_currencies : list
            The currencies whose price will be passed to the chosen strategy.
        api : str
            The name of the crypto exchange that will provide the prices. For now only api='binance' is
            fully functional, but more are to come.
        synchronize_wallet : bool
            When livetrading, you need to set this as True, False otherwise.
        api_keys_location : str
            The location of your binance keys, if you did not want to store them in API/binance/config.py.
            The default is ''. The .py file containing the keys needs to have two string variables named
            api_key and api_secret.
        '''

        self.startTime = time.localtime()

        # Create portfolio
        self.portfolio = pd.DataFrame()
        self.fiat_currency = fiat_currency  # The currency obtained when selling
        if type(crypto_currencies) is str:
            self.crypto_currencies = [crypto_currencies]
        elif type(crypto_currencies) is list:
            self.crypto_currencies = crypto_currencies
        else:
            self.crypto_currencies = []

        # Set API
        self.api = api
        if self.api is not None:
            self.api_location = str(PATH_ROOT) + f'/API/{self.api}'
            # If no location is provided, the default location is used
            if api_keys_location == '':
                api_keys_location = os.path.join(self.api_location, 'config.py')

            if self.api == 'binance':
                sys.path.append(self.api_location)
                from binance.client import Client
                path, file = os.path.split(api_keys_location)
                keys_file = module_from_file(file, api_keys_location)

                self.client = Client(keys_file.api_key, keys_file.api_secret)
            else:
                raise NotImplementedError

        self.budgets = {}
        # Synchronize with a real wallet         
        # /!\ be carefull, this will trade with your actual money /!\
        self.synchronize_wallet = synchronize_wallet
        if (self.api is not None) and synchronize_wallet == True:
            dic_balance = self.synchronize_portfolio(self.api)
            self.budgets = dic_balance
        # No wallet synch. : it is a simulation
        elif not synchronize_wallet:
            self.budgets = budgets_simulation

        # Setting the initial budget for the portfolio
        if self.budgets.get(fiat_currency, None) is None:
            self.budgets.update({fiat_currency: 0})
        for crypto_currency in self.crypto_currencies:
            if self.budgets.get(crypto_currency, None) is None:
                self.budgets.update({crypto_currency: 0})
        for crypto_pair in self.budgets.keys():
            if (crypto_pair != fiat_currency) and (crypto_pair not in self.crypto_currencies):
                self.crypto_currencies.append(crypto_pair)

    def synchronize_portfolio(self, api=None):
        if api is not None:
            self.api = api
        # Select api
        if self.api == 'binance':
            sys.path.append(self.api_location)
            from private_access import get_balance
        else:
            raise NotImplementedError
        dic_balances = get_balance(client=self.client)
        return dic_balances

    def add_strategy(self, crypto_pair, timeframe, lookback=20, **kwargs):
        ## Add strategy to the portfolio
        Strategy.__init__(self, crypto_pair, timeframe, lookback, **kwargs)

    def update_portfolio(self, lookback=None, last_value=False):
        if lookback is not None:
            lookback = set_lookback(lookback, self.timeframe)
        else:
            lookback = set_lookback(self.lookback, self.timeframe)

        # ToDo: import_history from wallet
        if len(self.portfolio) != 0:
            last_value = True

        new_portfolio = pd.DataFrame(self.data[self.crypto_output][-lookback:], columns=['TimeStamp', 'Date'])
        # Update fiat
        last_fiat = get_last_quantity(self, self.fiat_currency, last_value)
        new_portfolio[self.fiat_currency] = last_fiat * np.ones(lookback)  # fiat budget
        new_portfolio[self.fiat_currency + '(transaction)'] = np.zeros(lookback)  # crypto bought history

        # Update cryptos
        for i, crypto_output in enumerate(self.crypto_currencies):
            last_crypto = get_last_quantity(self, crypto_output, last_value)
            new_portfolio[crypto_output] = last_crypto * np.ones(lookback)  # crypto budget
            new_portfolio[crypto_output + '(transaction)'] = np.zeros(lookback)  # crypto bought history

        # Avoid adding duplicate TimeStamps
        if len(self.portfolio) > 0:
            new_portfolio = new_portfolio.loc[new_portfolio['TimeStamp'] > self.portfolio['TimeStamp'].iloc[-1]]
        # Append portfolio
        self.portfolio = self.portfolio.append(new_portfolio, ignore_index=True)

        # self.portfolio = self.portfolio[~self.portfolio.TimeStamp.duplicated(keep='last')].sort_values('TimeStamp')

    def buy_action(self, i, buy_price, qty=None, prop_to_buy=None, fees=0,
                   buy_limit=np.inf):
        fees = fees / 100

        if prop_to_buy is not None:
            # Cost of transaction in fiat currency
            budget_spent = self.portfolio[self.fiat_currency].values[i - 1] * prop_to_buy
            # Amount of crypto bought
            crypto_bought = budget_spent / buy_price
            crypto_bought = crypto_bought * (1 - fees)

        elif qty is not None:
            crypto_bought = qty * (1 - fees)
            # Cost of transaction in fiat currency
            budget_spent = buy_price * qty
        crypto_owned_lastStep = self.portfolio[self.crypto_name()].values[i - 1]

        # Check crypto possession are not above the threshold
        if crypto_bought + crypto_owned_lastStep > buy_limit * 1:
            print(f'----- #{i} -----')
            wrng = f"WARNING: couldn't execute buying order due to reaching restricting fund limit for {self.fiat_currency}.\n"
            wrng += f"Current limit {buy_limit} is lower than given buy order {crypto_bought + crypto_owned_lastStep}"
            print(wrng)

            return False

        # Check sufficient funds, and update portfolio
        fiat_owned_lastStep = self.portfolio[self.fiat_currency].values[i - 1]
        if fiat_owned_lastStep - budget_spent < 0:  # Transaction not allowed
            print(f'----- #{i} -----')
            wrng = f"WARNING: couldn't execute buying order due to insufficient fund in {self.fiat_currency}.\n"
            wrng += f"Current budget {fiat_owned_lastStep} is lower than given buy order {budget_spent}"
            print(wrng)

            return False

        # Transaction is allowed
        else:
            if self.synchronize_wallet:
                # Select api
                if self.api == 'binance':
                    sys.path.append(self.api_location)
                else:
                    raise NotImplementedError
                from private_access import make_order
                # Make order
                order_passed = False
                count = 0
                if isinstance(self.crypto_output, str):
                    crypto_output = self.crypto_output
                elif isinstance(self.crypto_output, list):
                    crypto_output = self.crypto_output[0]
                print('crypto_bought', crypto_bought)
                while count < 5 and not order_passed:
                    try:
                        order = make_order(symbol=crypto_output, side='BUY', quantity=crypto_bought,
                                           client=self.client)
                        order_passed = True
                    except:
                        count += 1
                        print(f"({count}) Order failed: {sys.exc_info()[0]} occured.")
                        print(sys.exc_info()[1])
                        self.error_manager['order'].append(sys.exc_info()[1])
                if not order_passed:
                    return False

                budget_spent = float(order['cummulativeQuoteQty'])
                crypto_bought = float(order['executedQty']) - float(order['fills'][0]['commission'])
                self.data[self.crypto_output]['Adjusted Close'].iloc[i] = float(order['fills'][0]['price'])

            print('BOUGHT', crypto_bought)
            # Remove amount spent form current fiat portfolio
            self.portfolio[self.fiat_currency].loc[i:] = fiat_owned_lastStep - budget_spent
            # Add amount of crypto to your portfolio
            self.portfolio[self.crypto_name()].loc[i:] = crypto_owned_lastStep + crypto_bought
            self.portfolio[self.crypto_name() + '(transaction)'].loc[i] = crypto_bought
            self.portfolio[self.fiat_currency + '(transaction)'].loc[i] = - budget_spent

            return crypto_bought

    def sell_action(self, i, sell_price, qty=None, prop_to_sell=None, fees=0):
        fees = fees / 100
        if prop_to_sell is not None:
            crypto_sold = self.portfolio[self.crypto_name()].values[
                              i - 1] * prop_to_sell  # The amount of cryto that is sold
        elif qty is not None:
            crypto_sold = qty
        budget_earned = crypto_sold * sell_price * (1 - fees)  # The amount of fiat currency earned
        # Check sufficient funds, and update portfolio
        fiat_owned_lastStep = self.portfolio[self.fiat_currency].values[i - 1]
        crypto_owned_lastStep = self.portfolio[self.crypto_name()].values[i - 1]
        if crypto_owned_lastStep - crypto_sold < 0:  # Transaction not allowed
            wrng = f"WARNING: couldn't execute selling order due to insufficient fund in {self.crypto_output}.\n"
            wrng += f"Current budget {crypto_owned_lastStep} is lower than given sell order {crypto_sold}"
            print(wrng)

            return False

        else:  # Transaction is allowed
            if self.synchronize_wallet:
                if self.api == 'binance':
                    sys.path.append(self.api_location)
                else:
                    raise NotImplementedError
                from private_access import make_order
                # Make order
                order_passed = False
                count = 0

                if isinstance(self.crypto_output, str):
                    crypto_output = self.crypto_output
                elif isinstance(self.crypto_output, list):
                    crypto_output = self.crypto_output[0]
                while count < 5 and not order_passed:
                    try:
                        order = make_order(symbol=crypto_output, side='SELL', quantity=crypto_sold,
                                           client=self.client)
                        order_passed = True
                    except:
                        count += 1
                        print(f"({count}) Order failed: {sys.exc_info()[0]} occured.")
                        print(sys.exc_info()[1])
                        self.error_manager['order'].append(sys.exc_info()[1])
                if not order_passed:
                    return False

                budget_earned = float(order['cummulativeQuoteQty']) - float(order['fills'][0]['commission'])
                crypto_sold = float(order['executedQty'])
                self.data[self.crypto_output]['Adjusted Close'].iloc[i] = float(order['fills'][0]['price'])

            print('SOLD', crypto_sold)
            self.portfolio[self.fiat_currency].loc[
            i:] = fiat_owned_lastStep + budget_earned  # Add amount spent form current fiat portfolio
            self.portfolio[self.crypto_name()].loc[
            i:] = crypto_owned_lastStep - crypto_sold  # Remove amount of crypto to your portfolio
            self.portfolio[self.crypto_name() + '(transaction)'].loc[i] = - crypto_sold
            self.portfolio[self.fiat_currency + '(transaction)'].loc[i] = budget_earned

            return crypto_sold

    def crypto_name(self, crypto_output=None):
        if crypto_output is None:
            return self.crypto_output.split(self.fiat_currency)[0]
        else:
            crypto_output.split(self.fiat_currency)[0]

    def reset_portfolio(self):
        self.portfolio = pd.DataFrame()

    def save_portfolio(self, name='', folder='result', save_figs={}):

        summary = evaluate_strategy(self, _print=True, error=True)
        # Check if result folder exists, create it otherwise
        results_dir = os.path.join(PATH_ROOT, folder)  # Concatenate path
        if not os.path.isdir(results_dir):  # Check directory is not existing
            os.makedirs(results_dir)  # make that directory

        # Get all transactions
        buys_sells_portfolio = self.portfolio.loc[(self.portfolio[self.crypto_name() + '(transaction)'] != 0) | (
                self.portfolio[self.fiat_currency + '(transaction)'] != 0)]

        # Save summary in a csv file
        name_summary = timeStructured(self.startTime) + '_' + name + '_summary_portfolio'
        filepath_summary = results_dir + '/' + name_summary
        if len(buys_sells_portfolio) > 0:
            print('SAVE :', filepath_summary)
            index_low = np.clip(len(buys_sells_portfolio) - self.timeframe_to_save, 0, np.inf)
            index_low = int(index_low)
            df_to_save = buys_sells_portfolio[index_low:len(buys_sells_portfolio)]
            df_to_save.to_csv(filepath_summary + '.csv')

        # Save error in a csv file
        if len(self.error_manager['order']) > 0:
            name_error = timeStructured(self.startTime) + '_' + name + '_error_API'
            filepath_error = results_dir + '/' + name_error
            error_manager = self.error_manager.copy()
            error_manager.pop('update')
            pd.DataFrame(error_manager).to_csv(filepath_error + '.csv')

        # Save summary
        with open(filepath_summary + '.txt', 'w') as f:
            f.write(summary)

        for n, f in save_figs.items():
            figpath = os.path.join(results_dir, name + '_' + n + '.png')
            f.savefig(figpath)
