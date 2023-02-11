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

from kiwano_portfolio.model.metrics import update_dataframe


'''
This file contain all relevant functions to interface your strategy
with the exchange_interface. 
For single crypto strategy use: singlecrypto_strategy()
For multiple crypto strategy use: multiplecrypto_strategy()
'''


############################################################################
# %% Usefull functions
############################################################################

def split_data_strat(func_strategy):
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
            skrink_data = full_data.iloc[np.arange(size - lookback, size)]
            shrink_datas.update({crypto_pair: skrink_data})

        # Compute function and update data
        if len(crypto_pairs) == 1:  # Unique input crypto, unique output
            full_data = func_strategy(shrink_datas, full_datas, **kwargs)
            full_datas[crypto_outputs[0]] = full_data
        elif (len(crypto_pairs) > 1) and (len(crypto_outputs) == 1):  # Multiple input crypto, unique output
            full_data = func_strategy(shrink_datas, full_datas, crypto_output=crypto_outputs[0],
                                      **kwargs)
            full_datas[crypto_outputs[0]] = full_data
        else:  # Multiple input crypto, multiple output
            for crypto_output in crypto_outputs:
                full_datas[crypto_output] = func_strategy(shrink_datas, full_datas, crypto_output=crypto_output,
                                                          **kwargs)

        return full_datas

    return inner


def open_position(full_data):
    cond = False
    if 'Orders' in full_data.columns:
        # For livetesting
        last_orders = full_data['Orders']
        last_orders = last_orders[last_orders != 0].dropna()
        if len(last_orders) > 0:
            last_order = last_orders.values[-1]
            position = True if last_order == 1 else False
            if position:
                last_buys = last_orders[last_orders == 1]
                idx_last_buy = last_buys.index[-1]
                return position, idx_last_buy

    if not cond:
        position = False
        idx_last_buy = None

    return position, idx_last_buy


def get_cluster_intervall(dic, current_price, ptg_up=None, ptg_down=None):
    for key in dic.keys():
        if key[0] < current_price < key[1]:
            return key
    else:
        low = np.round(current_price * ((100 - ptg_down) / 100), 1)
        high = np.round(current_price * (100 + ptg_up) / 100, 1)
        return (low, high)


def append_dictionary(dic, key, value):
    if dic.get(key, None) is None:
        dic.update({key: [value]})
    else:
        dic[key].append(value)
        dic[key] = list(np.sort(dic[key])[::-1])  # Order in decreasing values
    return dic


def get_sell_signal(dic, sell_price):
    if len(dic) > 0:
        for key in dic.keys():
            cluster_to_sell = dic[key]
            if len(cluster_to_sell) > 0:
                if sell_price > key[1]:
                    # Get qty to sell and remove it
                    quantity_to_sell = sum(dic[key])
                    print('Price to sell:', sell_price, 'bracket bought:', key[1])
                    print('Qty to sell:', quantity_to_sell)
                    dic[key] = []  # Remove the quantity from portfolio

                    return quantity_to_sell, dic

    return 0, dic


############################################################################
# %% Strategies
############################################################################


@split_data_strat
def singlecrypto_strategy(shrink_datas, full_datas, func_order, crypto_output=None, **kwargs):
    if crypto_output is None:
        crypto_output = [crypto for crypto in shrink_datas.keys()][0]
    full_output = full_datas[crypto_output]
    new_output = shrink_datas[crypto_output]

    if 'Orders' in new_output.columns:
        '''
        "Orders" MUST BE REMOVED before applying dropna()
        Otherwise it might remove relevant information.
        '''
        new_output = new_output.drop('Orders', 1)

    # Compute metric with given function
    Orders = func_order(new_output, **kwargs)

    if len(new_output) < len(full_output):
        live = True
    else:
        live = False

    full_output = full_output.copy()
    full_output = update_dataframe(Orders, full_output, 'Orders', live=live)

    return full_output


@split_data_strat
def multicrypto_strategy(shrink_datas, full_datas, func_order, crypto_output=None, **kwargs):
    if crypto_output is None:
        crypto_output = [crypto for crypto in shrink_datas.keys()][0]
    full_output = full_datas[crypto_output]
    new_output = shrink_datas[crypto_output]

    if 'Orders' in new_output.columns:
        '''
        "Orders" MUST BE REMOVED before applying dropna()
        Otherwise it might remove relevant informations.
        '''
        new_output = new_output.drop('Orders', 1)

    # Compute metric with given function
    Orders = func_order(shrink_datas, crypto_output=crypto_output, **kwargs)

    if len(new_output) < len(full_output):
        live = True
    else:
        live = False

    full_output = full_output.copy()
    full_output = update_dataframe(Orders, full_output, 'Orders', live=live)

    return full_output


def order66():  # Used to force selling
    strategy_layer1 = forceSell  # Strategy to force selling
    strategy_kwargs1 = dict(lookback=1,
                            _type='hidden')
    strategy_layer2 = no_filter_portfolio
    strategy_kwargs2 = dict(lookback=1,
                            prop_to_sell=1.0,
                            _type='readout')  # Set to sell all
    return [(strategy_layer1, strategy_kwargs1), (strategy_layer2, strategy_kwargs2)]


@split_data_strat
def forceSell(shrink_datas, full_datas,
              crypto_output=None,
              **kwargs):
    if crypto_output is None:
        crypto_output = [crypto for crypto in shrink_datas.keys()][0]
    full_output = full_datas[crypto_output]
    new_output = shrink_datas[crypto_output]

    Orders = -np.ones(len(new_output))
    if len(new_output) < len(full_output):
        live = True
    else:
        live = False
    full_output = update_dataframe(Orders, full_output, 'Orders', live=live)

    return full_output


############################################################################
# %% Portfolio actions
############################################################################


def no_filter_portfolio(self, lookback=1, prop_to_buy=0.1, prop_to_sell=1.0,
                        price_label='Close', fees=0.0, **kwargs):
    crypto_output = self.crypto_output
    current_data = self.data[crypto_output]
    # Get last order
    orders = current_data['Orders'].values[-lookback:]
    size = self.size

    ### Buy crypto ###
    for i, order in enumerate(orders):
        Index = size - lookback + i
        if order == 1:
            print(f'Buy order for {self.crypto_output}')
            buy_price = current_data.at[Index, price_label]
            crypto_bought = self.buy_action(Index, buy_price, prop_to_buy=prop_to_buy, fees=fees)
            if crypto_bought == False:
                print('Cancel buy')
                current_data.loc[i, 'Orders'] = 2
            else:
                print(f'-> Buy price ({self.fiat_currency}):', crypto_bought * buy_price)
            print('------------------')

        ### Sell crypto ###
        elif order == -1:
            # print(i, Index, order)
            print(f'Sell order for {self.crypto_output}')
            sell_price = current_data.at[Index, price_label]
            crypto_sell = self.sell_action(Index, sell_price, prop_to_sell=prop_to_sell,
                                           fees=fees)

            if crypto_sell == False:
                print('Cancel sell')
                current_data.loc[i, 'Orders'] = -2
            else:
                print(f'-> Sell price ({self.fiat_currency}): ', crypto_sell * sell_price)
