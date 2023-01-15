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


import config
import sys
import numpy as np
from binance.client import Client
import math


def round_decimals_down(number: float, decimals: int = 2):
    """
    Returns a value rounded down to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.floor(number)

    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def make_order(symbol, side, quantity, client, test=False, print_=True):
    # Check limit order
    info = client.get_symbol_info(symbol)
    minQty_round = abs(np.log10(float(info['filters'][1]['minQty'])))
    quantity = round_decimals_down(quantity, int(minQty_round))
    # print(info['filters'][1]['minQty'], minQty_round, quantity)
    if not test:
        order = client.create_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    else:
        order = client.create_test_order(symbol=symbol, side=side, type='MARKET', quantity=quantity)
    if print_:
        print(order)
    return order


def get_balance(client):
    info = client.get_account()
    balances = info['balances']
    dic_balance = {}
    for balance in balances:

        if float(balance['free']) > 0.0:
            dic_balance.update({balance['asset']: float(balance['free'])})
    return dic_balance


def test():
    API_KEY = config.api_key
    SECRET_KEY = config.api_secret

    client = Client(API_KEY, SECRET_KEY)

    dic_balance0 = get_balance(client)
    print(dic_balance0)
    # print('----------------')
    symbol = 'SOLUSDT'
    order_passed = False
    count = 0
    while count < 5 and not order_passed:
        try:
            order = make_order(symbol, 'SELL', 0.97, test=False)
            order_passed = True
        except:
            count += 1
            print(f"({count}) Order failed: {sys.exc_info()[0]} occured.")
            print(sys.exc_info()[1])

    dic_balance1 = get_balance()
    print('----------------')
    print(dic_balance1)
