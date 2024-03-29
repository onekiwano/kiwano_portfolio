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

import sys
import requests
from pathlib import Path

from datetime import datetime
from datetime import timedelta

from kiwano_portfolio.model.model_utils import set_lookback, list2df

PATH_ROOT = Path(__file__).resolve().parents[2]  # Adjust the number if needed
sys.path.append(str(PATH_ROOT) + '/model/')


def get_realtime_price(symbol):
    key = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    data = requests.get(key)
    data = data.json()
    return data['price']


def date_round(date):
    return datetime(year=date.year,
                    month=date.month,
                    day=date.day,
                    hour=date.hour)


def get_candlestick(crypto_name, interval, lookback, client, end_date=None, as_df=True):

    last_time = None
    if end_date in [None, 'now']:
        hk = client.get_historical_klines(crypto_name, interval, lookback + ' ago UTC')
        if 'd' in interval:
            last_time = client.get_historical_klines(crypto_name, '1m', '1m' + ' ago UTC')
    else:
        d = datetime(*[int(s) for s in end_date.split('-')])
        utc_delta = date_round(datetime.utcnow()) - date_round(datetime.now())
        d = d + utc_delta
        lookback_int = set_lookback(lookback, interval, 'int')
        if 'd' in lookback:
            dpast = d - timedelta(days=lookback_int)
        elif 'h' in lookback:
            dpast = d - timedelta(hours=lookback_int)  # 60
        elif 'm' in lookback:
            dpast = d - timedelta(minutes=lookback_int)  # 60
        else:
            raise NotImplementedError
        
        current_time = str(date_round(datetime.now()).date())
        if (str(d.date()) == current_time) and ('d' in interval):
            d = d - timedelta(days=1)
            last_time = client.get_historical_klines(crypto_name, '1m', '1m' + ' ago UTC')
        
        hk = client.get_historical_klines(crypto_name, interval, str(dpast), str(d))
        if last_time is not None:
            hk.append(last_time[0])
        
        hk_len = len(hk)
        lookback_int = set_lookback(lookback, interval, 'int')
        if hk_len > lookback_int:
            nb_elem_to_remove = hk_len - lookback_int
            hk = hk[nb_elem_to_remove:]

    if as_df:
        hk = list2df(hk)
    return hk
