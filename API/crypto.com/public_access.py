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
import requests
import json
from datetime import datetime

BASE_URL = "https://api.crypto.com/v2/"
API_KEY = config.public_key


def get_generic(specific_URL):
    complete_URL = BASE_URL + specific_URL
    print(complete_URL)
    informations = requests.get(complete_URL)
    date_today = datetime.today()
    return json.loads(informations.text), date_today


def get_candlestick(instrument_name, timeframe):
    specific_URL = f"public/get-candlestick?instrument_name={instrument_name}&timeframe={timeframe}"
    return get_generic(specific_URL)


def get_book(instrument_name, depth):
    specific_URL = f"public/get-candlestick?instrument_name={instrument_name}&depth={depth}"
    return get_generic(specific_URL)


def get_trade():
    specific_URL = f"public/get-trades"
    return get_generic(specific_URL)
