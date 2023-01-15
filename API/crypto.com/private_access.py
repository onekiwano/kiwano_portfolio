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
import hmac
import hashlib
import time
import config
import requests
import json
from pathlib import Path
import sys

PATH_ROOT = Path(__file__).parents[1] # Adjust the number if needed
sys.path.append(str(PATH_ROOT))
from encrypt import decrypt_key

path = Path(__file__).parents[4] # Adjust the number if needed
# See https://exchange-docs.crypto.com/spot/index.html?python#private-get-withdrawal-history

BASE_URL = "https://api.crypto.com/v2/"
# API keys
API_KEY = config.public_key
SECRET_KEY = decrypt_key(config.secret_key, path=path)

# First ensure the params are alphabetically sorted by key
param_str = ""

MAX_LEVEL = 3


def params_to_str(obj, level):
    if level >= MAX_LEVEL:
        return str(obj)

    return_str = ""
    for key in sorted(obj):
        return_str += key
        if isinstance(obj[key], list):
            for subObj in obj[key]:
                return_str += params_to_str(subObj, ++level)
        else:
            return_str += str(obj[key])
    return return_str

def generic_req_setup(req):
    
    if "params" in req:
        param_str = params_to_str(req['params'], 0)
    
    payload_str = req['mode'] + str(req['id']) + req['api_key'] + param_str + str(req['nonce'])
    
    req['sig'] = hmac.new(
        bytes(str(SECRET_KEY), 'utf-8'),
        msg=bytes(payload_str, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    return req

def get_order_history(option='deposit', req=None):
    '''
    option : 'deposit' or 'withdrawal'
    '''
    
    method = f"private/get-{option}-history"
    
    if req is None:
        req = {
            "id": 11,
            "method":method,
            "api_key": API_KEY,
            "params": {},
            "nonce": int(time.time() * 1000)
            }
    
    # Setup req parameter
    req = generic_req_setup(req)
    
    deposit_history = requests.post(BASE_URL+method, 
                                   json=req,
                                   headers={'Content-type':'application/json'})
    
    return json.loads(deposit_history.text)