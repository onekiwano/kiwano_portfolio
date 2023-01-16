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

from pathlib import Path

'''
/!\ Do not forget to change the name of this 
file to 'config.py', then it will be ignored by the  `.gitignore` file and not 
uploaded in the git repo. /!\
Go to the website and generate your own keys, then replace them
in this file.
'''

api_key = "465VS4DG645F65D4S654G65D4D" # Fake, replace by your own
api_secret = '6BAD45FAGFBADFBVA545A45A34VA54' 

'''
(Optional) 
Use the function 'encrypt_key' in the file 'encrypt.py' 
in the folder "API/", to encrypt your private key. 
Save the generated key in another folder, and specify
the path_key here, then the snippet below will decrypt the key
each time you want to access it.
'''
# import sys
# PATH_ROOT = Path(__file__).parents[1] # Adjust the number if needed
# sys.path.append(str(PATH_ROOT))
# from encrypt import decrypt_key
# encrypted_key = 'fdjhlsakjvakqj-vs4qaac-vkjlfjfjg' 
# path_key = "specify/the/location/"
# api_secret = decrypt_key(encrypted_key, path=path_key)

