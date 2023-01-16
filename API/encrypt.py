from cryptography.fernet import Fernet
import numpy as np
from os.path import exists, join

# generate a key for encryption and decryption
# You can use fernet to generate
# the key or use random key generator
# here I'm using fernet to generate the key.

def encrypt_key(secret_key, key=None, path='', name='key_crypto_bot.npy'):
    
    if (key is None):
        path_file = join(path, name)
        if not exists(path_file):
            key = Fernet.generate_key()
            np.save(path_file, key)
        else:
            key = str(np.load(path_file))[1:]
    
    # Instance the Fernet class with the key
    fernet = Fernet(key)
    
    # then use the Fernet class instance
    # to encrypt the string string must
    # be encoded to byte string before encryption
    encMessage = str(fernet.encrypt(secret_key.encode()))[1:]
    decMessage = fernet.decrypt(encMessage.encode()).decode()
    if decMessage != secret_key:
        raise Exception('Issues when encrypting, source and decoded messages are not the same')
    return encMessage

def decrypt_key(encMessage, key=None, path=None, name='key_crypto_bot.npy'):
    
    if (key is None) and (path is not None):
        path_file = join(path, name)
        if exists(path_file):
            key = str(np.load(path_file))[1:]    
    fernet = Fernet(key)
    decMessage = fernet.decrypt(encMessage.encode()).decode()
    return decMessage
