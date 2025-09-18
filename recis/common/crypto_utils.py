import base64

from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA


def encode(data, public_key, binary_code="utf-8"):
    """Encode data
    Args:
      data: string, information to encrypt
      public_key: string, public key to encode information
      binary_code: string, default to be utf-8
    Return:
      The encoded information in string.
    """
    data = str(data)
    public_key_bin = base64.b64decode(str(public_key))
    bin_data = data.encode(binary_code)
    cipher = PKCS1_v1_5.new(RSA.import_key(public_key_bin))
    encrypted = cipher.encrypt(bin_data)
    return base64.b64encode(encrypted).decode(binary_code)


def decode(data, private_key, binary_code="utf-8"):
    """Decode data
    Args:
      data: string, information to decrypt
      private_key: string, private key to decode information
      binary_code: string, default to be utf-8
    Return:
      The decoded information in string.
    """
    data = str(data)
    private_key_bin = base64.b64decode(str(private_key))
    bin_data = base64.b64decode(data)
    cipher = PKCS1_v1_5.new(RSA.import_key(private_key_bin))
    decrypted = cipher.decrypt(bin_data, "ERROR")
    str_data = decrypted.decode(binary_code)
    return str_data
