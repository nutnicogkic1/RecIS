import json
import os
from typing import Optional

import recis.common.crypto_utils as crypto_utils
from recis.utils.logger import Logger


logger = Logger(__name__)
_CONFIG = {}


def _apply_transforms(d, transforms):
    for k, v in d.items():
        if isinstance(v, dict):
            _apply_transforms(v, transforms)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    _apply_transforms(item, transforms)
                elif isinstance(item, str) and item in transforms:
                    v[i] = transforms[item]
        elif isinstance(v, str) and v in transforms:
            d[k] = transforms[v]


def set_config(path: Optional[str] = None, transforms: Optional[dict] = None):
    global _CONFIG
    _CONFIG = json.load(open(path)) if path else {}
    if transforms:
        _apply_transforms(_CONFIG, transforms)
    logger.info(f"Configuration : \n {json.dumps(_CONFIG, indent=2)}")
    return _CONFIG


def get_config(*args, **kwargs):
    default_value = kwargs.pop("default", None)
    global _CONFIG
    value = _CONFIG
    for arg in args:
        if not isinstance(value, dict) or arg not in value:
            value = default_value
            break
        value = value[arg]
    return value


def get_access_id():
    encoded_access_id = os.getenv("ENCODED_ODPS_ACCESS_ID", None)
    dynamic_private_key = os.getenv("XDL_AK_PRIVATE_KEY_INTERNAL", None)
    if encoded_access_id and dynamic_private_key:
        return crypto_utils.decode(encoded_access_id, dynamic_private_key)
    else:
        return None


def get_access_key():
    encoded_access_key = os.getenv("ENCODED_ODPS_ACCESS_KEY", None)
    dynamic_private_key = os.getenv("XDL_AK_PRIVATE_KEY_INTERNAL", None)
    if encoded_access_key and dynamic_private_key:
        return crypto_utils.decode(encoded_access_key, dynamic_private_key)
    else:
        return None
