from typing import Any, Union, Dict, List, Callable, Optional
import hashlib
import pickle
from pandas.util import hash_pandas_object
import pandas as pd
from pathlib import Path
from inspect import signature
import numpy as np
from collections import OrderedDict
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping
import operator
import functools
import json
from json import JSONEncoder


class Cache:
    def __init__(self, dir_path: str, rerun: bool = False, with_param: bool = False):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(exist_ok=True)
        self.with_param = with_param
        self.rerun = rerun

    def __call__(self, func: Callable):
        func_name = func.__name__

        def wrapper(*args, **kwargs):
            sig = signature(func)
            # ignore default value
            bound_args = sig.bind(*args, **kwargs)
            unique_id: str = self._get_unique_id(bound_args.arguments)
            path: Path = self.dir_path.joinpath(f'{func_name}_{unique_id}')

            ret = self._read_cache(path)
            if ret is None:
                ret = func(*args, **kwargs)
                self._write(path, ret)
            return ret

        return wrapper

    def _write(self, path, obj: Union[pd.DataFrame, Any]):
        # TODO: FileProcessor
        if isinstance(obj, pd.DataFrame):
            path = f'{path}.feather'
            obj.to_feather(str(path))
        else:
            path = f'{path}.pickle'
            with open(str(path), 'wb') as f:
                pickle.dump(obj, f, protocol=4)

    def _read_cache(self, path: Path) -> Optional[Any]:
        if self.rerun:
            return None
        if Path(f'{path}.pickle').exists():
            return pickle.load(open(f'{path}.pickle', 'rb'))
        if Path(f'{path}.feather').exists():
            return pd.read_feather(f'{path}.feather')
        return None

    def _get_unique_id(self, params: Dict) -> str:
        if not params:
            return '_with_no_param'
        dependencies = [f'{key}_{self._get_hash(param)}' for key, param in sorted(params.items(), key=lambda item: str(item[0]))]
        return hashlib.md5(str(dependencies).encode()).hexdigest()

    @classmethod
    def _get_hash(cls, obj: Any) -> int:
        if isinstance(obj, (str, int, float)):
            return cls._literals(obj)
        elif isinstance(obj, pd.DataFrame):
            return cls._data_frame(obj)
        elif isinstance(obj, np.ndarray):
            return cls._ndarray(obj)
        elif isinstance(obj, (list, dict, tuple)):
            return cls._mutables(obj)
        else:
            return hash(pickle.dumps(obj))

        return -1

    @staticmethod
    def _data_frame(obj: pd.DataFrame):
        return hash_pandas_object(obj)

    @staticmethod
    def _ndarray(obj: np.ndarray):
        # not implemented
        return hash(bytes(obj))

    @staticmethod
    def _mutables(obj: List[Any]):
        return hash(json.dumps(obj, cls=_DictParamEncoder))

    @staticmethod
    def _literals(obj: Union[int, str, float]):
        return id(obj)


# from https://github.com/spotify/luigi/blob/master/luigi/parameter.py#L940


class _DictParamEncoder(JSONEncoder):
    """
    JSON encoder for :py:class:`~DictParameter`, which makes :py:class:`~FrozenOrderedDict` JSON serializable.
    """
    def default(self, obj):
        if isinstance(obj, FrozenOrderedDict):
            return obj.get_wrapped()
        return json.JSONEncoder.default(self, obj)


class FrozenOrderedDict(Mapping):
    """
    It is an immutable wrapper around ordered dictionaries that implements the complete :py:class:`collections.Mapping`
    interface. It can be used as a drop-in replacement for dictionaries where immutability and ordering are desired.
    """
    def __init__(self, *args, **kwargs):
        self.__dict = OrderedDict(*args, **kwargs)
        self.__hash = None

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        # We should use short representation for beautiful console output
        return repr(dict(self.__dict))

    def __hash__(self):
        if self.__hash is None:
            hashes = map(hash, self.items())
            self.__hash = functools.reduce(operator.xor, hashes, 0)

        return self.__hash

    def get_wrapped(self):
        return self.__dict


def recursively_freeze(value):
    """
    Recursively walks ``Mapping``s and ``list``s and converts them to ``FrozenOrderedDict`` and ``tuples``, respectively.
    """
    if isinstance(value, Mapping):
        return FrozenOrderedDict(((k, recursively_freeze(v)) for k, v in value.items()))
    elif isinstance(value, list) or isinstance(value, tuple):
        return tuple(recursively_freeze(v) for v in value)
    return value