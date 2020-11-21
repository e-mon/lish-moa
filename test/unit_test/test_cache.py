import unittest
from unittest.mock import patch
from src.utils.cache import Cache
from typing import List, Any
import pandas as pd


class TestCache(unittest.TestCase):
    def test_hash(self):
        test_case: List[Any] = [1, 'a', {1, 2, 3}, [1, 2, 3], (1, 2, 3), {'key': 'value'}]
        expecteds = [
            'c4ca4238a0b923820dcc509a6f75849b', '0cc175b9c0f1b6a831c399e269772661', '4c24e01fa26fc915e3f057d6c6bfd560', '49a5a960c5714c2e29dd1a7e7b950741',
            '49a5a960c5714c2e29dd1a7e7b950741', '88bac95f31528d13a072c05f2a1cf371'
        ]

        for obj, expected in zip(test_case, expecteds):
            result = Cache._get_hash(obj)
            self.assertEqual(result, expected)

    def test_dataframe(self):
        df = pd.DataFrame(dict(col_1=[1, 2, 3], col_2=['a', 'b', 'c']))
        expected = -5174853151898171182
        result = Cache._get_hash(df)

        self.assertEqual(result, expected)

        df = pd.DataFrame(dict(col_1=[3, 2, 1], col_2=['a', 'b', 'c']))
        result = Cache._get_hash(df)
        self.assertNotEqual(result, expected)

    def test_unique_id(self):
        params = {'param_a': 123, 'param_b': [1, 2, 3], 'param_c': {'key': 'value'}}

        expected = '1fd1c9224dc3180dea4d058e90e095df'
        result = Cache._get_unique_id(params)

        self.assertEqual(result, expected)

    def test_read_path(self):
        def read_cache(path, rerun):
            self.path = path
            return path

        with patch('src.utils.cache.Cache._read_cache', read_cache):

            @Cache('test')
            def func(param):
                return param

            expected = 'test/func_b4216b72b74587638f054cc8e5e9825c'
            ret = func('abc')
            self.assertEqual(str(self.path), expected)

            ret = func('def')
            self.assertNotEqual(str(self.path), expected)