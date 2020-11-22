from typing import Optional
from sklearn.preprocessing import QuantileTransformer, RobustScaler
import pandas as pd

from src.utils.cache import Cache

TRANSFORMERS = {
    'quantile': QuantileTransformer,
    'robust': RobustScaler,
}


# return transformer
@Cache('./cache')
def normalizer(transformer: str, df: pd.DataFrame, params: Optional[dict]):
    if params is None:
        params = dict()
    trans = TRANSFORMERS[transformer](**params)
    trans.fit(df)

    return trans