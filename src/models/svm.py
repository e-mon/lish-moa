from typing import List, Optional
import pandas as pd
import numpy as np
from cuml.svm import SVC, SVR
from tqdm import tqdm

from src.utils.misc import LoggerFactory
from src.models.base import MoaBase, AllZerosClassifier

logger = LoggerFactory().getLogger(__name__)


class SVMTrainer(MoaBase):
    def __init__(self, params: Optional[dict] = None, **kwargs):
        if params is None:
            self.params = {}
        else:
            self.params = params
        super().__init__(**kwargs)

    def _get_default_params(self):
        return {
            'cache_size': 5000,
            'probability': True,
        }

    def _train_predict(self, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray,
                       seed: int):
        _params = self._get_default_params()
        _params.update(self.params)

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        target_cols = y_valid.columns.tolist()

        pred_valid = np.zeros_like(y_valid).astype(float)
        preds = np.zeros(shape=(X_test.shape[0], y_train.shape[1]))

        # multilabel分回す
        for idx, target_col in tqdm(enumerate(target_cols), total=len(target_cols)):
            # Since cuml SVC calls CalibratedClassifierCV(n_folds=5), more than 5 positive samples is required
            if y_train[target_col].sum() < 5:
                logger.info(f'{target_col} is all zeros')
                clf = AllZerosClassifier()
            else:
                clf = SVC(**_params)
                clf.fit(X_train[predictors].values, y_train[target_col].values.astype(int), convert_dtype=False)
            pred_valid[:, idx] = clf.predict_proba(X_valid[predictors].values)[:, 1]
            preds[:, idx] = clf.predict_proba(X_test[predictors].values)[:, 1]

        return pred_valid, preds
