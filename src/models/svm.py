from typing import List, Any
import pandas as pd
import numpy as np
from cuml.svm import SVC, SVR
from tqdm import tqdm
from joblib import dump, load

from src.utils.cache import Cache
from src.utils.misc import LoggerFactory
from src.models.base import MoaBase, AllZerosClassifier

logger = LoggerFactory().getLogger(__name__)


class SVMTrainer(MoaBase):
    def _train(self, X: pd.DataFrame, y: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        # BUG: 以下が起動ごとに全errorしたりしなかったり確率的にコケる (以下参照)
        # なかなか解決しないので、何回か回してコケなかったら通すようにする
        # https://kaggler-ja.slack.com/archives/G01F5QWJ5SM/p1605936016147700

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        target_cols = y_valid.columns.tolist()

        pred_valid = np.zeros_like(y_valid).astype(float)

        # multilabel分回す
        models = []
        for idx, target_col in tqdm(enumerate(target_cols), total=len(target_cols)):
            # Since cuml SVC calls CalibratedClassifierCV(n_folds=5), more than 5 positive samples is required
            if y_train[target_col].sum() < 5:
                logger.info(f'{target_col} is all zeros')
                clf = AllZerosClassifier()
            else:
                clf = SVC(cache_size=5000, probability=True)
                clf.fit(X_train[predictors].values, y_train[target_col].values.astype(int), convert_dtype=False)
            pred_valid[:, idx] = clf.predict_proba(X_valid[predictors].values)[:, 1]

            # 一時保存のためにparameterのhash値取得
            unique_id = Cache._get_unique_id(params={
                'train_idx': train_idx,
                'valid_idx': valid_idx,
                'seed': seed,
                'target_col': target_col,
            })
            model_path = f'./cache/SVM_model_{unique_id}.joblib'
            logger.debug(f'{target_col} is dumped as {model_path}')
            dump(clf, model_path)
            models.append(model_path)

        logger.info(pred_valid.shape)
        return pred_valid, models

    def _predict(self, model: Any, X_valid: pd.DataFrame, predictors: List[str]):
        assert type(model) is list, 'model is not list'

        preds = np.zeros(shape=(X_valid.shape[0], len(model)))
        for idx, path in enumerate(model):
            clf = load(path)
            logger.debug(f'{path} is loaded')
            preds[:, idx] = clf.predict_proba(X_valid[predictors].values)[:, 1]

        return preds
