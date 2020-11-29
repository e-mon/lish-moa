from typing import List, Optional
import xgboost as xgb
import pandas as pd
import numpy as np
from src.models.base import BaseModel, MoaBase, AllZerosClassifier
from logging import getLogger
from tqdm import tqdm
import lightgbm as lgb

logger = getLogger(__name__)


class LGBModel(BaseModel):
    def __init__(self, params: dict, **kwargs):
        self.params = params
        super().__init__(**kwargs)

    def _get_default_params(self):
        return {
            "n_estimators": 5000,
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "None",
            "first_metric": True,
            "subsample": 0.8,
            "subsample_freq": 1,
            "learning_rate": 0.01,
            "feature_fraction": 0.7,
            "num_leaves": 12,
            "max_depth": -1,
            "early_stopping_rounds": 300,
            "seed": 42,
        }

    def _train(self, train, test, targets, train_idx, valid_idx):
        predictors = [col for col in train.columns if col not in self.ignore_cols]
        logger.info(predictors)
        X_train, y_train = train.iloc[train_idx][predictors], targets.iloc[train_idx]
        X_valid, y_valid = train.iloc[valid_idx][predictors], targets.iloc[valid_idx]

        logger.info(f"train shape: {X_train.shape}, positive frac: {y_train.sum()/y_train.shape[0]}")
        logger.info(f"valid shape: {X_valid.shape}, positive frac: {y_valid.sum()/y_valid.shape[0]}")

        train_set = lgb.Dataset(X_train, y_train, categorical_feature=self.categorical_cols)
        val_set = lgb.Dataset(X_valid, y_valid, categorical_feature=self.categorical_cols)

        _params = self._get_default_params()
        _params.update(self.params)

        clf = lgb.train(
            _params,
            train_set,
            valid_sets=[train_set, val_set],
            verbose_eval=100,
            fobj=None,
        )

        return clf.predict(X_valid), clf.predict(test[predictors]), clf


class XGBTrainer(MoaBase):
    def __init__(self, params: Optional[dict] = None, **kwargs):
        if params is None:
            self.params = {}
        else:
            self.params = params
        super().__init__(**kwargs)

    def _get_default_params(self):
        return {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'gpu_hist',
            'verbosity': 0,
            'colsample_bytree': 0.1818593017814899,
            'eta': 0.012887963193108452,
            'gamma': 6.576022976359221,
            'max_depth': 8,
            'min_child_weight': 8.876744371188476,
            'subsample': 0.7813380253086911,
        }

    def _train(self, X: pd.DataFrame, y: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        X_train, y_train = X.iloc[train_idx][predictors], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx][predictors], y.iloc[valid_idx]

        logger.info(f"train shape: {X_train.shape}, positive frac: {y_train.sum()/y_train.shape[0]}")
        logger.info(f"valid shape: {X_valid.shape}, positive frac: {y_valid.sum()/y_valid.shape[0]}")

        _params = self._get_default_params()
        _params.update(self.params)
        _params['seed'] = seed

        # multilabel分回す
        target_cols = y_valid.columns.tolist()
        pred_valid = np.zeros_like(y_valid).astype(float)
        models = []

        for idx, target_col in tqdm(enumerate(target_cols), total=len(target_cols)):
            xgb_train = xgb.DMatrix(X_train.values, label=y_train[target_col].values.astype(int), nthread=-1)
            xgb_valid = xgb.DMatrix(X_valid.values, label=y_valid[target_col].values.astype(int), nthread=-1)
            clf = xgb.train(_params, xgb_train, 1000, [(xgb_valid, "eval")], early_stopping_rounds=25, verbose_eval=0)
            pred_valid[:, idx] = clf.predict(xgb_valid)
            models.append(clf)

        return pred_valid, models

    def _predict(self, model: List, X_valid: pd.DataFrame, predictors: List[str]):
        assert type(model) is list, 'model is not list'

        preds = np.zeros(shape=(X_valid.shape[0], len(model)))
        for idx, clf in enumerate(model):
            xgb_valid = xgb.DMatrix(X_valid.values, nthread=-1)
            preds[:, idx] = clf.predict(xgb_valid)

        return preds
