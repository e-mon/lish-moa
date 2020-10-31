from src.models.base import BaseModel
from logging import getLogger
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
