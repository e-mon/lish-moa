from typing import Any, List, Optional
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.misc import LoggerFactory
from src.models.loss import SmoothBCEwLogits, LogitsLogLoss
from src.models.base import MoaBase
from src.models.pytorch_tabnet.tab_model import TabNetRegressor
from src.utils.environment import get_device

DEVICE = get_device()
logger = LoggerFactory().getLogger(__name__)


class Tabnet(MoaBase):
    def __init__(self, params: Optional[dict] = None, **kwargs):
        if params is None:
            self.params = {}
        else:
            self.params = params
        super().__init__(**kwargs)

    def _get_default_params(self):
        return dict(loss_fn='logloss',
                    max_epoch=200,
                    batch_size=1024,
                    initialize_params=dict(n_d=32,
                                           n_a=32,
                                           n_steps=1,
                                           gamma=1.3,
                                           lambda_sparse=0,
                                           optimizer_fn=optim.Adam,
                                           optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                                           mask_type="entmax",
                                           scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
                                           scheduler_fn=ReduceLROnPlateau,
                                           seed=42,
                                           verbose=10))

    def _train(self, X: pd.DataFrame, y: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        X_train, y_train = X.iloc[train_idx][predictors].values, y.iloc[train_idx].values
        X_valid, y_valid = X.iloc[valid_idx][predictors].values, y.iloc[valid_idx].values

        logger.info(f"train shape: {X_train.shape}, positive frac: {y_train.sum()/y_train.shape[0]}")
        logger.info(f"valid shape: {X_valid.shape}, positive frac: {y_valid.sum()/y_valid.shape[0]}")

        _params = self._get_default_params()
        _params.update(self.params)
        _params['initialize_params']['seed'] = seed

        model = TabNetRegressor(**_params['initialize_params'])
        loss_fn = F.binary_cross_entropy_with_logits if _params['loss_fn'] == 'logloss' else SmoothBCEwLogits(smoothing=0.001)
        logger.info(loss_fn)

        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid, y_valid)],
            eval_name=["val"],
            eval_metric=["logits_ll"],
            max_epochs=_params['max_epoch'],
            patience=20,
            batch_size=_params['batch_size'],
            virtual_batch_size=32,
            num_workers=1,
            drop_last=False,
            # To use binary cross entropy because this is not a regression problem
            loss_fn=loss_fn)

        preds = self._sigmoid(model.predict(X_valid))
        return preds, model

    def _predict(self, model: Any, X_valid: pd.DataFrame, predictors: List[str]):
        preds = model.predict(X_valid[predictors].values)
        return self._sigmoid(preds)

    @staticmethod
    def _sigmoid(preds: np.ndarray):
        return 1 / (1 + np.exp(-preds))
