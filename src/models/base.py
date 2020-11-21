from abc import abstractmethod
from src.utils import SplitFactory
from typing import NamedTuple, Callable, List, Dict
import numpy as np
import pandas as pd
from logging import getLogger
from src.experiment.experiment import Experiment

logger = getLogger(__name__)


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: np.ndarray
    models: List[any]
    scores: Dict[float]


class BaseModel:
    def __init__(self, ignore_cols: List[str], target_col: str, categorical_cols: List[str], metric: Callable, exp: Experiment):
        self.ignore_cols = ignore_cols
        self.metric = metric
        self.result = None

    @abstractmethod
    def _train_predict(self, train: pd.DataFrame, targets: pd.DataFrame, train_idx, valid_idx):
        raise NotImplementedError

    def train_predict(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, splitter: SplitFactory):

        models = []
        scores = []
        oof_preds = np.zeros(shape=(X_train.shape[0], ))
        preds = np.zeros(shape=(X_test.shape[0], ))
        folds = splitter.split(X_train, y_train)

        for fold, (train_idx, valid_idx) in enumerate(folds):
            valid_preds, _preds, model = self._train_predict(X_train, X_test, y_train, train_idx, valid_idx)
            oof_preds[valid_idx] = valid_preds
            preds += _preds / len(folds)

            score = self.metric(y_train[valid_idx], valid_preds)
            logger.info(f"fold {fold}: {score}")
            models.append(model)
            scores.append(score)
        oof_score = self.metric(y_train, oof_preds)
        logger.info(f"{len(folds)} folds cv mean: {np.mean(scores)}")
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(oof_preds=oof_preds,
                                  models=models,
                                  preds=preds,
                                  scores={
                                      'oof_score': self.metric(y_train, oof_preds),
                                      'KFoldsScores': scores,
                                  })

        return True
