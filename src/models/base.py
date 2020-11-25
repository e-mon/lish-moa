from abc import abstractmethod
from src.utils.splitter import SplitFactory
from typing import NamedTuple, Callable, List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.experiment.experiment import Experiment
from src.utils.misc import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


class ModelResult(NamedTuple):
    oof_preds: np.ndarray
    preds: Optional[np.ndarray]
    models: Dict[str, any]
    scores: Dict[str, float]
    folds: List[Tuple[np.ndarray, np.ndarray]]


class BaseModel:
    def __init__(self, ignore_cols: List[str], target_cols: str, categorical_cols: List[str], metric: Callable, exp: Experiment):
        self.ignore_cols = ignore_cols
        self.metric = metric
        self.result = None

    @abstractmethod
    def _train(self, train: pd.DataFrame, targets: pd.DataFrame, train_idx, valid_idx):
        raise NotImplementedError

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, splitter: Optional[SplitFactory],
              folds: Optional[List[Tuple[np.ndarray, np.ndarray]]]):

        models = dict()
        scores = dict()
        oof_preds = np.zeros_like(y_train).astype(float)
        preds = np.zeros(shape=(X_test.shape[0], y_train.shape[1]))
        assert (folds is not None) or (splitter is not None), 'splitter or folds is must be specified'
        if folds is None:
            folds = splitter.split(X_train, y_train)

        for fold, (train_idx, valid_idx) in enumerate(folds):
            valid_preds, _preds, model = self._train(X_train, X_test, y_train, train_idx, valid_idx)
            oof_preds[valid_idx] += valid_preds
            preds += _preds / len(folds)

            score = self.metric(y_train[valid_idx].values, valid_preds)
            logger.info(f"fold {fold}: {score}")
            models[f'fold_{fold}'] = model
            scores[f'fold_{fold}'] = score
        oof_score = self.metric(y_train.values, oof_preds)
        logger.info(f"{len(folds)} folds cv mean: {np.mean(scores)}")
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(oof_preds=oof_preds, models=models, preds=preds, folds=folds, scores={
            'oof_score': oof_score,
            'KFoldsScores': scores,
        })

        return True

    def predict(self, X_test):
        assert self.result is None, 'Model is not tained Error'
        pass


class MoaBase:
    def __init__(self, target_cols: List[str], categorical_cols: List[str], ignore_cols: Optional[List[str]], num_seed_blends: int, metric: Callable,
                 exp: Experiment):
        self.exp = exp
        self.ignore_cols = ignore_cols
        self.categorical_cols = categorical_cols
        self.metric = metric
        self.result = None
        self.num_seed_blends = num_seed_blends

    @abstractmethod
    def _train(self, X: pd.DataFrame, y: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        raise NotImplementedError

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]):

        models = dict()
        scores = dict()
        oof_preds = np.zeros_like(y_train).astype(float)
        self.predictors = [col for col in X_train.columns.tolist() if col not in self.ignore_cols]

        logger.info(f'{self.__class__.__name__} train start')
        logger.info(f'X shape: {X_train.shape}, y shape: {y_train.shape}')
        for fold, (train_idx, valid_idx) in enumerate(folds):
            logger.info(f'fold {fold}: #row of train: {len(train_idx)}, #row of valid: {len(valid_idx)}')
            for i in range(self.num_seed_blends):
                valid_preds, model = self._train(X=X_train, y=y_train, predictors=self.predictors, train_idx=train_idx, valid_idx=valid_idx, seed=i)

                oof_preds[valid_idx, :] += valid_preds / self.num_seed_blends
                models[f'fold_{fold}_{i}'] = model

            score = self.metric(y_train.iloc[valid_idx].values, oof_preds[valid_idx, :])
            logger.info(f"fold {fold}: {score}")
            scores[f'fold_{fold}'] = score
        oof_score = self.metric(y_train.values, oof_preds)
        logger.info(f"{len(folds)} folds cv mean: {np.mean(list(scores.values()))}")
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(oof_preds=oof_preds, models=models, preds=None, folds=folds, scores={
            'oof_score': oof_score,
            'KFoldsScores': scores,
        })

        return True

    @abstractmethod
    def _predict(self, model: Any, X_valid: pd.DataFrame, predictors: List[str]):
        pass

    def predict(self, X_test) -> np.ndarray:
        assert self.result is not None, 'Model is not trained Error'

        folds = self.result.folds

        n_targets = self.result.oof_preds.shape[1]
        preds = np.zeros(shape=(X_test.shape[0], n_targets))

        for fold, (train_idx, valid_idx) in enumerate(folds):
            for i in range(self.num_seed_blends):
                model = self.result.models[f'fold_{fold}_{i}']
                preds += self._predict(model=model, X_valid=X_test, predictors=self.predictors) / (len(folds) * self.num_seed_blends)

        return preds


class MoaBaseOnline:
    def __init__(self, target_cols: List[str], categorical_cols: List[str], ignore_cols: Optional[List[str]], num_seed_blends: int, metric: Callable,
                 exp: Experiment):
        self.exp = exp
        self.ignore_cols = ignore_cols
        self.categorical_cols = categorical_cols
        self.metric = metric
        self.result = None
        self.num_seed_blends = num_seed_blends

    @abstractmethod
    def _train_predict(self, X: pd.DataFrame, y: pd.DataFrame, X_test: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray,
                       seed: int):
        raise NotImplementedError

    def train_predict(self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, folds: List[Tuple[np.ndarray, np.ndarray]]):

        scores = dict()
        oof_preds = np.zeros_like(y_train).astype(float)
        preds = np.zeros(shape=(X_test.shape[0], y_train.shape[1]))
        self.predictors = [col for col in X_train.columns.tolist() if col not in self.ignore_cols]

        logger.info(f'{self.__class__.__name__} train start')
        logger.info(f'X shape: {X_train.shape}, y shape: {y_train.shape}')
        for fold, (train_idx, valid_idx) in enumerate(folds):
            logger.info(f'fold {fold}: #row of train: {len(train_idx)}, #row of valid: {len(valid_idx)}')
            for i in range(self.num_seed_blends):
                _preds, valid_preds, = self._train_predict(X=X_train,
                                                           y=y_train,
                                                           X_test=X_test,
                                                           predictors=self.predictors,
                                                           train_idx=train_idx,
                                                           valid_idx=valid_idx,
                                                           seed=i)

                oof_preds[valid_idx, :] += valid_preds / self.num_seed_blends
                preds += _preds / (self.num_seed_blends * self.num_seed_blends)

            score = self.metric(y_train.iloc[valid_idx].values, oof_preds[valid_idx, :])
            logger.info(f"fold {fold}: {score}")
            scores[f'fold_{fold}'] = score
        oof_score = self.metric(y_train.values, oof_preds)
        logger.info(f"{len(folds)} folds cv mean: {np.mean(list(scores.values()))}")
        logger.info(f"oof score: {oof_score}")

        self.result = ModelResult(oof_preds=oof_preds, models=None, preds=preds, folds=folds, scores={
            'oof_score': oof_score,
            'KFoldsScores': scores,
        })

        return True


class AllZerosClassifier:
    def __init__(self, label=0):
        self.label = label

    def predict(self, X):
        return np.ones(X.shape[0]) * self.label

    def predict_proba(self, X):
        labels = np.ones(shape=(X.shape[0], 2))
        labels[:, 1 - self.label] = 0
        return labels
