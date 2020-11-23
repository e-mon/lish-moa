import numpy as np

from scipy.optimize import minimize
from sklearn.model_selection import KFold

from src.metrics import calc_competition_metric_torch
from src.utils.misc import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


def get_best_weights(oof_1, oof_2, train_features_df, targets, n_splits=10):
    weight_list = []
    weights = np.array([0.5])
    for i in range(2):
        kf = KFold(n_splits=n_splits, random_state=i, shuffle=True)
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X=oof_1)):
            res = minimize(
                get_score,
                weights,
                args=(train_features_df, train_idx, oof_1, oof_2, targets),
                method="Nelder-Mead",
                tol=1e-6,
            )
            logger.info(f"i: {i} fold: {fold} res.x: {res.x}")
            weight_list.append(res.x)
    mean_weight = np.mean(weight_list)
    logger.info(f"optimized weight: {mean_weight}")
    return mean_weight


def get_score(weights, train_features_df, train_idx, oof_1, oof_2, targets):
    _oof_1 = oof_1[train_idx, :].copy()
    _oof_2 = oof_2[train_idx, :].copy()
    blend = (_oof_1 * weights[0]) + (_oof_2 * (1 - weights[0]))
    return calc_competition_metric_torch(train_features_df, targets, blend, train_idx)