import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss

from src.utils.misc import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


def calc_competition_metric_torch(train_features_df, target_cols, oof_arr, train_idx):
    # competition_metric = [log_loss(train_features_df.loc[train_idx, target_cols[i]], oof_arr[:, i]) for i in range(len(target_cols))]
    y = torch.tensor(train_features_df.loc[train_idx, target_cols].values, dtype=float)
    p = torch.tensor(oof_arr, dtype=float)
    p = torch.clamp(p, 1e-9, 1 - (1e-9))
    competition_metric = nn.BCELoss()(p, y).item()
    return np.mean(competition_metric)


def calc_competition_metric_np(train_features_df, target_cols, oof_arr):
    competition_metric = []
    for i in range(len(target_cols)):
        competition_metric.append(log_loss(train_features_df[:, target_cols[i]], oof_arr[:, i]))
    logger.info(f"competition metric: {np.mean(competition_metric)}")

    return np.mean(competition_metric)


def logloss_for_multilabel(actual, preds, ignore_all_zeros: bool = True):
    """
    actual, preds: [n_samples, n_classes]
    log_loss(actual[:, c], preds[:, c])
    """

    actual = torch.tensor(actual, dtype=float)
    preds = torch.tensor(preds, dtype=float)
    preds = torch.clamp(preds, 1e-9, 1 - (1e-9))

    return np.mean(nn.BCELoss()(preds, actual).item())
