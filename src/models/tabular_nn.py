from typing import List, Optional

from src.utils.misc import LoggerFactory
from src.models.loss import SmoothBCEwLogits
from src.models.base import MoaBase
from src.utils.environment import get_device
import copy
import pandas as pd
import numpy as np
from tqdm.auto import trange

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim

DEVICE = get_device()
logger = LoggerFactory().getLogger(__name__)


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: Optional[pd.DataFrame], predictors):
        self.predictors = predictors
        self.X = X[predictors].values

        if y is not None:
            self.y = y.values
        else:
            self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return torch.tensor(self.X[idx], dtype=torch.float).to(DEVICE)
        else:
            return (
                torch.tensor(self.X[idx], dtype=torch.float).to(DEVICE),
                torch.tensor(self.y[idx], dtype=torch.float).to(DEVICE),
            )


class TabularMLP_1_1(nn.Module):
    def __init__(self, features, targets):
        super(TabularMLP_1_1, self).__init__()

        self.sq = nn.Sequential(
            nn.BatchNorm1d(len(features)),
            nn.utils.weight_norm(nn.Linear(len(features), 1024)),
            #             nn.Dropout(0.8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.utils.weight_norm(nn.Linear(1024, 500)),
            nn.Dropout(0.8),
            nn.LeakyReLU(),
            nn.Linear(500, len(targets)),
        )

    def forward(self, x):
        x = self.sq(x)
        return x


class TabularMLP_1_2(nn.Module):
    def __init__(self, n_features, n_targets, hidden_size=512, dropratio=0.2):
        super(TabularMLP_1_2, self).__init__()
        n_features = len(n_features)
        n_targets = len(n_targets)
        self.batch_norm1 = nn.BatchNorm1d(n_features)
        self.dropout1 = nn.Dropout(dropratio)
        self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropratio)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class TabularMLP_2(nn.Module):
    def __init__(self, features, targets):
        super(TabularMLP_2, self).__init__()

        self.sq = nn.Sequential(
            nn.BatchNorm1d(len(features)),
            nn.Linear(len(features), 2048),
            #             nn.Dropout(0.8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 500),
            nn.Dropout(0.8),
            nn.LeakyReLU(),
            nn.Linear(500, len(targets)),
        )

    def forward(self, x):
        x = self.sq(x)
        return x


class NNTrainer(MoaBase):
    def __init__(self, params: Optional[dict] = None, **kwargs):
        if params is None:
            self.params = {}
        else:
            self.params = params
        super().__init__(**kwargs)

    def _get_default_params(self):
        return {
            'lr': 1e-4,
            'batch_size': 256,
            'epoch': 20,
            'model_class': TabularMLP_1_1,
        }

    def _train(self, X: pd.DataFrame, y: pd.DataFrame, predictors: List[str], train_idx: np.ndarray, valid_idx: np.ndarray, seed: int):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        target_cols = y_valid.columns.tolist()

        _params = self._get_default_params()
        _params.update(self.params)

        # define model & schedulers
        num_epoch = _params['epoch']
        batch_size = _params['batch_size']
        net = _params['model_class'](predictors, target_cols)
        net.to(DEVICE)

        optimizer = optim.Adam(net.parameters(), lr=_params['lr'], weight_decay=1e-6)
        valid_criterion = nn.BCEWithLogitsLoss()
        criterion = SmoothBCEwLogits(smoothing=0.001)
        scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

        # 学習時はlength=1の破片などを回避するためdrop_last=1とする
        train_dataset = TabularDataset(X_train, y_train, predictors)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        valid_dataset = TabularDataset(X_valid, y_valid, predictors)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        bar = trange(num_epoch, desc=f"seed: {seed} train : {X_train.shape[0]}  valid:{X_valid.shape[0]}====")
        train_loss = []
        valid_loss = []

        best_loss = np.inf
        best_preds = None
        best_loss_epoch = 1

        for epoch in bar:
            running_loss = []
            valid_loss = []

            # train
            net.train()
            for x, y in train_dataloader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad()
                out = net(x)
                loss = criterion(out, y)
                loss.backward()
                running_loss.append(loss.item())
                optimizer.step()
            scheduler.step()

            net.eval()

            preds_valid = []
            _valid_loss = []

            with torch.no_grad():
                for x, y in valid_dataloader:
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    out = net(x)
                    loss = valid_criterion(out, y)
                    preds_valid.append(out.sigmoid().detach().cpu().numpy())
                    _valid_loss.append(loss.item())

                bar.set_postfix(
                    running_loss=f"{np.mean(running_loss):.5f}",
                    valid_loss=f"{np.mean(_valid_loss):.5f}",
                    best_loss=f"{best_loss:.5f}",
                    best_loss_epoch=f"{best_loss_epoch}",
                )

            train_loss.append(np.mean(running_loss))
            valid_loss.append(np.mean(_valid_loss))

            if best_loss > np.mean(_valid_loss):
                best_loss = np.mean(_valid_loss)
                best_loss_epoch = epoch + 1
                best_preds = np.concatenate(preds_valid)
                best_state = copy.deepcopy(net.state_dict())

        logger.info(f"best loss : {best_loss}")
        model = _params['model_class'](predictors, target_cols)
        model.load_state_dict(best_state)
        model.to(DEVICE)
        return best_preds, model

    def _predict(self, model: Any, X_valid: pd.DataFrame, predictors: List[str]):
        _params = self._get_default_params()
        _params.update(self.params)
        batch_size = _params['batch_size']
        valid_dataset = TabularDataset(X_valid, None, predictors)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        tmp_pred = []
        with torch.no_grad():
            for x in valid_dataloader:
                x = x.to(DEVICE)
                out = model(x)
                tmp_pred.append(out.sigmoid().detach().cpu().numpy())
        return np.concatenate(tmp_pred)
