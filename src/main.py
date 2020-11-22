import datetime
import logging
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from textwrap import dedent

from cuml.svm import SVC, SVR
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from joblib import dump, load
from numba import cuda
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch import optim
from torch.nn.modules.loss import _WeightedLoss
from tqdm.auto import tqdm, trange

import warnings

warnings.simplefilter("ignore")

formatter = "%(levelname)s : %(asctime)s : %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)
logger = logging.getLogger(__name__)

# 以下は先に定義しておく必要がある
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tprint(*args, **kwargs):
    if config_d["is_kernel"]:
        print(datetime.datetime.now(), *args)
    else:
        logger.info(dedent(*args))
    return


config_d = {
    "is_kernel": False,  # kaggle notebookで実行するときにはTrueとする
    "debug": False,  # 各所で処理をバイパスする
    "phase": "train",  # train/pred
    # "phase": "pred",  # train/pred
}

# モデルやキャッシュ出力用のディレクトリを整備
CACHE_DIR = "./work/cache"
MODEL_DIR = "./work/model"
RESULT_DIR = "./work/result"

output_dirs = [
    CACHE_DIR,
    MODEL_DIR,
    RESULT_DIR,
]
for output_dir in output_dirs:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

INPUT_DIR = "./data"
MYSEED = 42
seed_everything(seed=MYSEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _calc_competition_metric(train_features_df, target_cols, oof_arr, train_idx):
    # competition_metric = [log_loss(train_features_df.loc[train_idx, target_cols[i]], oof_arr[:, i]) for i in range(len(target_cols))]
    y = torch.tensor(train_features_df.loc[train_idx, target_cols].values, dtype=float)
    p = torch.tensor(oof_arr, dtype=float)
    p = torch.clamp(p, 1e-9, 1 - (1e-9))
    competition_metric = nn.BCELoss()(p, y).item()
    return np.mean(competition_metric)


def calc_competition_metric(train_features_df, target_cols, oof_arr):
    competition_metric = []
    for i in range(len(target_cols)):
        competition_metric.append(log_loss(train_features_df[target_cols[i]], oof_arr[:, i]))
    tprint(f"competition metric: {np.mean(competition_metric)}")


def get_best_weights(oof_1, oof_2, train_features_df, stage_1_2_target_cols):
    weight_list = []
    weights = np.array([0.5])
    for n_splits in tqdm(range(10, 11)):
        for i in range(2):
            kf = KFold(n_splits=n_splits, random_state=i, shuffle=True)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=oof_1)):
                res = minimize(
                    get_score,
                    weights,
                    args=(train_features_df, train_idx, oof_1, oof_2, stage_1_2_target_cols),
                    method="Nelder-Mead",
                    tol=1e-6,
                )
                tprint(f"i: {i} fold: {fold} res.x: {res.x}")
                weight_list.append(res.x)
    mean_weight = np.mean(weight_list)
    tprint(f"optimized weight: {mean_weight}")
    return mean_weight


def get_cp_time_feature(s):
    if s == 72:
        return 2
    elif s == 48:
        return 1
    else:
        return 0


def get_cp_dose_feature(s):
    return 1 if s == "D1" else 0


def get_fold(scored, stage_1_2_target_cols, n_splits=5, seed=42):
    folds = []
    scored = scored.copy()
    # LOCATE DRUGS
    vc = scored.drug_id.value_counts()

    vc1 = vc.loc[(vc == 6) | (vc == 12) | (vc == 18)].index.sort_values()
    vc2 = vc.loc[(vc != 6) & (vc != 12) & (vc != 18)].index.sort_values()

    # STRATIFY DRUGS 18X OR LESS
    dct1 = {}
    dct2 = {}
    skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tmp = scored.groupby("drug_id")[stage_1_2_target_cols].mean().loc[vc1]
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[stage_1_2_target_cols])):
        dd = {k: fold for k in tmp.index[idxV].values}
        dct1.update(dd)

    # STRATIFY DRUGS MORE THAN 18X
    skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tmp = scored.loc[scored.drug_id.isin(vc2)].reset_index(drop=True)
    for fold, (idxT, idxV) in enumerate(skf.split(tmp, tmp[stage_1_2_target_cols])):
        dd = {k: fold for k in tmp.sig_id[idxV].values}
        dct2.update(dd)

    # ASSIGN FOLDS
    scored["fold"] = scored.drug_id.map(dct1)
    scored.loc[scored.fold.isna(), "fold"] = scored.loc[scored.fold.isna(), "sig_id"].map(dct2)
    scored.fold = scored.fold.astype("int8")
    folds.append(scored.fold.values)

    return np.stack(folds).flatten()


def get_score(weights, train_features_df, train_idx, oof_1, oof_2, stage_1_2_target_cols):
    _oof_1 = oof_1[train_idx, :].copy()
    _oof_2 = oof_2[train_idx, :].copy()
    blend = (_oof_1 * weights[0]) + (_oof_2 * (1 - weights[0]))
    return _calc_competition_metric(train_features_df, stage_1_2_target_cols, blend, train_idx)


def torch_clear():
    cuda.select_device(0)
    cuda.close()


class SVMTrainer(object):
    def __init__(self, predictors, targets, X, n_splits, n_rsb):
        self.predictors = predictors
        self.targets = targets
        self.X = X
        self.n_splits = n_splits
        self.n_rsb = n_rsb
        self.validation_score = []
        self.folds = []
        self.oof = []

    def _fit(self, X_train, y_train, X_valid, y_valid, target, fold, seed):
        seed_everything(seed=seed)
        target_oof = np.zeros(len(self.X))

        # BUG: 以下が起動ごとに全errorしたりしなかったり確率的にコケる (以下参照)
        # なかなか解決しないので、何回か回してコケなかったら通すようにする
        # https://kaggler-ja.slack.com/archives/G01F5QWJ5SM/p1605936016147700
        try:
            clf = SVC(cache_size=6000, probability=True)
            clf.fit(X_train[self.predictors].values, y_train.values, convert_dtype=False)
            pred_valid = clf.predict_proba(X_valid[self.predictors].values)[:, 1]
            model_path = f"{MODEL_DIR}/stage2_svm_{target}_fold{fold+1}_model_seed_{seed}.model.joblib"
            tprint(f"output model -> {model_path}")
            dump(clf, model_path)

        except:
            pred_valid = np.zeros(X_valid.shape[0])
            tprint("error")

        return pred_valid

    def fit(self):
        for i in tqdm(range(len(self.targets))):
            target = self.targets[i]
            target_oof = np.zeros(len(self.X))

            # DEBUGでは一部のtargetのみを学習
            if config_d["debug"] and i >= 3:
                self.oof.append(target_oof / self.n_rsb)
                continue

            fold = get_fold(self.X, self.targets, n_splits=self.n_splits, seed=MYSEED)
            for k in range(self.n_splits):
                train_idx = np.where(fold != k)[0]
                valid_idx = np.where(fold == k)[0]
                X_train, X_valid = self.X.loc[train_idx], self.X.loc[valid_idx]
                y_train, y_valid = self.X.loc[train_idx, target], self.X.loc[valid_idx, target]
                for rsb_idx in range(self.n_rsb):
                    pred_valid = self._fit(
                        X_train=X_train,
                        y_train=y_train,
                        X_valid=X_valid,
                        y_valid=y_valid,
                        target=target,
                        fold=k,
                        seed=rsb_idx,
                    )
                    target_oof[X_valid.index] = pred_valid / self.n_rsb
            loss = log_loss(self.X[target], target_oof)
            tprint(f"{target} oof logloss : {loss:.5f}")
            self.oof.append(target_oof / self.n_rsb)

        self.oof = np.stack(self.oof).T

    def predict(self, X_test):
        pred = []
        for i in tqdm(range(len(self.targets))):
            target = self.targets[i]
            target_pred = np.zeros(len(X_test))
            for k in range(self.n_splits):
                for rsb_idx in range(self.n_rsb):
                    try:
                        model_path = f"{MODEL_DIR}/stage2_svm_{target}_fold{k+1}_model_seed_{rsb_idx}.model.joblib"
                        clf = load(model_path)
                        tprint(f"loaded: {model_path}")

                        target_pred += clf.predict_proba(X_test[self.predictors])[:, 1] / (self.n_splits * self.n_rsb)
                    except:
                        target_pred += np.zeros(len(X_test))
            pred.append(target_pred)
        pred = np.stack(pred).T
        return pred


class NNTrainer(object):
    def __init__(self, predictors, targets, X, n_splits, n_rsb, config_d, stage, model_class):
        self.predictors = predictors
        self.targets = targets
        self.X = X
        self.n_splits = n_splits
        self.n_rsb = n_rsb
        self.validation_score = []
        self.folds = []
        self.config_d = config_d
        self.stage = stage
        self.oof = np.zeros((len(X), len(targets)))
        self.model_class = model_class

    def _fit(self, X_train, y_train, X_valid, y_valid, fold, seed):
        seed_everything(seed=seed)

        learning_rate = self.config_d["lr"]
        batch_size = self.config_d["batch_size"]
        epoch = self.config_d["epoch"]

        net = self.model_class(self.predictors, self.targets).to(device)
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)
        valid_criterion = nn.BCEWithLogitsLoss()
        criterion = SmoothBCEwLogits(smoothing=0.001)

        scheduler = MultiStepLR(optimizer, milestones=[10, 15], gamma=0.1)

        # 学習時はlength=1の破片などを回避するためdrop_last=1とする
        train_dataset = TabularDataset(X_train, True, self.targets, self.predictors)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        valid_dataset = TabularDataset(X_valid, True, self.targets, self.predictors)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        bar = trange(epoch, desc=f"seed: {seed} fold : {fold+1} train : {X_train.shape[0]}  valid:{X_valid.shape[0]}====")
        train_loss = []
        valid_loss = []

        best_loss = 10000
        best_loss_epoch = 1

        for e in bar:
            running_loss = []
            valid_loss = []

            # train
            net.train()
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                out = net(x)
                loss = criterion(out, y)
                loss.backward()
                running_loss.append(loss.item())
                optimizer.step()
            scheduler.step()

            net.eval()

            tmp_oof = []
            with torch.no_grad():
                for x, y in valid_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    out = net(x)
                    loss = valid_criterion(out, y)
                    tmp_oof.append(out.sigmoid().detach().cpu().numpy())
                    valid_loss.append(loss.item())

                bar.set_postfix(
                    running_loss=f"{np.mean(running_loss):.5f}",
                    valid_loss=f"{np.mean(valid_loss):.5f}",
                    best_loss=f"{best_loss:.5f}",
                    best_loss_epoch=f"{best_loss_epoch}",
                )

            train_loss.append(np.mean(running_loss))
            valid_loss.append(np.mean(valid_loss))

            if best_loss > np.mean(valid_loss):
                best_loss = np.mean(valid_loss)
                best_loss_epoch = e + 1
                self.oof[X_valid.index] = np.concatenate(tmp_oof) / self.n_rsb

                model_path = f"{MODEL_DIR}/stage{self.stage}_fold{fold+1}_best_model_seed_{seed}.pt"
                # tprint(f"output model -> {model_path}")
                torch.save(net.state_dict(), model_path)

        tprint(f"best loss : {best_loss}")
        # stage1_1_rsb_nn_oof += stage1_1_nn_oof / len(random_seeds)

    def fit(self):
        fold = get_fold(self.X, self.targets, n_splits=self.n_splits, seed=MYSEED)
        for k in range(self.n_splits):
            train_idx = np.where(fold != k)[0]
            valid_idx = np.where(fold == k)[0]
            X_train, X_valid = self.X.loc[train_idx], self.X.loc[valid_idx]
            y_train, y_valid = self.X.loc[train_idx, self.targets], self.X.loc[valid_idx, self.targets]

            for rsb_idx in range(self.n_rsb):
                self._fit(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, fold=k, seed=rsb_idx)

    def predict(self, X_test):
        pred = np.zeros((len(X_test), len(self.targets)))
        for k in range(self.n_splits):
            for rsb_idx in range(self.n_rsb):
                model_path = f"{MODEL_DIR}/stage{self.stage}_fold{k+1}_best_model_seed_{rsb_idx}.pt"

                net = self.model_class(self.predictors, self.targets).to(device)
                net.load_state_dict(torch.load(model_path))
                net.eval()
                tprint(f"loaded: {model_path}")

                test_dataset = TabularDataset(X_test, False, self.targets, self.predictors)
                test_dataloader = DataLoader(test_dataset, batch_size=self.config_d["batch_size"], shuffle=False)

                tmp_pred = []
                with torch.no_grad():
                    for x in test_dataloader:
                        x = x.to(device)
                        out = net(x)
                        tmp_pred.append(out.sigmoid().detach().cpu().numpy())

                pred += np.concatenate(tmp_pred) / (self.n_splits * self.n_rsb)
        return pred


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1), self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class TabularDataset(Dataset):
    def __init__(self, df, is_train, targets, features):
        self.df = df
        self.is_train = is_train
        self.features = features
        self.X = df[features].values
        if self.is_train:
            self.y = df[targets].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.is_train:
            return (
                torch.tensor(self.X[idx], dtype=torch.float).to(device),
                torch.tensor(self.y[idx], dtype=torch.float).to(device),
            )
        else:
            return torch.tensor(self.X[idx], dtype=torch.float).to(device)


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


def common_preprocess():
    # train/predに共通する読み込みを実施
    train_features_df = pd.read_csv(f"{INPUT_DIR}/train_features.csv")
    test_features_df = pd.read_csv(f"{INPUT_DIR}/test_features.csv")
    train_drug_df = pd.read_csv(f"{INPUT_DIR}/train_drug.csv")
    train_targets_scored_df = pd.read_csv(f"{INPUT_DIR}/train_targets_scored.csv")
    train_targets_nonscored_df = pd.read_csv(f"{INPUT_DIR}/train_targets_nonscored.csv")
    sample_submission_df = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

    tprint(
        f"""
    [初回読み込み]
    train_features_df: {train_features_df.shape}
    train_drug_df: {train_drug_df.shape}
    train_targets_scored_df: {train_targets_scored_df.shape}
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    test_features_df: {test_features_df.shape}
    sample_submission_df: {sample_submission_df.shape}
    """
    )

    # nonscoreで全部0のラベルは外す
    drop_cols = list(train_targets_nonscored_df.columns[train_targets_nonscored_df.sum() == 0])
    use_cols = [x for x in train_targets_nonscored_df.columns if x not in drop_cols]
    train_targets_nonscored_df = train_targets_nonscored_df.loc[:, use_cols]
    tprint(
        f"""
    [全部0のラベル除去]
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    """
    )

    # indexを使ってmerge
    # chainで書いてしまうとformatが狂ったり大変だったため、1行ずつに...
    train_features_df = train_features_df.merge(train_targets_scored_df)
    train_features_df = train_features_df.merge(train_drug_df)
    train_features_df = train_features_df.merge(train_targets_nonscored_df)
    tprint(
        f"""
    [merge後]
    train_features_df: {train_features_df.shape}
    """
    )

    # cp_type=ctl_vehicleのデータは学習から外す
    train_features_df = train_features_df[train_features_df.cp_type == "trt_cp"].reset_index(drop=True)

    # cp系特徴量追加
    train_features_df["cp_time_feature"] = train_features_df["cp_time"].map(get_cp_time_feature)
    test_features_df["cp_time_feature"] = test_features_df["cp_time"].map(get_cp_time_feature)
    train_features_df["cp_dose_feature"] = train_features_df["cp_dose"].map(get_cp_dose_feature)
    test_features_df["cp_dose_feature"] = test_features_df["cp_dose"].map(get_cp_dose_feature)

    # 不要カラム除去
    train_features_df = train_features_df.drop(columns=["cp_type", "cp_time", "cp_dose"])

    # g-, c-系特徴量取得
    features_g = list([x for x in train_features_df.columns if x.startswith("g-")])
    features_c = list([x for x in train_features_df.columns if x.startswith("c-")])

    if config_d["debug"]:
        # 算出済みのtrain_features_df, test_features_dfを読み込み
        train_features_df = pd.read_pickle(f"{CACHE_DIR}/train_features_df.pickle")
        test_features_df = pd.read_pickle(f"{CACHE_DIR}/test_features_df.pickle")
    elif config_d["phase"] == "pred":
        # trainについては学習済みのものを読み込む
        train_features_df = pd.read_pickle(f"{CACHE_DIR}/train_features_df.pickle")

        # これに従い、必要なカラムのみを作る
        for c in tqdm(list(itertools.combinations(features_g + features_c, 2))):
            col_name = f"{c[0]}_{c[1]}_diff"
            if col_name in train_features_df.columns:
                test_features_df[col_name] = test_features_df[c[0]] - test_features_df[c[1]]
    else:
        # total確認したいのでlistにして組み合わせを回す (メモリも全然食わないので)
        var_list = []
        for c in tqdm(list(itertools.combinations(features_g + features_c, 2))):
            col_name = f"{c[0]}_{c[1]}_diff"
            d = train_features_df[c[0]] - train_features_df[c[1]]
            diff_val = np.var(d)
            if diff_val > 15:
                train_features_df[col_name] = d
                var_list.append(diff_val)
        train_features_df.to_pickle(f"{CACHE_DIR}/train_features_df.pickle")

    stage_1_2_target_cols = [x for x in train_targets_scored_df.columns if x not in ["sig_id", "drug_id"]]
    stage_1_1_target_cols = [
        x for x in train_targets_nonscored_df.columns if x not in ["sig_id", "drug_id"]
    ] + stage_1_2_target_cols

    stage_1_train_features = [
        x for x in train_features_df.columns if x not in ["sig_id", "drug_id"] + stage_1_1_target_cols
    ]

    return (
        train_features_df,
        test_features_df if config_d["phase"] == "pred" else None,
        sample_submission_df if config_d["phase"] == "pred" else None,
        stage_1_1_target_cols,
        stage_1_2_target_cols,
        stage_1_train_features,
    )


def exec_train():
    (
        train_features_df,
        _,
        _,
        stage_1_1_target_cols,
        stage_1_2_target_cols,
        stage_1_train_features,
    ) = common_preprocess()

    # Create object for transformation.
    transformer = QuantileTransformer(n_quantiles=100, random_state=42, output_distribution="normal")

    # fitting
    transformer.fit(train_features_df.loc[:, stage_1_train_features])
    dump(transformer, f"{MODEL_DIR}/transformer.joblib")

    # transforming
    train_features_df[stage_1_train_features] = transformer.transform(train_features_df.loc[:, stage_1_train_features])

    tprint("--- stage 1-1: predict scored + nonscored target ---")
    stage1_1_nn_trainer = NNTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_1_target_cols,
        X=train_features_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=1,
        model_class=TabularMLP_1_1,
    )
    stage1_1_nn_trainer.fit()

    tprint("[metric: stage1_1_rsb_nn_oof]")
    calc_competition_metric(train_features_df, stage_1_1_target_cols, stage1_1_nn_trainer.oof)

    tprint(
        f"""
    stage1_1_rsb_nn_oof: {stage1_1_nn_trainer.oof.shape}
    """
    )

    tprint("--- stage 1-2: predict scored target ---")
    stage1_2_svm_trainer = SVMTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_2_target_cols,
        X=train_features_df,
        n_splits=4 if not config_d["debug"] else 2,
        n_rsb=1,
    )
    stage1_2_svm_trainer.fit()

    competition_metric = []
    for i in range(len(stage_1_2_target_cols)):
        competition_metric.append(log_loss(train_features_df[stage_1_2_target_cols[i]], stage1_2_svm_trainer.oof[:, i]))
    tprint(f"competition_metric: {np.mean(competition_metric)}")
    stage1_2_nn_trainer = NNTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_2_target_cols,
        X=train_features_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=2,
        model_class=TabularMLP_1_2,
    )
    stage1_2_nn_trainer.fit()

    tprint("[metric: stage1_1_rsb_nn_oof]")
    calc_competition_metric(train_features_df, stage_1_1_target_cols, stage1_1_nn_trainer.oof)

    tprint("[metric: stage1_2_rsb_nn_oof]")
    calc_competition_metric(train_features_df, stage_1_2_target_cols, stage1_2_nn_trainer.oof)

    tprint("[metric: stage1_2_rsb_svm_oof]")
    calc_competition_metric(train_features_df, stage_1_2_target_cols, stage1_2_svm_trainer.oof)

    tprint("--- stage 2 ---")
    tprint(
        f"""
    [stage1-1]
    stage1_1_rsb_nn_oof: {stage1_1_nn_trainer.oof.shape}

    [stage1-2]
    stage1_2_rsb_nn_oof: {stage1_2_nn_trainer.oof.shape}
    """
    )

    stack_train_df = pd.concat(
        [
            pd.DataFrame(stage1_1_nn_trainer.oof).add_prefix("stage1_1_nn_"),
            #     pd.DataFrame(stage1_2_rsb_svm_oof).add_prefix('stage1_2_svm_'),
            pd.DataFrame(stage1_2_nn_trainer.oof).add_prefix("stage1_2_nn_"),
            train_features_df[stage_1_2_target_cols + ["drug_id", "sig_id"]],
        ],
        axis=1,
    )
    stack_train_df.to_pickle(f"{CACHE_DIR}/stack_train_df.pickle")

    tprint(
        f"""
    stack_train_df: {stack_train_df.shape}
    """
    )

    stack_train_features = [x for x in stack_train_df.columns if x not in stage_1_2_target_cols + ["drug_id", "sig_id"]]
    stack_trainer = NNTrainer(
        predictors=stack_train_features,
        targets=stage_1_2_target_cols,
        X=stack_train_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=3,
        model_class=TabularMLP_2,
    )
    stack_trainer.fit()

    tprint("[metric: rsb_stack_oof]")
    calc_competition_metric(train_features_df, stage_1_2_target_cols, stack_trainer.oof)

    tprint("--- blend ---")
    tprint(
        f"""
    rsb_stack_oof: {stack_trainer.oof.shape}
    stage1_2_rsb_svm_oof: {stage1_2_svm_trainer.oof.shape}
    """
    )

    best_weight = get_best_weights(
        stack_trainer.oof, stage1_2_svm_trainer.oof, train_features_df, stage_1_2_target_cols
    )
    dump(best_weight, f"{MODEL_DIR}/best_weight.joblib")
    tprint(f"best_weight: {best_weight}")

    blend_oof = stack_trainer.oof * best_weight + stage1_2_svm_trainer.oof * (1 - best_weight)

    tprint("[metric: blend_oof]")
    calc_competition_metric(train_features_df, stage_1_2_target_cols, blend_oof)


def exec_pred():
    (
        train_features_df,
        test_features_df,
        sample_submission_df,
        stage_1_1_target_cols,
        stage_1_2_target_cols,
        stage_1_train_features,
    ) = common_preprocess()

    # transformer読み込み、実行
    # TODO: trainについても再実行する必要はあるか？
    transformer = load(f"{MODEL_DIR}/transformer.joblib")
    train_features_df[stage_1_train_features] = transformer.transform(train_features_df.loc[:, stage_1_train_features])
    test_features_df[stage_1_train_features] = transformer.transform(test_features_df.loc[:, stage_1_train_features])

    tprint("--- stage 1-1: predict scored + nonscored target ---")
    stage1_1_nn_trainer = NNTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_1_target_cols,
        X=train_features_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=1,
        model_class=TabularMLP_1_1,
    )
    stage1_1_rsb_nn_pred = stage1_1_nn_trainer.predict(test_features_df)

    tprint("--- stage 1-2: predict scored target ---")
    stage1_2_svm_trainer = SVMTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_2_target_cols,
        X=train_features_df,
        n_splits=4 if not config_d["debug"] else 2,
        n_rsb=1,
    )
    stage1_2_rsb_svm_pred = stage1_2_svm_trainer.predict(test_features_df)

    stage1_2_nn_trainer = NNTrainer(
        predictors=stage_1_train_features,
        targets=stage_1_2_target_cols,
        X=train_features_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=2,
        model_class=TabularMLP_1_2,
    )
    stage1_2_rsb_nn_pred = stage1_2_nn_trainer.predict(test_features_df)

    tprint("--- stage 2 ---")
    tprint(
        f"""
    [stage1-1]
    stage1_1_rsb_nn_pred: {stage1_1_rsb_nn_pred.shape}

    [stage1-2]
    stage1_2_rsb_nn_pred: {stage1_2_rsb_nn_pred.shape}
    """
    )

    stack_train_df = pd.read_pickle(f"{CACHE_DIR}/stack_train_df.pickle")
    stack_test_df = pd.concat(
        [
            pd.DataFrame(stage1_1_rsb_nn_pred).add_prefix("stage1_1_nn_"),
            #     pd.DataFrame(stage1_2_rsb_svm_pred).add_prefix('stage1_2_svm_'),
            pd.DataFrame(stage1_2_rsb_nn_pred).add_prefix("stage1_2_nn_"),
        ],
        axis=1,
    )

    stack_train_features = [x for x in stack_train_df.columns if x not in stage_1_2_target_cols + ["drug_id", "sig_id"]]

    stack_trainer = NNTrainer(
        predictors=stack_train_features,
        targets=stage_1_2_target_cols,
        X=stack_train_df,
        n_splits=7 if not config_d["debug"] else 2,
        n_rsb=5 if not config_d["debug"] else 1,
        config_d={
            "epoch": 20 if not config_d["debug"] else 1,
            "lr": 0.01,
            "batch_size": 128,
        },
        stage=3,
        model_class=TabularMLP_2,
    )
    rsb_stack_pred = stack_trainer.predict(stack_test_df)
    print(rsb_stack_pred.shape)

    best_weight = load(f"{MODEL_DIR}/best_weight.joblib")
    blend_pred = rsb_stack_pred * best_weight + stage1_2_rsb_svm_pred * (1 - best_weight)
    test_features_df[stage_1_2_target_cols] = blend_pred

    # post process
    test_features_df.loc[test_features_df["cp_type"] != "trt_cp", stage_1_2_target_cols] = 0

    sub = (
        sample_submission_df.drop(columns=stage_1_2_target_cols)
        .merge(test_features_df[["sig_id"] + stage_1_2_target_cols], on="sig_id", how="left")
        .fillna(0)
    )
    sub.to_csv(f"{RESULT_DIR}/submission.csv", index=False)


if __name__ == "__main__":
    if config_d["phase"] == "train":
        exec_train()
    elif config_d["phase"] == "pred":
        exec_pred()
    else:
        raise NotImplementedError