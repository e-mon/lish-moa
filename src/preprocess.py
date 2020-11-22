from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np

from src.utils.cache import Cache
from src.utils.misc import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


def get_cp_time_feature(s):
    if s == 72:
        return 2
    elif s == 48:
        return 1
    else:
        return 0


def get_cp_dose_feature(s):
    return 1 if s == "D1" else 0


@Cache(dir_path='./cache/')
def common(input_dir='../input/lish-moa/'):
    # train/predに共通する読み込みを実施
    train_features_df = pd.read_csv(f"{input_dir}/train_features.csv")
    test_features_df = pd.read_csv(f"{input_dir}/test_features.csv")
    train_drug_df = pd.read_csv(f"{input_dir}/train_drug.csv")
    train_targets_scored_df = pd.read_csv(f"{input_dir}/train_targets_scored.csv")
    train_targets_nonscored_df = pd.read_csv(f"{input_dir}/train_targets_nonscored.csv")
    sample_submission_df = pd.read_csv(f"{input_dir}/sample_submission.csv")

    logger.info(f"""
    [初回読み込み]
    train_features_df: {train_features_df.shape}
    train_drug_df: {train_drug_df.shape}
    train_targets_scored_df: {train_targets_scored_df.shape}
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    test_features_df: {test_features_df.shape}
    sample_submission_df: {sample_submission_df.shape}
    """)

    # nonscoreで全部0のラベルは外す
    drop_cols = list(train_targets_nonscored_df.columns[train_targets_nonscored_df.sum() == 0])
    use_cols = [x for x in train_targets_nonscored_df.columns if x not in drop_cols]
    train_targets_nonscored_df = train_targets_nonscored_df.loc[:, use_cols]
    logger.info(f"""
    [全部0のラベル除去]
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    """)

    # indexを使ってmerge
    # chainで書いてしまうとformatが狂ったり大変だったため、1行ずつに...
    train_features_df = train_features_df.merge(train_targets_scored_df)
    train_features_df = train_features_df.merge(train_drug_df)
    train_features_df = train_features_df.merge(train_targets_nonscored_df)
    logger.info(f"""
    [merge後]
    train_features_df: {train_features_df.shape}
    """)

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

    var_list = []
    for c in tqdm(list(itertools.combinations(features_g + features_c, 2))):
        col_name = f"{c[0]}_{c[1]}_diff"
        d = train_features_df[c[0]] - train_features_df[c[1]]
        diff_val = np.var(d)
        if diff_val > 15:
            train_features_df[col_name] = d
            var_list.append(diff_val)

    # これに従い、必要なカラムのみを作る
    for c in tqdm(list(itertools.combinations(features_g + features_c, 2))):
        col_name = f"{c[0]}_{c[1]}_diff"
        if col_name in train_features_df.columns:
            test_features_df[col_name] = test_features_df[c[0]] - test_features_df[c[1]]

    stage_1_2_target_cols = [x for x in train_targets_scored_df.columns if x not in ["sig_id", "drug_id"]]
    stage_1_1_target_cols = [x for x in train_targets_nonscored_df.columns if x not in ["sig_id", "drug_id"]] + stage_1_2_target_cols

    stage_1_train_features = [x for x in train_features_df.columns if x not in ["sig_id", "drug_id"] + stage_1_1_target_cols]

    return (
        train_features_df,
        test_features_df,
        sample_submission_df,
        stage_1_1_target_cols,
        stage_1_2_target_cols,
        stage_1_train_features,
    )
