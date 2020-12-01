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


def get_feature(df):
    features_g = list([x for x in df.columns if x.startswith("g-")])
    features_c = list([x for x in df.columns if x.startswith("c-")])

    df["g_sum"] = df[features_g].sum(axis=1)
    df["g_mean"] = df[features_g].mean(axis=1)
    df["g_median"] = df[features_g].median(axis=1)
    df["g_std"] = df[features_g].std(axis=1)
    df["g_kurt"] = df[features_g].kurtosis(axis=1)
    df["g_skew"] = df[features_g].skew(axis=1)
    df["c_sum"] = df[features_c].sum(axis=1)
    df["c_mean"] = df[features_c].mean(axis=1)
    df["c_std"] = df[features_c].std(axis=1)
    df["c_median"] = df[features_c].median(axis=1)
    df["c_kurt"] = df[features_c].kurtosis(axis=1)
    df["c_skew"] = df[features_c].skew(axis=1)
    df["gc_sum"] = df[features_g + features_c].sum(axis=1)
    df["gc_mean"] = df[features_g + features_c].mean(axis=1)
    df["gc_std"] = df[features_g + features_c].std(axis=1)
    df["gc_kurt"] = df[features_g + features_c].kurtosis(axis=1)
    df["gc_skew"] = df[features_g + features_c].skew(axis=1)
    df["gc_median"] = df[features_g + features_c].median(axis=1)

    return df


@Cache(dir_path='./cache/')
def preprocess_train(input_dir='../input/lish-moa/', sub: bool = False):
    train_features_df = pd.read_csv(f"{input_dir}/train_features.csv")
    train_drug_df = pd.read_csv(f"{input_dir}/train_drug.csv")
    train_targets_scored_df = pd.read_csv(f"{input_dir}/train_targets_scored.csv")
    train_targets_nonscored_df = pd.read_csv(f"{input_dir}/train_targets_nonscored.csv")

    logger.info(f"""
    train_features_df: {train_features_df.shape}
    train_drug_df: {train_drug_df.shape}
    train_targets_scored_df: {train_targets_scored_df.shape}
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    """)

    drop_cols = list(train_targets_nonscored_df.columns[train_targets_nonscored_df.sum() == 0])
    use_cols = [x for x in train_targets_nonscored_df.columns if x not in drop_cols]
    train_targets_nonscored_df = train_targets_nonscored_df.loc[:, use_cols]
    logger.info(f"""
    train_targets_nonscored_df: {train_targets_nonscored_df.shape}
    """)

    train_features_df = train_features_df.merge(train_targets_scored_df)
    train_features_df = train_features_df.merge(train_drug_df)
    train_features_df = train_features_df.merge(train_targets_nonscored_df)
    logger.info(f"""
    train_features_df: {train_features_df.shape}
    """)

    train_features_df = train_features_df[train_features_df.cp_type == "trt_cp"].reset_index(drop=True)

    train_features_df["cp_time_feature"] = train_features_df["cp_time"].map(get_cp_time_feature)
    train_features_df["cp_dose_feature"] = train_features_df["cp_dose"].map(get_cp_dose_feature)

    train_features_df = train_features_df.drop(columns=["cp_type", "cp_time", "cp_dose"])

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

    stage_1_2_target_cols = [x for x in train_targets_scored_df.columns if x not in ["sig_id", "drug_id"]]
    stage_1_1_target_cols = [x for x in train_targets_nonscored_df.columns if x not in ["sig_id", "drug_id"]] + stage_1_2_target_cols

    stage_1_train_features = [x for x in train_features_df.columns if x not in ["sig_id", "drug_id"] + stage_1_1_target_cols]

    return (
        train_features_df,
        stage_1_1_target_cols,
        stage_1_2_target_cols,
        stage_1_train_features,
    )


def preprocess_test(train_features_df, input_dir='../input/lish-moa/'):
    test_features_df = pd.read_csv(f"{input_dir}/test_features.csv")
    sample_submission_df = pd.read_csv(f"{input_dir}/sample_submission.csv")
    test_features_df["cp_time_feature"] = test_features_df["cp_time"].map(get_cp_time_feature)
    test_features_df["cp_dose_feature"] = test_features_df["cp_dose"].map(get_cp_dose_feature)

    features_g = list([x for x in train_features_df.columns if x.startswith("g-")])
    features_c = list([x for x in train_features_df.columns if x.startswith("c-")])

    for c in tqdm(list(itertools.combinations(features_g + features_c, 2))):
        col_name = f"{c[0]}_{c[1]}_diff"
        if col_name in train_features_df.columns:
            test_features_df[col_name] = test_features_df[c[0]] - test_features_df[c[1]]

    return test_features_df, sample_submission_df
