import pandas as pd


def generate_gas_features(df):
    """
    Generate features based on gas data
    :param df: gas dataframe, train or test
    :return: dataframe with features with NPLV as index
    """
    duration_feature = df[["NPLV", "Time"]].groupby(by="NPLV").count().rename(columns={"Time": "gas_duration"})
    mean_features = gas_test.groupby(by="NPLV").mean().rename(columns=lambda col : "gas_mean_" + col)
    std_features = gas_test.groupby(by="NPLV").std().rename(columns=lambda col : "gas_std_" + col)
    return pd.concat([duration_feature, mean_features, std_features], axis=1)