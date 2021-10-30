import pandas as pd


def generate_gas_features(df):
    """
    Generate features based on gas data
    :param df: gas dataframe, train or test
    :return: dataframe with features with NPLV as index
    """
    duration_feature = df[["NPLV", "Time"]].groupby(by="NPLV").count().rename(columns={"Time": "gas_duration"})

    mean_features = df.groupby(by="NPLV").mean().rename(columns=lambda col : "gas_mean_" + col)
    std_features = df.groupby(by="NPLV").std().rename(columns=lambda col : "gas_std_" + col)
    
    fraction_columns = ['O2', 'N2', 'H2', 'CO2', 'CO', 'AR']
    volume_df = df[["NPLV"]].join(df[fraction_columns].multiply(df["V"], axis="index"))
    mean_volume_features = volume_df.groupby(by="NPLV").mean().rename(columns=lambda col : "gas_mean_volume_" + col)
    std_volume_features = volume_df.groupby(by="NPLV").std().rename(columns=lambda col : "gas_std_volume_" + col)
    
    return pd.concat([duration_feature, mean_features, std_features, mean_volume_features, std_volume_features], axis=1)