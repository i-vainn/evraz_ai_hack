import numpy as np


def sip_features(chugun_train, chugun_test, sip_train, sip_test):
    train = None
    test = None
    for i in ["train", "test"]:
        chugun = eval(f"chugun_{i}")
        sip = eval(f"sip_{i}")

        df = (
            sip.merge(chugun, on="NPLV", how="inner")
            .assign(type_syp=lambda x: x.VDSYP.astype("str") + "_" + x.NMSYP,
                    sip_ratio=lambda x: x.VSSYP / x.VES)
            .drop(columns=["VDSYP", "NMSYP"])

            .groupby("NPLV")
            .agg(
                min_mass=("VSSYP", "min"),
                max_mass=("VSSYP", "max"),

                total_count=("DAT_OTD", "count"),
                unique_count=("type_syp", "nunique"),

                min_ratio=("sip_ratio", "min"),
                max_ratio=("sip_ratio", "max")
            )
            .assign(unique_ratio=lambda x: x.unique_count / x.total_count)
        )
        df.loc[np.isinf(df["min_ratio"]), "min_ratio"] = 0
        df.loc[np.isinf(df["max_ratio"]), "max_ratio"] = 0

        if i == "train":
            train = df
        else:
            test = df

    return train, test, list(train.columns), []
