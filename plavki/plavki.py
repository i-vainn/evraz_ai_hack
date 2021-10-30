import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer


def process_plavki(plavki_train, plavki_test, clip_duration=False, top_grades=25, bow_count=50):
    plavki_train["is_train"] = 1
    plavki_test["is_train"] = 0
    data = pd.concat([plavki_train.groupby("NPLV").first().reset_index(), plavki_test])
    data["plavka_VR_NACH"] = pd.to_datetime(data["plavka_VR_NACH"])
    data["plavka_VR_KON"] = pd.to_datetime(data["plavka_VR_KON"])
    data["duration"] = (data["plavka_VR_KON"] - data["plavka_VR_NACH"]).dt.total_seconds()

    if clip_duration:
        data = data[(data["is_train"] == 0) | (data["duration"] < 6000)]
        data["duration"] = data["duration"].clip(0, 6000)

    data["dayofweek"] = data["plavka_VR_NACH"].dt.dayofweek
    data["dayofmonth"] = data["plavka_VR_NACH"].dt.day
    data["hour"] = data["plavka_VR_NACH"].dt.hour

    data["st_diff_is_zero"] = (data["plavka_ST_GOL"] == data["plavka_ST_FURM"]).astype(int)

    vect = CountVectorizer(lowercase=False, analyzer="char", ngram_range=(3, 3), max_features=bow_count)
    bow = pd.DataFrame(vect.fit_transform(data["plavka_NMZ"]).toarray(),
                       columns=["bow_" + el for el in vect.get_feature_names()])
    data = pd.concat([data.reset_index(drop=True), bow], axis=1)

    good_grades = data["plavka_NMZ"][data["is_train"] == 1].value_counts()[:top_grades].index

    data["truncated_NMZ"] = data["plavka_NMZ"].copy()
    data.loc[~data["plavka_NMZ"].isin(good_grades), "truncated_NMZ"] = "other"

    cat_features = ["truncated_NMZ", "st_diff_is_zero", "dayofweek", "plavka_TIPE_GOL", "plavka_TIPE_FUR",
                "plavka_NAPR_ZAD"] + ["bow_" + el for el in vect.get_feature_names()]
    num_features = ["plavka_STFUT", "plavka_ST_FURM", "plavka_ST_GOL", "dayofmonth", "hour", "duration"]
    return data, num_features, cat_features
