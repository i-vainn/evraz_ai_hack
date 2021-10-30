import numpy as np
import pandas as pd


import sys
sys.path.append("../gas")
sys.path.append("../chugun")

from plavki import process_plavki
from features import generate_gas_features
from features_chugun import process_data


def merge_data(sample, target, plavki_train, plavki_test, gas_train, gas_test, chugun_train, chugun_test, params):
    plavki_data, plavki_num_features, plavki_cat_features = process_plavki(plavki_train, plavki_test, **params["plavki"])
    gas_train, gas_test = generate_gas_features(gas_train), generate_gas_features(gas_test)
    chugun_data, chugun_num_features, chugun_cat_features = process_data(chugun_train, chugun_test, **params["chugun"])
    gas_num_features = list(gas_train.columns)
    gas_cat_features = []

    train = target[["NPLV"]][~target["C"].isna()]
    test = sample[["NPLV"]]

    train = train.join(gas_train, on="NPLV")
    train = train.join(plavki_data.set_index("NPLV")[plavki_num_features + plavki_cat_features], on="NPLV")
    train = train.join(chugun_data.set_index("NPLV")[chugun_num_features + chugun_cat_features], on="NPLV")

    test = test.join(gas_test, on="NPLV")
    test = test.join(plavki_data.set_index("NPLV")[plavki_num_features + plavki_cat_features], on="NPLV")
    test = test.join(chugun_data.set_index("NPLV")[chugun_num_features + chugun_cat_features], on="NPLV")

    train.set_index("NPLV", inplace=True)
    test.set_index("NPLV", inplace=True)
    y = target.set_index("NPLV").loc[train.index]

    num_features = gas_num_features + plavki_num_features + chugun_num_features
    cat_features = gas_cat_features + plavki_cat_features + chugun_cat_features 

    return train, test, y, num_features, cat_features


def train_eval_split(train, y, test_size=0.2):
    train_len = int(train.shape[0] * (1 - test_size))
    return train[:train_len], train[train_len:], y[:train_len], y[train_len:]
