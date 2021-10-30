import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("../gas")
sys.path.append("../chugun")
sys.path.append("../lomproduv")
sys.path.append("../chronom")

from plavki import process_plavki
from features import generate_gas_features
from features_chugun import process_data
from features_lomproduv import lom_transform, produv_transform
from features_chronom import chron_features


def merge_data(sample, target, plavki_train, plavki_test, gas_train, gas_test, chugun_train, 
               chugun_test, lom_train, lom_test, produv_train, produv_test, 
               chronom_train, chronom_test, params):
    plavki_data, plavki_num_features, plavki_cat_features = process_plavki(plavki_train, plavki_test, **params["plavki"])
    gas_train, gas_test = generate_gas_features(gas_train), generate_gas_features(gas_test)
    chugun_data, chugun_num_features, chugun_cat_features = process_data(chugun_train, chugun_test, **params["chugun"])
    lom_data, lom_num_features, lom_cat_features = lom_transform(lom_train, lom_test, chugun_train, chugun_test)
    produv_data, produv_num_features, produv_cat_features = produv_transform(produv_train, produv_test)
    chronom_train, chronom_test, chronom_num_features, chronom_cat_features = \
        chron_features(chronom_train, chronom_test, plavki_train, plavki_test)
    
    gas_num_features = list(gas_train.columns)
    gas_cat_features = []

    train = target[["NPLV"]][~target["C"].isna()]
    test = sample[["NPLV"]]

    train = train.join(gas_train, on="NPLV")
    train = train.join(plavki_data.set_index("NPLV")[plavki_num_features + plavki_cat_features], on="NPLV")
    train = train.join(chugun_data.set_index("NPLV")[chugun_num_features + chugun_cat_features], on="NPLV")
    train = train.join(lom_data[lom_num_features + lom_cat_features], on="NPLV")
    train = train.join(produv_data[produv_num_features + produv_cat_features], on="NPLV", rsuffix="produv_")
    train = train.join(chronom_train[chronom_num_features + chronom_cat_features], on="NPLV")

    test = test.join(gas_test, on="NPLV")
    test = test.join(plavki_data.set_index("NPLV")[plavki_num_features + plavki_cat_features], on="NPLV")
    test = test.join(chugun_data.set_index("NPLV")[chugun_num_features + chugun_cat_features], on="NPLV")
    test = test.join(lom_data[lom_num_features + lom_cat_features], on="NPLV")
    test = test.join(produv_data[produv_num_features + produv_cat_features], on="NPLV", rsuffix="produv_")
    test = test.join(chronom_test[chronom_num_features + chronom_cat_features], on="NPLV")

    train.set_index("NPLV", inplace=True)
    test.set_index("NPLV", inplace=True)
    y = target.set_index("NPLV").loc[train.index]

    num_features = gas_num_features + plavki_num_features + chugun_num_features + lom_num_features + \
        produv_num_features + chronom_num_features
    cat_features = gas_cat_features + plavki_cat_features + chugun_cat_features + lom_cat_features + \
        produv_cat_features + chronom_cat_features

    return train, test, y, num_features, cat_features


def train_eval_split(train, y, test_size=0.2):
    train_len = int(train.shape[0] * (1 - test_size))
    return train[:train_len], train[train_len:], y[:train_len], y[train_len:]


def transform_for_linear(train, num_features, cat_features, eval_data=None, test_data=None):
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    res = [column_transformer.fit_transform(train)]
    if eval_data is not None:
        res.append(column_transformer.transform(eval_data))

    if test_data is not None:
        res.append(column_transformer.transform(test_data))
    
    return res