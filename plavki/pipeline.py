import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from gensim.models import Word2Vec

import sys
sys.path.append("../gas")
sys.path.append("../chugun")
sys.path.append("../lomproduv")
sys.path.append("../chronom")
sys.path.append("../sip")

from plavki import process_plavki
from features import generate_gas_features
from features_chugun import process_data
from features_edited import lom_transform, produv_transform
from features_chronom import chron_features
from sip_features import sip_features


def tokens2m(tokens, model_wv):
    res = []
    for token in tokens:
        idx = model_wv.wv.key_to_index[token]
        res.append(model_wv.wv[idx])
        
    return np.mean(np.array(res), axis=0).tolist() + np.std(np.array(res), axis=0).tolist() + \
        [np.linalg.norm(np.array(res))]


def train_w2v(chronom_train, chronom_test, vector_size=10):
    chronom_train["concat"] = chronom_train["TYPE_OPER"] + "_" + chronom_train["NOP"] 
    chronom_test["concat"] = chronom_test["TYPE_OPER"] + "_" + chronom_test["NOP"] 

    train_texts = chronom_train.groupby("NPLV")["concat"].apply(list)
    test_texts = chronom_test.groupby("NPLV")["concat"].apply(list)

    texts_for_w2v = pd.concat([train_texts, test_texts])

    model_wv = Word2Vec(min_count=1, vector_size=vector_size)

    model_wv.build_vocab(list(texts_for_w2v), progress_per=200)

    model_wv.train(texts_for_w2v, total_examples = model_wv.corpus_count, 
                epochs=10, report_delay=1)
    model_wv.init_sims(replace=True)

    train_mean = train_texts.apply(lambda x: tokens2m(x, model_wv))
    test_mean = test_texts.apply(lambda x: tokens2m(x, model_wv))

    train_mean = pd.DataFrame.from_dict(dict(zip(train_mean.index, train_mean.values))).T
    test_mean = pd.DataFrame.from_dict(dict(zip(test_mean.index, test_mean.values))).T

    train_mean.columns = ["w2v_" + str(i) for i in range(2 * vector_size + 1)]
    test_mean.columns = ["w2v_" + str(i) for i in range(2 * vector_size + 1)]

    train_mean = train_mean.reset_index().rename(columns={"index": "NPLV"}).set_index("NPLV")
    test_mean = test_mean.reset_index().rename(columns={"index": "NPLV"}).set_index("NPLV")

    return train_mean, test_mean

def train_w2v_sip(sip_train, sip_test, vector_size=10):
    sip_train["concat"] = sip_train["VDSYP"].astype(str) + "_" + sip_train["NMSYP"] 
    sip_test["concat"] = sip_test["VDSYP"].astype(str) + "_" + sip_test["NMSYP"] 

    train_texts = sip_train.groupby("NPLV")["concat"].apply(list)
    test_texts = sip_test.groupby("NPLV")["concat"].apply(list)

    texts_for_w2v = pd.concat([train_texts, test_texts])

    model_wv = Word2Vec(min_count=1, vector_size=vector_size)

    model_wv.build_vocab(list(texts_for_w2v), progress_per=200)

    model_wv.train(texts_for_w2v, total_examples = model_wv.corpus_count, 
                epochs=10, report_delay=1)
    model_wv.init_sims(replace=True)

    train_mean = train_texts.apply(lambda x: tokens2m(x, model_wv))
    test_mean = test_texts.apply(lambda x: tokens2m(x, model_wv))

    train_mean = pd.DataFrame.from_dict(dict(zip(train_mean.index, train_mean.values))).T
    test_mean = pd.DataFrame.from_dict(dict(zip(test_mean.index, test_mean.values))).T

    train_mean.columns = ["w2v_sip_" + str(i) for i in range(2 * vector_size + 1)]
    test_mean.columns = ["w2v_sip_" + str(i) for i in range(2 * vector_size + 1)]

    train_mean = train_mean.reset_index().rename(columns={"index": "NPLV"}).set_index("NPLV")
    test_mean = test_mean.reset_index().rename(columns={"index": "NPLV"}).set_index("NPLV")

    return train_mean, test_mean



def merge_data(sample, target, plavki_train, plavki_test, gas_train, gas_test, chugun_train, 
               chugun_test, lom_train, lom_test, produv_train, produv_test, 
               chronom_train, chronom_test, sip_train, sip_test, params):
    vector_size = params["vector_size"]

    plavki_data, plavki_num_features, plavki_cat_features = process_plavki(plavki_train, plavki_test, **params["plavki"])
    gas_train, gas_test = generate_gas_features(gas_train), generate_gas_features(gas_test)
    chugun_data, chugun_num_features, chugun_cat_features = process_data(chugun_train, chugun_test, **params["chugun"])
    lom_data, lom_num_features, lom_cat_features = lom_transform(lom_train, lom_test, chugun_train, chugun_test)
    produv_data, produv_num_features, produv_cat_features = produv_transform(produv_train, produv_test, 
        chronom_train, chronom_test)

    train_mean, test_mean = train_w2v(chronom_train, chronom_test, vector_size=vector_size)
    train_mean_sip, test_mean_sip = train_w2v_sip(sip_train, sip_test, vector_size=vector_size)

    chronom_train, chronom_test, chronom_num_features, chronom_cat_features = \
        chron_features(chronom_train, chronom_test, plavki_train, plavki_test)
    sip_train, sip_test, sip_num_features, sip_cat_features = sip_features(chugun_train, chugun_test, \
        sip_train, sip_test)
    
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
    train = train.join(sip_train[sip_num_features + sip_cat_features], on="NPLV")
    train = train.join(train_mean, on="NPLV")
    train = train.join(train_mean_sip, on="NPLV")

    test = test.join(gas_test, on="NPLV")
    test = test.join(plavki_data.set_index("NPLV")[plavki_num_features + plavki_cat_features], on="NPLV")
    test = test.join(chugun_data.set_index("NPLV")[chugun_num_features + chugun_cat_features], on="NPLV")
    test = test.join(lom_data[lom_num_features + lom_cat_features], on="NPLV")
    test = test.join(produv_data[produv_num_features + produv_cat_features], on="NPLV", rsuffix="produv_")
    test = test.join(chronom_test[chronom_num_features + chronom_cat_features], on="NPLV")
    test = test.join(sip_test[sip_num_features + sip_cat_features], on="NPLV")
    test = test.join(test_mean, on="NPLV")
    test = test.join(test_mean_sip, on="NPLV")

    train.set_index("NPLV", inplace=True)
    test.set_index("NPLV", inplace=True)
    y = target.set_index("NPLV").loc[train.index]

    num_features = gas_num_features + plavki_num_features + chugun_num_features + lom_num_features + \
        produv_num_features + chronom_num_features + sip_num_features + list(train_mean.columns) + \
        list(train_mean_sip.columns)
    cat_features = gas_cat_features + plavki_cat_features + chugun_cat_features + lom_cat_features + \
        produv_cat_features + chronom_cat_features + sip_cat_features

    return train, test, y, num_features, cat_features


def train_eval_split(train, y, test_size=0.2):
    train_len = int(train.shape[0] * (1 - test_size))
    return train[:train_len], train[train_len:], y[:train_len], y[train_len:]


def transform_for_linear(train, num_features, cat_features, eval_data=None, test_data=None):
    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='error', drop='first'), cat_features),
        ('scaling', StandardScaler(), num_features)
    ])

    res = [column_transformer.fit_transform(train)]
    if eval_data is not None:
        res.append(column_transformer.transform(eval_data))

    if test_data is not None:
        res.append(column_transformer.transform(test_data))
    
    return res