import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import umap.umap_ as umap


def cut_produv(produv_train, produv_test, chronom_train, chronom_test):

    produv_train["SEC"] = pd.to_datetime(produv_train["SEC"])
    produv_test["SEC"] = pd.to_datetime(produv_test["SEC"])
    
    dates_end = chronom_train[chronom_train.NOP == 'Продувка']['VR_KON'].values
    j = 0
    indexes = []
    for i in tqdm(produv_train.NPLV.unique()):
        index = produv_train.loc[(produv_train.NPLV==i) & (produv_train.SEC < dates_end[j])].index
        for ind in index:
            indexes.append(ind)
        j += 1
    produv_train = produv_train.loc[indexes]
    produv_train['is_train'] = 1

    dates_end = chronom_test[chronom_test.NOP == 'Продувка']['VR_KON'].values
    j = 0
    indexes = []
    for i in tqdm(produv_test.NPLV.unique()):
        index = produv_test.loc[(produv_test.NPLV==i) & (produv_test.SEC < dates_end[j])].index
        for ind in index:
            indexes.append(ind)
        j += 1
    produv_test = produv_test.loc[indexes]
    produv_test['is_train'] = 0

    produv = pd.concat([produv_train, produv_test])
    return produv

def make_umap_produv(produv):
    
    time_series = {}
    for i in produv.NPLV.unique():
        time_series[i] = produv[produv.NPLV == i]['RAS'].values.tolist()

    length = []
    for i in time_series.values():
        length.append(len(i))
    
    max_len = 0
    for i in time_series.values():
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in time_series.values()])
    
    cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
                              n_components=10, random_state=42).fit_transform(padded)
    return cluster_embedding
    
def produv_transform(produv_train, produv_test, chromom_train, chronom_test):

    produv = cut_produv(produv_train, produv_test, chromom_train, chronom_test)
    cluster_embedding = make_umap_produv(produv)

    produv_transformed = pd.DataFrame(index=produv.NPLV.unique())
    produv_transformed['duration'] = (produv.groupby('NPLV').SEC.max()- produv.groupby('NPLV').SEC.min()).dt.total_seconds().values
    produv_transformed['RAS_mean'] = produv.groupby('NPLV').RAS.mean().values
    produv_transformed['POL_mean'] = produv.groupby('NPLV').POL.mean().values

    for i in range(cluster_embedding.shape[1]):
        produv_transformed[str(i)] = cluster_embedding[:, i] 

    return produv_transformed, ['duration', 'RAS_mean', 'POL_mean', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], []


def lom_transform(lom_train, lom_test, chugun_train, chugun_test):
    lom_train_transformed = pd.DataFrame(index=lom_train.NPLV.unique())
    lom_train_transformed['ves_loma'] = lom_train.groupby('NPLV').VES.sum()
    lom_train_transformed['ves_loma/ves_chuguna'] = lom_train_transformed['ves_loma'].values/chugun_train['VES'].values
    lom_train_transformed.loc[np.isinf(lom_train_transformed).any(axis=1), 'ves_loma/ves_chuguna'] = 0
    lom_train_transformed['is_train'] = 1

    lom_test_transformed = pd.DataFrame(index=lom_test.NPLV.unique())
    lom_test_transformed['ves_loma'] = lom_test.groupby('NPLV').VES.sum()
    lom_test_transformed['ves_loma/ves_chuguna'] = lom_test_transformed['ves_loma'].values/chugun_test['VES'].values
    lom_test_transformed.loc[np.isinf(lom_test_transformed).any(axis=1), 'ves_loma/ves_chuguna'] = 0
    lom_test_transformed['is_train'] = 0

    lom_transformed = pd.concat([lom_train_transformed, lom_test_transformed])

    return lom_transformed, ['ves_loma', 'ves_loma/ves_chuguna'], []
