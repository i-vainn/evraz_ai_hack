import pandas as pd


def produv_transform(produv_train, produv_test):
    produv_train_transformed = pd.DataFrame(index=produv_train.NPLV.unique())
    produv_train_transformed['duration'] = (produv_train.groupby('NPLV').SEC.max()- produv_train.groupby('NPLV').SEC.min()).dt.total_seconds().values
    produv_train_transformed['RAS_mean'] = produv_train.groupby('NPLV').RAS.mean().values
    produv_train_transformed['POL_mean'] = produv_train.groupby('NPLV').POL.mean().values

    produv_test_transformed = pd.DataFrame(index=produv_test.NPLV.unique())
    produv_test_transformed['duration'] = (produv_test.groupby('NPLV').SEC.max()- produv_test.groupby('NPLV').SEC.min()).dt.total_seconds().values
    produv_test_transformed['RAS_mean'] = produv_test.groupby('NPLV').RAS.mean().values
    produv_test_transformed['POL_mean'] = produv_test.groupby('NPLV').POL.mean().values

    produv_transformed = pd.concat([produv_train_transformed, produv_test_transformed])

    return produv_transformed


def lom_transform(lom_train, lom_test):
    lom_train_transformed = pd.DataFrame(index=lom_train.NPLV.unique())
    lom_train_transformed['ves_loma'] = lom_train.groupby('NPLV').VES.sum()
    lom_train_transformed['ves_loma/ves_chuguna'] = lom_train_transformed['ves_loma'].values/chugun_train['VES'].values

    lom_test_transformed = pd.DataFrame(index=lom_test.NPLV.unique())
    lom_test_transformed['ves_loma'] = lom_test.groupby('NPLV').VES.sum()
    lom_test_transformed['ves_loma/ves_chuguna'] = lom_test_transformed['ves_loma'].values/chugun_test['VES'].values

    lom_transformed = pd.concat([lom_train_transformed, lom_test_transformed])

    return lom_transformed
