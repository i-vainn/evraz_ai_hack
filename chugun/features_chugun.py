import pandas as pd
import numpy as np

from datetime import datetime

def process_data(chugun_train, chugun_test, drop_outliers=False):
    train = chugun_train.copy()
    test = chugun_test.copy()
    if not drop_outliers:
        train.loc[train.VES==0, 'VES'] = train.VES.mean()
        test.loc[test.VES==0, 'VES'] = train.VES.mean()
        train.loc[train['T']==0, 'T'] = train['T'].mean()
        test.loc[test['T']==0, 'T'] = train['T'].mean()
    else:
        train = train[(train.VES!=0)&(train['T']!=0)]
        test.loc[test.VES==0, 'VES'] = train.VES.median()
        test.loc[test['T']==0, 'T'] = train['T'].median()

    train['is_train'] = 1
    test['is_train'] = 0
    train['DATA_ZAMERA'] = pd.to_datetime(train['DATA_ZAMERA'])
    test['DATA_ZAMERA'] = pd.to_datetime(test['DATA_ZAMERA'])
    data = pd.concat([train, test], ignore_index=True)
    
    elements = ['SI', 'MN', 'S', 'P', 'CR', 'NI', 'CU', 'V', 'TI']
    for col in elements:
        data[col.lower() + '_portion'] = data[col] / data['VES']
    
    data['total_seconds'] = (data.DATA_ZAMERA - datetime(1970, 1, 1)).dt.total_seconds()

    ok_cols = [
        'VES', 'T', 'SI', 'MN', 'S', 'P', 'CR', 'NI', 'CU', 'V', 'TI',
        *[col.lower() + '_portion' for col in elements],
    ]
    num_cols = ok_cols
    cat_cols = []
    
    return data[ok_cols + ['NPLV']], num_cols, cat_cols