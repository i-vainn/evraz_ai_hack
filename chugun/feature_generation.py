from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def scoring(data, target, new_feature):
    X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)
    selection_model_c = XGBRegressor(n_estimators=50).fit(X_train.drop(new_feature, axis=1), y_train['C'])
    selection_model_t = XGBRegressor(n_estimators=50).fit(X_train.drop(new_feature, axis=1), y_train['TST'])
    small_pred = y_test.copy()
    small_pred['C'] = selection_model_c.predict(X_test.drop(new_feature, axis=1))
    small_pred['TST'] = selection_model_t.predict(X_test.drop(new_feature, axis=1))
    small_score = metric(y_test, small_pred)
    
    selection_model_c = XGBRegressor(n_estimators=50).fit(X_train, y_train['C'])
    selection_model_t = XGBRegressor(n_estimators=50).fit(X_train, y_train['TST'])
    big_pred = y_test.copy()
    big_pred['C'] = selection_model_c.predict(X_test)
    big_pred['TST'] = selection_model_t.predict(X_test)
    big_score = metric(y_test, big_pred)
    
    return np.array(big_score), np.array(small_score)

def feature_selection(data, initial_features='VES'):
    iters = 500
    logs = []
    operations = {
        '*': [2, np.multiply],
        '/': [2, np.divide],
        'log1p': [1, np.log1p],
        'sqrt': [1, np.sqrt],
        'square': [1, np.square],
        '': [1, lambda x: x]
    }
    cur_subset = list([initial_features])

    for _ in tqdm(range(iters)):
        op_name = np.random.choice(np.array(list(operations.keys())))
        nary, func = operations[op_name]

        new_features = np.random.choice(data.columns, size=nary, replace=False)
        
        if op_name in ['sqrt', 'log1p'] and (data[new_features[0]]<0).any()\
        or op_name in ['/'] and (data[new_features[0]]==0).any():
            continue
            
        if nary == 2:
            col_name = '#' + new_features[-1] + '$' + op_name + '$' + new_features[-2] + '#'
            if col_name in cur_subset:
                continue
            data[col_name] = func(data[new_features[-1]], data[new_features[-2]])
        if nary == 1:
            col_name = '#' + new_features[-1] + (('$' + op_name) if op_name else '' ) + '#'
            if col_name in cur_subset:
                continue
            data[col_name] = func(data[new_features[-1]].values)

        big_score, small_score = scoring(data[[col_name] + cur_subset], target, col_name)

        logs.append({
            'main_features': cur_subset,
            'other_feature': col_name,
            'big_c': big_score[0],
            'big_t': big_score[1],
            'big_mean': big_score[2],
            'small_c': small_score[0],
            'small_t': small_score[1],
            'small_mean': small_score[2]
        })

        if (big_score - small_score > 0.01).any():
            cur_subset.append(col_name)
        else:
            data = data.drop(col_name, axis=1)
    
    return logs, cur_subset