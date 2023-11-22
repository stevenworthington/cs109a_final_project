
# Function to calculate all performance metrics
def get_performance_metrics(model, classifier_name: str, data: tuple) -> dict:
    '''
    Parameters
    ----------
    model : (sklearn estimator) The fitted sklearn model
    classifier_name : (str) The name of the classifier used
    data : (tuple) Contains train and test split for X and y

    Returns:
    metrics : (dict)
    '''
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    X_train, X_test, y_train, y_test = data
    d = {}
    y_score_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = model.predict(X_train)
    y_score_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = model.predict(X_test)
    
    d['model'] = classifier_name
    d['train_AUC-ROC'] = roc_auc_score(y_train, y_score_train)
    d['train_AUC-PR'] = average_precision_score(y_train, y_score_train)
    d['train_precision'] = precision_score(y_train, y_pred_train)
    d['train_recall-sensitivity'] = recall_score(y_train, y_pred_train, pos_label=1)
    d['train_specificity'] = recall_score(y_train, y_pred_train, pos_label=0)
    d['train_F1'] = f1_score(y_train, y_pred_train)

    d['test_AUC-ROC'] = roc_auc_score(y_test, y_score_test)
    d['test_AUC-PR'] = average_precision_score(y_test, y_score_test)
    d['test_precision'] = precision_score(y_test, y_pred_test)
    d['test_recall-sensitivity'] = recall_score(y_test, y_pred_test, pos_label=1)
    d['test_specificity'] = recall_score(y_test, y_pred_test, pos_label=0)
    d['test_F1'] = f1_score(y_test, y_pred_test)
    
    return d

# get performance metrics in a long format DataFrame
def get_results_df(results_list: list):
    import pandas as pd
    df = pd.DataFrame(results_list).melt(id_vars='model')
    df = pd.concat([df, df['variable'].str.split('_', expand=True)], axis=1).drop('variable', axis=1)
    df = df.rename(columns={0: 'partition', 1: 'metric'})
    df = df.reindex(columns=['model', 'partition', 'metric', 'value'])
    return df
    
    