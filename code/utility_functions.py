
#################################################################################
##### function to calculate all performance metrics
#################################################################################
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
    d['train_F1'] = f1_score(y_train, y_pred_train)
    d['train_recall-sensitivity'] = recall_score(y_train, y_pred_train, pos_label=1)
    d['train_specificity'] = recall_score(y_train, y_pred_train, pos_label=0)
    d['train_precision'] = precision_score(y_train, y_pred_train)
    
    d['test_AUC-ROC'] = roc_auc_score(y_test, y_score_test)
    d['test_AUC-PR'] = average_precision_score(y_test, y_score_test)
    d['test_F1'] = f1_score(y_test, y_pred_test)
    d['test_recall-sensitivity'] = recall_score(y_test, y_pred_test, pos_label=1)
    d['test_specificity'] = recall_score(y_test, y_pred_test, pos_label=0)
    d['test_precision'] = precision_score(y_test, y_pred_test)
    
    return d


#################################################################################
##### get performance metrics in a long format DataFrame
#################################################################################
def get_results_df(results: list):
    '''
    Parameters
    ----------
    results : (list) A list of dictionaries

    Returns:
    df : (DataFrame) pandas DataFrame in long format
    '''
    import pandas as pd
    df = pd.DataFrame(results).melt(id_vars='model')
    df = pd.concat([df, df['variable'].str.split('_', expand=True)], axis=1).drop('variable', axis=1)
    df = df.rename(columns={0: 'partition', 1: 'metric'})
    df = df.reindex(columns=['model', 'partition', 'metric', 'value'])
    return df


#################################################################################
##### plot performance metrics for train and test sets
#################################################################################
def plot_performance_metrics(df):
    '''
    Parameters
    ----------
    df : (DataFrame) pandas data frame of model performance metrics

    Returns: (plot)
    '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    g = sns.catplot(x='model', y='value', hue='partition', col='metric', kind='bar', 
                    ci=None, height=5, aspect=1.1, col_wrap=3, data=df)

    # manually set x-tick labels for each subplot
    model_names = df['model'].unique()
    for ax in g.axes.flat:
        ax.set_xticklabels(model_names, rotation=45)
        ax.tick_params(labelbottom=True)
    
    g.set_titles("{col_name}")
    g.set_axis_labels("Model", "Value")
    g._legend.remove()

    # create a 'supra' axis that spans the entire grid for the legend and title
    plt.subplots_adjust(top=0.9)  # adjust the top margin to make space for the title and legend
    supra_ax = g.fig.add_subplot(111, frame_on=False)
    supra_ax.grid(False)
    supra_ax.tick_params(labelcolor="none", bottom=False, left=False)

    # add the title and adjusting the legend
    supra_ax.set_title('Model Performance by Metric and Data Partition', fontsize=16, pad=80)
    g.add_legend(title='Partition', loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=2)

    plt.subplots_adjust(bottom=0.1, left=0.06, hspace=0.2)
    plt.tight_layout()
    plt.show()
    
    
#################################################################################
##### compute ROC metrics and AUC, and plot ROC curves
#################################################################################
def plot_ROC_curves(models: dict, data: tuple):
    '''
    Parameters
    ----------
    models : (dict) model names (keys) and fitted sklearn estimators (items)
    data : (tuple) Contains train and test split for X and y

    Returns: (plot)
    '''
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier 
    from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, auc
    
    # unpack the data
    X_train, X_test, y_train, y_test = data

    # set up the subplots for training and test data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # plot for Training Data
    for name, model in models.items():
        # predict probabilities for the positive class
        if hasattr(model, "predict_proba"):
            y_scores_train = model.predict_proba(X_train)[:, 1]
        else:
            y_scores_train = model.decision_function(X_train)

        # compute ROC metrics and AUC for training data
        fpr_train, tpr_train, _ = roc_curve(y_train, y_scores_train)
        auc_train = roc_auc_score(y_train, y_scores_train)

        # plot using RocCurveDisplay on training data axis
        RocCurveDisplay(fpr=fpr_train, tpr=tpr_train).plot(ax=ax1, label=f'{name} (AUC = {auc_train:.2f})')

    # add the diagonal line for AUC = 0.5 on training data axis
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--', label='AUC = 0.5 (Random)')

    # plot for Test Data
    for name, model in models.items():
        # use already trained model to predict on test data
        if hasattr(model, "predict_proba"):
            y_scores_test = model.predict_proba(X_test)[:, 1]
        else:
            y_scores_test = model.decision_function(X_test)

        # compute ROC metrics and AUC for test data
        fpr_test, tpr_test, _ = roc_curve(y_test, y_scores_test)
        auc_test = roc_auc_score(y_test, y_scores_test)

        # plot using RocCurveDisplay on test data axis
        RocCurveDisplay(fpr=fpr_test, tpr=tpr_test).plot(ax=ax2, label=f'{name} (AUC = {auc_test:.2f})')

    # add the diagonal line for AUC = 0.5 on test data axis
    ax2.plot([0, 1], [0, 1], color='black', linestyle='--', label='AUC = 0.5 (Random)')

    # set y-axis limits for both subplots
    ax1.set_ylim([0, 1.05])
    ax2.set_ylim([0, 1.05])

    # add plot details for both subplots
    ax1.set_title('ROC Curves and AUC for Training Data')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    ax2.set_title('ROC Curves and AUC for Test Data')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    plt.show()
    
    
#################################################################################    
##### compute PR metrics and AUC, and plot PR curves
#################################################################################
def plot_PR_curves(models: dict, data: tuple):
    '''
    Parameters
    ----------
    models : (dict) model names (keys) and fitted sklearn estimators (items)
    data : (tuple) Contains train and test split for X and y

    Returns: (plot)
    '''
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.tree import DecisionTreeClassifier 
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier 
    from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay

    # unpack the data
    X_train, X_test, y_train, y_test = data
    
    # set up the subplots for training and test data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # plot for Training Data
    for name, model in models.items():
        # predict probabilities for the positive class
        if hasattr(model, "predict_proba"):
            y_scores_train = model.predict_proba(X_train)[:, 1]
        else:
            y_scores_train = model.decision_function(X_train)

        # compute Precision-Recall metrics and Average Precision (AP) for training data
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_scores_train)
        ap_train = average_precision_score(y_train, y_scores_train)

        #pPlot using PrecisionRecallDisplay on training data axis
        PrecisionRecallDisplay(precision=precision_train, recall=recall_train).plot(ax=ax1, label=f'{name} (AP = {ap_train:.2f})')

    # add the horizontal line for random (no-skill) classifier on training data axis
    prevalence_train = y_train.mean()
    ax1.hlines(prevalence_train, 0, 1, colors='black', linestyles='--', label=f'No Skill (AP = {prevalence_train:.2f})')

    # plot for Test Data
    for name, model in models.items():
        # use already trained model to predict on test data
        if hasattr(model, "predict_proba"):
            y_scores_test = model.predict_proba(X_test)[:, 1]
        else:
            y_scores_test = model.decision_function(X_test)

        # compute Precision-Recall metrics and Average Precision (AP) for test data
        precision_test, recall_test, _ = precision_recall_curve(y_test, y_scores_test)
        ap_test = average_precision_score(y_test, y_scores_test)

        # plot using PrecisionRecallDisplay on test data axis
        PrecisionRecallDisplay(precision=precision_test, recall=recall_test).plot(ax=ax2, label=f'{name} (AP = {ap_test:.2f})')

    # add the horizontal line for random (no-skill) classifier on test data axis
    prevalence_test = y_test.mean()
    ax2.hlines(prevalence_test, 0, 1, colors='black', linestyles='--', label=f'No Skill (AP = {prevalence_test:.2f})')

    # set y-axis limits for both subplots
    ax1.set_ylim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])

    # add plot details for both subplots
    ax1.set_title('Precision-Recall Curves with AP (average precision) for Training Data')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    ax2.set_title('Precision-Recall Curves with AP (average precision) for Test Data')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left')
    ax2.grid(True)

    plt.show()    
    

    