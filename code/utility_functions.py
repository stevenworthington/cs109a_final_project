
# Utility functions for CS109A final project

#################################################################################
##### Function to Process Max Glu Serum
#################################################################################
def process_max_glu_serum(df):

    df_copy = df.copy() 

    if 'max_glu_serum' not in df_copy.columns:
        print("Max glu serum is not a feature in the dataframe")
        return None

    df_copy['max_glu_serum_high'] = df_copy['max_glu_serum'].fillna(0).replace({'>300':1, 'Norm':0,'>200':0})
    
    df_copy['max_glu_serum'] = df_copy['max_glu_serum'].fillna('UNK') #not measured
    
    return df_copy
    

#################################################################################
##### Function to Process A1C Result
#################################################################################
def process_A1Cresult(df):

    df_copy = df.copy() 

    if 'A1Cresult' not in df_copy.columns:
        print("A1Cresult is not a feature in the dataframe")
        return None
    
    df_copy['a1c_result_high'] = df_copy['A1Cresult'].fillna(0).replace({'>8':1,'Norm':0,'>7':0})
    
    df_copy['A1Cresult'] = df_copy['A1Cresult'].fillna('UNK')

    return df_copy
    

#################################################################################
##### Function to Process Medical Specialty
#################################################################################
# appears to modify in place
def process_medical_specialty(df, take_top_num = 9):
    df_copy = df.copy()

    frequent_specialties = df_copy['medical_specialty'].value_counts().index[:take_top_num].values

    df_copy.loc[~df_encounters['medical_specialty'].isin(frequent_specialties), 'medical_specialty'] = 'Other'

    df_copy.loc[df_encounters['medical_specialty'].isin(['Orthopedics','Orthopedics-Reconstructive']), 
                  'medical_specialty'] = 'Orthopedics'
    
    return df_copy
    

#################################################################################
##### Function to Process Race
#################################################################################
def process_race(df):
    df_copy = df.copy()
    df_copy.race.fillna('UNK',inplace=True)
    
    return df_copy
    

#################################################################################
##### Function to Process Diagnosis Codes
#################################################################################
# this function needs to be fixed. We also need to check against the output Karim had in MS3. My output is slightly different (for the value counts for each category).
# need to fix the E thing in the dataset and the regex
# ignore this for now, look into it later
def process_diag_codes(df):
    df_copy = df.copy()

    group_to_code_patterns = dict()
    group_to_code_patterns['diabetes']=r'250\..*'
    group_to_code_patterns['circulatory']=r'(39[0-9])|(45[0-9])|(785)'
    group_to_code_patterns['respiratory']=r'(4[6-9][0-9])|(5[0-1][0-9])|(786)'
    group_to_code_patterns['digestive']=r'(5[2-7][0-9])|(787)'
    group_to_code_patterns['injury']=r'([8|9][0-9][0-9])|(E[8|9][0-9][0-9])'
    group_to_code_patterns['musculoskeletal']=r'(7[1-3][0-9])'
    group_to_code_patterns['genitournary']=r'(5[8-9][0-9])|(6[0-2][0-9])|(788)'
    group_to_code_patterns['neoplasms']=r'(1[4-9][0-9])|(2[0-9][0-9])|(78(0|1|4))|(79[0-9])|(2[4-7][0-9])|(6[8-9][0-9])|(70[0-9])|(782)'
    group_to_code_patterns['other']=r'(V[0-9]+)|([0-9]+)|other.*'

    for diag in ['diag_1','diag_2','diag_3']:
        df_copy[diag].fillna('UNK',inplace=True)
        for group,pattern in group_to_code_patterns.items():
            df_copy[diag]=df_copy[diag].str.replace(pattern,group,regex=True)

    return df_copy


#################################################################################
##### Function to Process Age
#################################################################################
# would be nice to better generalize this, but it may not be worth it
def process_age(df):
    df_copy = df.copy()

    df_copy['age'].replace({'[0-10)' :'[0-50)',
                            '[10-20)':'[0-50)',
                            '[20-30)':'[0-50)',
                            '[30-40)':'[0-50)',
                            '[40-50)':'[0-50)',
                            '[80-90)':'[80-100)',
                            '[90-100)':'[80-100)'},
                            inplace=True)
    
    return df_copy
    

#################################################################################
##### Function to Process Discharge Disposition ID
#################################################################################
# will only provide descriptive labels for the 3 most common categories
def process_discharge_disposition_id(df, keep_most_freq = 3):
    df_copy = df.copy()

    #remove patients transferred to hospice or those that died
    df_copy = df_copy[~df_encounters.discharge_disposition_id.isin([11,13,14,19,20,21])]

    #collapse unimportant ones

    most_freq = df_copy['discharge_disposition_id'].value_counts()[:keep_most_freq].index.values
    
    df_copy['discharge_disposition_id']=df_copy['discharge_disposition_id'].apply(lambda id: id if id in most_freq else 0)

    df_copy.discharge_disposition_id.replace({0:'Other',1:'Home',3:'SNF',6:'Home w/ Service'}, inplace=True)

    return df_copy
    

#################################################################################
##### Function to Process Admission Type ID
#################################################################################
def process_admission_type_id(df, keep_most_freq=3):

        df_copy = df.copy()

        most_freq = df_copy['admission_type_id'].value_counts()[:keep_most_freq].index.values

        df_copy['admission_type_id'] = df_copy['admission_type_id'].apply(lambda id: 0 if id >=4 else id)

        df_copy.admission_type_id.replace({0:'Other',1:'Emergency',2:'Urgent',3:'Elective'}, inplace=True)

        return df_copy
        

#################################################################################
##### Function to Process Admission Source ID
#################################################################################
def process_admission_source_ID(df):
    df_copy = df.copy()
    df_copy.loc[~df_encounters['admission_source_id'].isin([1,7]) , 'admission_source_id'] = 0
    df_copy.admission_source_id.replace({0:'Other',1:'Physician Referral',7:'Emergency Room'},
                                           inplace=True)
    
    return df_copy
    

#################################################################################
##### Function to Process Readmitted
#################################################################################
def process_readmitted(df):
    df_copy = df.copy()
    df_copy['readmitted'].replace({'<30':1,'NO':0,'>30':0},inplace=True)

    return df_copy
    

#################################################################################
##### Function to Process Diabetes Med and Change Features
#################################################################################
def process_diabetesMed_and_change(df):

    df_copy = df.copy()

    df_copy['diabetesMed'].replace({'Yes':1,'No':0},inplace=True)
    df_copy['change'].replace({'Ch':1,'No':0},inplace=True)

    return df_copy
    

#################################################################################
##### Function to Chain Together All Preprocessing Functions
#################################################################################
def preprocess_df(df, process_list):
    
    df_copy = df.copy()

    df_copy.drop(['weight', 'payer_code'], axis=1, inplace=True)

    for func in process_list:
        df_copy = func(df_copy)

    return df_copy
    

#################################################################################
##### Function to get previous encounters
#################################################################################
def get_previous_encounters(df):
    df_copy = df.copy()

    df_mult_encounter_patients = df_copy.groupby('patient_nbr').filter(lambda group: len(group)>1)
    df_mult_encounter_patients_previous = df_mult_encounter_patients.groupby('patient_nbr').apply(lambda group: group.iloc[:-1]).reset_index(drop=True)
    
    return df_mult_encounter_patients_previous
    

#################################################################################
##### Function to get aggregate previous encounters
#################################################################################
def aggregate_previous_encounters(df):
    df_copy = df.copy()

    df_patient_agg = df_copy.groupby('patient_nbr').agg({'encounter_id':'nunique',
                                  'time_in_hospital':['mean','min','max'],
                                  'num_lab_procedures':['mean','min','max'],
                                  'num_procedures': ['mean','min','max'],
                                  'num_medications':['mean','min','max'],
                                  'number_diagnoses':['mean','min','max'],
                                  'max_glu_serum':['nunique'],
                                  'max_glu_serum_high':['mean','sum','all','any'],
                                  'A1Cresult':'nunique',
                                  'a1c_result_high':['mean','sum','all','any'],
                                  'change':['mean','sum','all','any'],
                                  'diabetesMed': ['mean','sum','all','any'],
                                  'readmitted': ['mean','sum','all','any']})
    

    aggr_cols = ['num_encounters',
                'avg_time_in_hospital','min_time_in_hospital','max_time_in_hospital',
                'avg_num_lab_procedures','min_num_lab_procedures','max_num_lab_procedures',
                'avg_num_procedures','min_num_procedures','max_num_procedures',
                'avg_num_medications','min_num_medications','max_num_medications',
                'mean_diagnoses','min_diagnoses','max_diagnoses',
                'unique_glu_measurements',
                'avg_times_glu_high','num_times_glu_high','glu_always_high','glu_ever_high',
                'unique_a1c_results',
                'avg_times_a1c_high','num_times_a1c_high','a1c_always_high','a1c_ever_high',
                'avg_times_med_changed','num_times_med_changed','med_always_changed','med_ever_changed',
                'avg_times_diabetic_med_prescribed','num_times_diabetic_med_prescribed',
                'diabetic_med_always_prescribed', 'diabetic_med_ever_prescribed',
                'avg_times_readmitted','num_times_readmitted','always_readmitted','ever_readmitted']
    
    df_patient_agg.columns = aggr_cols

    for col in df_patient_agg.select_dtypes(bool):
        df_patient_agg[col] = df_patient_agg[col].astype(int)

    return df_patient_agg
    

#################################################################################
##### Function to get last encounters
#################################################################################
def get_last_encounter(df):
    df_copy = df.copy()

    return df_copy.groupby('patient_nbr').last()
    

#################################################################################
##### Function to aggregate encounters
#################################################################################
def aggregate_encounters(df):

    if 'patient_nbr' not in df.columns:
        print("Patient_nbr is not in the dataframe")
        return None

    if max(df['patient_nbr'].value_counts()) < 2:
        print('There are no patients to aggregate')
        return 0
    
    df_copy = df.copy()

    last_encounters = get_last_encounter(df_copy)

    previous_encounters = get_previous_encounters(df_copy)
    agg_previous_encounters = aggregate_previous_encounters(previous_encounters)

    df_patient = last_encounters.merge(agg_previous_encounters, on='patient_nbr', how='left').fillna(0)

    #no use for encounter_id or the two other temp columns anymore
    df_patient.drop(['encounter_id','a1c_result_high','max_glu_serum_high'],axis=1,inplace=True)

    return df_patient
    
    
#################################################################################
##### Function to calculate all performance metrics
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
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    X_train, X_test, y_train, y_test = data
    d = {}
    y_score_train = model.predict_proba(X_train)[:, 1] # prob
    y_pred_train = model.predict(X_train) # class
    y_score_test = model.predict_proba(X_test)[:, 1] # prob
    y_pred_test = model.predict(X_test) # class
    
    d['model'] = classifier_name
    d['Train_Readmitted-Rate-Observed'] = np.mean(y_train)
    d['Train_Readmitted-Rate-Predicted'] = np.mean(y_score_train)
    d['Train_Naive-Accuracy'] = 1-np.mean(y_train)
    d['Train_Accuracy'] = accuracy_score(y_train, y_pred_train)
    d['Train_AUC-ROC'] = roc_auc_score(y_train, y_score_train)
    d['Train_AUC-PR'] = average_precision_score(y_train, y_score_train)
    d['Train_F1-Score'] = f1_score(y_train, y_pred_train)
    d['Train_Recall-Sensitivity'] = recall_score(y_train, y_pred_train, pos_label=1)
    d['Train_Specificity'] = recall_score(y_train, y_pred_train, pos_label=0)
    d['Train_Precision'] = precision_score(y_train, y_pred_train)
    
    d['Test_Readmitted-Rate-Observed'] = np.mean(y_test)
    d['Test_Readmitted-Rate-Predicted'] = np.mean(y_score_test)
    d['Test_Naive-Accuracy'] = 1-np.mean(y_test)
    d['Test_Accuracy'] = accuracy_score(y_test, y_pred_test)
    d['Test_AUC-ROC'] = roc_auc_score(y_test, y_score_test)
    d['Test_AUC-PR'] = average_precision_score(y_test, y_score_test)
    d['Test_F1-Score'] = f1_score(y_test, y_pred_test)
    d['Test_Recall-Sensitivity'] = recall_score(y_test, y_pred_test, pos_label=1)
    d['Test_Specificity'] = recall_score(y_test, y_pred_test, pos_label=0)
    d['Test_Precision'] = precision_score(y_test, y_pred_test)
    
    return d


#################################################################################
##### Function to get performance metrics in a long format DataFrame
#################################################################################
def get_results_df(results: list, model: str = None):
    '''
    Parameters
    ----------
    results : (list) A list of dictionaries
    model : (str, optional) Model name to filter the DataFrame. If None, no filtering is applied.

    Returns:
    df : (DataFrame) pandas DataFrame in long format
    '''
    import pandas as pd
    
    df = pd.DataFrame(results).melt(id_vars='model')
    df = pd.concat([df, df['variable'].str.split('_', expand=True)], axis=1).drop('variable', axis=1)
    df = df.rename(columns={0: 'partition', 1: 'metric'})
    df = df.reindex(columns=['model', 'partition', 'metric', 'value'])
    df['value'] = df['value'] * 100
    
    # optionally filter by model
    if model is not None:
        df = df[df['model'] == model]
    
    # turn off scientific notation
    pd.set_option('display.float_format', '{:.1f}'.format)
    
    return df


#################################################################################
##### Function to plot performance metrics for train and test sets
#################################################################################
def plot_performance_metrics(df):
    '''
    Parameters
    ----------
    df : (DataFrame) pandas data frame of model performance metrics

    Returns: (plot)
    '''
    import pandas as pd
    from plotnine import ggplot, aes, geom_bar, geom_text, facet_wrap, labs, theme
    from plotnine import element_text, scale_fill_manual, scale_y_continuous, position_dodge
    
    # set the desired order of the models in the DataFrame
    model_order = ['Base Model', 'Logistic Regularized', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    
    # set the desired order of the metrics in the DataFrame
    metric_order = ['Readmitted-Rate-Observed', 'Readmitted-Rate-Predicted', 'Naive-Accuracy', 'Accuracy',
                             'AUC-ROC', 'AUC-PR', 'F1-Score', 'Recall-Sensitivity', 'Specificity', 'Precision']
    df['metric'] = pd.Categorical(df['metric'], categories=metric_order, ordered=True)

    # create a formatted label column
    df['formatted_value'] = df['value'].apply(lambda x: f'{x:.1f}%')
    
    plot = (ggplot(df, aes(x='model', y='value', fill='partition')) 
            + geom_bar(stat='identity', position='dodge') 
            + geom_text(aes(label='formatted_value'), position=position_dodge(width=0.9), va='bottom', size=8)
            + scale_fill_manual(values=['#4e79a7', '#f28e2b'])
            + scale_y_continuous(labels=lambda l: [f'{v:.0f}%' for v in l], limits=[0, 105])
            + facet_wrap('~metric', scales='free_x', ncol=2) 
            + labs(title='Model Performance by Metric and Data Partition', 
                   x='Model', 
                   y='Value', 
                   fill='Data Partition') 
            + theme(axis_text_x=element_text(rotation=45, hjust=1),
                    plot_title=element_text(size=14, ha='center'),
                    axis_title=element_text(size=12),
                    legend_title=element_text(size=10),
                    legend_text=element_text(size=10),
                    legend_position='top',
                    figure_size=(12, 20)
                   )
           )

    print(plot)
        
    
#################################################################################
##### Function to compute ROC metrics and AUC, and plot ROC curves
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
##### Function to compute PR metrics and AUC, and plot PR curves
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
    
