import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_score(y_test, y_pred, target):
    from sklearn.metrics import r2_score, mean_squared_error

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    accuracy, recall, _ = binning(y_test, y_pred, target)

    return np.round(rmse, 4), np.round(r2, 4), np.round(accuracy, 4), np.round(recall, 4)

def binning(y_test, y_pred, target):
    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
    
    if target in ['Synedra', 'ToxicCyano']: 
        cri = [-np.inf, 100, 300, 1000, np.inf]
        cri2 = [-np.inf, 0.02, np.inf]
    if target in ['2MIB', 'Geosmin']: 
        cri = [-np.inf, 0.01, 0.02, 0.05, np.inf]
        cri2 = [-np.inf, 0.02, np.inf]
        
    if "NTU" in target:
        cri = [-np.inf, 10, 30, 100, np.inf]  # Criteria for binning
        cri2 = [-np.inf, 30, np.inf]  # Criteria for binning

    y_true_binned = pd.cut(y_test, bins=cri, labels=[0,1,2,3])
    y_pred_binned = pd.cut(y_pred, bins=cri, labels=[0,1,2,3])
    y_true_binned2 = pd.cut(y_test, bins=cri2, labels=[0,1])
    y_pred_binned2 = pd.cut(y_pred, bins=cri2, labels=[0,1])

    # Calculate confusion matrix
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=[0,1,2,3])
    if cm.shape != (4, 4):
        new_cm = np.zeros((4, 4), dtype=int)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                new_cm[i, j] = cm[i, j]
        cm = new_cm
    
    return accuracy_score(y_true_binned, y_pred_binned), recall_score(y_true_binned2, y_pred_binned2), cm

## Function for save R2, RMSE, accuracy and recall
def performance_tab(y_tr, y_pred_tr,
                    y_vl, y_pred_vl,
                    y_te, y_pred_te,
                    fct,
                    model_nam,
                    info = None, ###
                    target = "NTU"):
    
    perf_save_temp = pd.DataFrame(columns = ['rmse', 'r2', 'accuracy', 'recall', 'fct', 'site', 'model', 'info'])
    
    perf_save_temp.loc[0, ['rmse', 'r2', 'accuracy', 'recall']] = calculate_score(y_tr.ravel(), y_pred_tr.ravel(), target)

    perf_save_temp.loc[1, ['rmse', 'r2', 'accuracy', 'recall']] = calculate_score(y_vl.ravel(), y_pred_vl.ravel(), target)

    perf_save_temp.loc[2, ['rmse', 'r2', 'accuracy', 'recall']] = calculate_score(y_te.ravel(), y_pred_te.ravel(), target)

    perf_save_temp.loc[0, 'data'] = 'tr'
    perf_save_temp.loc[1, 'data'] = 'vl'
    perf_save_temp.loc[2, 'data'] = 'te'

    perf_save_temp.loc[:, "fct"] = fct
    perf_save_temp.loc[:, "site"] = target
    perf_save_temp.loc[:, "model"] = model_nam
    perf_save_temp.loc[:, "info"] = info
    
    return perf_save_temp

# Code for saving confusion matrix
def confusion_tab(y_tr, y_pred_tr,
                    y_vl, y_pred_vl,
                    y_te, y_pred_te,
                    fct,
                    model_nam,
                    info = None, ###
                    target = "NTU"):
    
    columns = [''] + [f'plot1_{i}_{j}' for i in range(4) for j in range(4)] + ['accuracy'] + ['recall'] + ['data']
    conf_save_temp = pd.DataFrame(columns=columns)
    
    accuracy, recall, conf = binning(y_tr.ravel(), y_pred_tr.ravel(), target)
    add_metrics_to_df(conf_save_temp, 0, pd.DataFrame(conf), accuracy, recall, 'tr')

    accuracy, recall, conf = binning(y_vl.ravel(), y_pred_vl.ravel(), target)
    add_metrics_to_df(conf_save_temp, 1, pd.DataFrame(conf), accuracy, recall, 'vl')

    accuracy, recall, conf = binning(y_te.ravel(), y_pred_te.ravel(), target)
    add_metrics_to_df(conf_save_temp, 2, pd.DataFrame(conf), accuracy, recall, 'te')

    conf_save_temp.loc[:, "fct"] = fct
    conf_save_temp.loc[:, "site"] = target
    conf_save_temp.loc[:, "model"] = model_nam
    conf_save_temp.loc[:, "Info"] = info

    return conf_save_temp

# Function for save R2, RMSE, accuracy and recall
def performance_tab_week(y_tr, y_pred_tr,
                    y_te, y_pred_te,
                    fct, site,
                    model_nam,
                    info = None, ###
                    target = 'NTU'):
    
    perf_save_temp = pd.DataFrame(columns = ['rmse', 'r2', 'accuracy', 'recall', 'fct', 'site', 'model', 'info'])
    
    perf_save_temp.loc[0, ['rmse', 'r2', 'accuracy', 'recall']] = calculate_score(y_tr.ravel(), y_pred_tr.ravel(), target)

    perf_save_temp.loc[2, ['rmse', 'r2', 'accuracy', 'recall']] = calculate_score(y_te.ravel(), y_pred_te.ravel(), target)

    perf_save_temp.loc[0, 'data'] = 'tr'
    perf_save_temp.loc[2, 'data'] = 'te'

    perf_save_temp.loc[:, "fct"] = fct
    perf_save_temp.loc[:, "site"] = f'{site}.{target}'
    perf_save_temp.loc[:, "model"] = model_nam
    perf_save_temp.loc[:, "info"] = info
    
    return perf_save_temp

# Code for saving confusion matrix
def confusion_tab_week(y_tr, y_pred_tr,
                    y_te, y_pred_te,
                    fct, site,
                    model_nam,
                    info = None, ###
                    target = 'NTU'):
    
    columns = [''] + [f'plot1_{i}_{j}' for i in range(4) for j in range(4)] + ['accuracy'] + ['recall'] + ['data']
    conf_save_temp = pd.DataFrame(columns=columns)
    
    accuracy, recall, conf = binning(y_tr.ravel(), y_pred_tr.ravel(), target)
    add_metrics_to_df(conf_save_temp, 0, pd.DataFrame(conf), accuracy, recall, 'tr')

    accuracy, recall, conf = binning(y_te.ravel(), y_pred_te.ravel(), target)
    add_metrics_to_df(conf_save_temp, 2, pd.DataFrame(conf), accuracy, recall, 'te')

    conf_save_temp.loc[:, "fct"] = fct
    conf_save_temp.loc[:, "site"] = f'{site}.{target}'
    conf_save_temp.loc[:, "model"] = model_nam
    conf_save_temp.loc[:, "Info"] = info

    return conf_save_temp

def add_metrics_to_df(df, row_index, plot1, accuracy, recall, info):
    for i in range(4):
        for j in range(4):
            df.loc[row_index, f'plot1_{i}_{j}'] = plot1.iloc[i, j]
    df.loc[row_index, 'accuracy'] = accuracy
    df.loc[row_index, 'recall'] = recall
    df.loc[row_index, 'data'] = info

def confusion_matrix_file(df): ### TODO modifying code for weekly models
    n = len(df)

    new_df = pd.DataFrame()
    cols = ['10 to 30', '30 to 100', '100 이상', 'Accuracy', 'Recall', '']

    for i in range(n):
        temp = df.iloc[i:i + 1].copy()
        temp.iloc[0, 0] = '0 to 10'
        new_df = pd.concat([new_df, temp], ignore_index=True)

        for j, value in enumerate(cols):
            new_row = [''] * len(df.columns)
            new_row[0] = value
            new_df = pd.concat([new_df, pd.DataFrame([new_row], columns=df.columns)], ignore_index=True)
        
        new_df.iloc[i * (len(cols) + 1), 5:10] = df.iloc[i, 19:24].values
        new_df.iloc[i * (len(cols) + 1) + 1, 1:5] = df.iloc[i, 5:9].values
        new_df.iloc[i * (len(cols) + 1) + 2, 1:5] = df.iloc[i, 9:13].values
        new_df.iloc[i * (len(cols) + 1) + 3, 1:5] = df.iloc[i, 13:17].values
        new_df.iloc[i * (len(cols) + 1) + 4, 1] = df.iloc[i, 17]
        new_df.iloc[i * (len(cols) + 1) + 5, 1] = df.iloc[i, 18]
        new_df.drop(new_df.columns[10:24], axis=1, inplace=True)
        
    new_df.columns = ['','0 to 10', '10 to 30','30 to 100','100 이상', 'data', 'fct', 'site', 'model', 'Info']
    
    return new_df 
