import numpy as np
import pandas as pd

def move_data(overwrite = False):
    import shutil
    import os
    files = [f for f in os.listdir('./') if f.endswith('.xlsx')]

    for f in files:
        try:
            if overwrite == True:
                shutil.move(os.path.join(f'./{f}'), os.path.join(f'./Data/{f}'))
            else:
                if not os.path.isfile(os.path.join(f'./Data/{f}')):
                    shutil.move(os.path.join(f'./{f}'), os.path.join(f'./Data/{f}'))
        except:
            pass

def move_past_data(prev_date, overwrite=False):
    import shutil
    import os
    files = [f for f in os.listdir('./Results/') if os.path.isfile(os.path.join('./Results/', f))]

    if not os.path.exists('./Results/Legacy'):
        os.makedirs('./Results/Legacy')
    for f in files:
        try:
            src = os.path.join('./Results', f)
            if prev_date in f:
                dest = os.path.join('./Results/Legacy', f)
            else:
                dest = os.path.join('./Results', f)
            
            if overwrite or not os.path.isfile(dest):
                shutil.move(src, dest)
                print(f"Moved: {f} to {str(dest)}")
        except Exception as e:
            print(f"Failed to move {f}: {e}")

def move_past_data_mod(prev_date, overwrite=False):
    import shutil
    import os
    files = [f for f in os.listdir('./Results/Test/') if os.path.isfile(os.path.join('./Results/Test/', f))]

    if not os.path.exists('./Results/Test/Legacy'):
        os.makedirs('./Results/Test/Legacy')
    for f in files:
        try:
            src = os.path.join('./Results/Test', f)
            if prev_date in f:
                dest = os.path.join('./Results/Test/Legacy', f)
            else:
                dest = os.path.join('./Results/Test', f)
            
            if overwrite or not os.path.isfile(dest):
                shutil.move(src, dest)
                print(f"Moved: {f} to {str(dest)}")
        except Exception as e:
            print(f"Failed to move {f}: {e}")


def data_cleaning(data,
                  sites,
                  target = "NTU",
                  threshold_weight = [1.4, 1.25]):
    
    
    # Instruction (WIP)
    # INPUT
    # data : Data for performing preprocessing
    # sites : Included sites (list) e.g., ['PD1', 'PD2', ...]
    # target : Preprocessing target
    # threshold_weights : Weight for threshold (tuple) - used to modify sensitivity of cleasing algorithm (step 3, 4) e.g., (1.4, 1.25)
    # TODO --> MODULARIZE
    # TODO --> Improve computational efficiency
    
    # OUTPUT
    # data : Preprocessed data
    # am_info : Indexing for removed values
    
    num_iter = len(threshold_weight)
    ori_index = data.index ### To reverse original index
    
    data.index = range(len(data))
    am_info = pd.DataFrame(index = range(len(data)))
    
    for f in [f for f in data.columns if target in f]:
        print(f)
        for st in sites:
            if st in f:
                ##### Step 1 - Filtering values using detection limits (0~2000)
                am_info.loc[data[f] <= 0, "%s_S1_0"%(st)] = 1
                data.loc[data[f] <= 0, f] = np.nan 
                
                am_info.loc[data[f] >= 2000, "%s_S1_2000"%(st)] = 1
                data.loc[data[f] >= 2000, f] = np.nan
                
                print("Preprocessing for %s.%s, Step 1 complete"%(st, target))
                
                ###### Iteration for Step 2 and step 3
                for iter in range(num_iter):
                    ###### Step 2 - Remove fluctuated values
                    sub_dat = data[f].copy()
                    selected = []
                    window = 2

                    for i in range(window, len(sub_dat)-window):
                        if sub_dat[i-window : i+window+1].notna().all():
                            threshold = np.nanmean(sub_dat[[i-window, i+window]]) * threshold_weight[iter]
                            gradient_prev = abs(sub_dat[i] - sub_dat[i - window])
                            gradient_next = abs(sub_dat[i + window] - sub_dat[i])

                            # if ~np.isnan(gradient_next) or ~np.isnan(gradient_prev):
                            if gradient_prev > threshold and\
                                gradient_next > threshold:  ### and or check
                                selected.append(i)
                                
                        if ~np.isnan(sub_dat[i-window]) or ~np.isnan(sub_dat[i+window]):
                            threshold = np.nanmean(sub_dat[[i-window, i+window]]) * threshold_weight[iter]
                            
                            if sub_dat[i] > threshold:
                                selected.append(i)
                                globals()['selected'] = selected
                                
                    am_info.loc[selected, "%s_S2_%s"%(st, iter)] = 1
                    data.loc[selected, f] = np.nan 
                    print("Preprocessing for %s.%s, Step 2 complete, iteration %s"%(st, target, iter))
                    
                    ###### Step 3 - Remove fluctuated values - Sharpely decrease
                    sub_dat = data[f].copy()
                    selected = []
                    for i in range(window*2, len(sub_dat)-window*2):
                        # if sub_dat[i-window : i+window+1].notna().all():
                        if ~np.isnan(sub_dat[i-window*2]) or ~np.isnan(sub_dat[i]):
                            threshold = np.nanmean(sub_dat[(i-window*2):i + 1]) * 1.5 #### check later
                            # gradient_prev = abs(sub_dat[i] - sub_dat[i - window])
                            gradient_next = sub_dat[i-1] - sub_dat[i]

                            if gradient_next > 0 and abs(gradient_next) > threshold:  ### and or check
                                selected.append(list(range(i, min(i+24, len(sub_dat))))) #### Remove next 24 hr data
                                
                    selected = sum(selected, [])
                    selected = list(set(selected))
                    
                    am_info.loc[selected, "%s_S3_%s"%(st, iter)] = 1
                    data.loc[selected, f] = np.nan
                    
                    print("Preprocessing for %s.%s, Step 3 complete, iteration %s"%(st, target, iter))
                
                ##### Step 4 - Remove lagged values
                sub_dat = data[f].copy()
                
                ######### Filter 1
                selected_1 = []
                start_ = None
                count = 1
                threshold = 6

                for i in range(1, len(sub_dat)):
                    if np.round(sub_dat[i], 3) == np.round(sub_dat[i - 1], 3):
                        if start_ is None:
                            start_ = i - 1
                        count += 1
                    else:
                        if count > threshold:
                            selected_1.append(list((range(start_-3, i+2)))) # start_, i-1
                        start_ = None
                        count = 1
                selected_1 = sum(selected_1, [])

                ######### Filter 2
                selected_2 = []
                start_ = None
                count = 1
                threshold = 70
                            
                for i in range(1, len(sub_dat)):
                    if round(sub_dat[i], 1) == round(sub_dat[i - 1], 1):
                        if start_ is None:
                            start_ = i - 1
                        count += 1
                    else:
                        if count > threshold:
                            selected_2.append(list(range(start_-3, i+2)))
                        start_ = None
                        count = 1
                selected_2 = sum(selected_2, [])

                # Combine and remove duplicates
                selected = list(set(selected_1 + selected_2))
                selected.sort()  # Sort the list to maintain order
                
                am_info.loc[selected, "%s_S4"%(st)] = 1
                data.loc[selected, f] = np.nan 
                
                print("Preprocessing for %s.%s, Step 4 complete"%(st, target))
                
                sub_dat = data[f].copy()
                for i in range(1, len(sub_dat)-1):
                    if np.isnan(sub_dat[i-1]) and np.isnan(sub_dat[i+1]) and ~np.isnan(sub_dat[i]):
                        data.loc[i, f] = np.nan
                        am_info.loc[i, "%s_S5"%(st)] = 1
                print("Preprocessing for %s.%s, Step 5 complete"%(st, target))
    
    data.index = ori_index
    am_info.index = ori_index
            
    return data, am_info

def imputation_KF(data,
                  method = "UKF",
                  fitting = True):
    
    import numpy.ma as ma
    from tqdm import tqdm
    from pykalman import UnscentedKalmanFilter, KalmanFilter

    ### Method : UKF, FK, Interpolate
    data_imput = data.copy()

    if method == "UKF":
        #### Perform imputation
        pbar = tqdm(range(data_imput.shape[1]), ncols = 100, ascii = " =", leave = True)
        
        for i in pbar:
            pbar.set_description(f'Imputation for {data_imput.columns[i]}')
            
            X_masked = pd.isna(data_imput.iloc[:, i])
            X = data_imput.iloc[:, i]
            X = ma.array(X)
            X.mask = X_masked

            ukf = UnscentedKalmanFilter(initial_state_mean=np.nanmean(data),
                                        transition_covariance= 0.1,
                                        transition_functions=lambda state, noise: state + noise,
                                        observation_functions=lambda state, noise: state + noise
                                        )
            
            (smoothed_state_means, _) = ukf.smooth(X)
                
            data_imput.iloc[:, i] = np.where(pd.isna(data_imput.iloc[:, i]) == True, smoothed_state_means.reshape(-1), data_imput.iloc[:, i])
        pbar.close()
        
    if method == "KF":
        #### Perform imputation
        pbar = tqdm(range(data_imput.shape[1]), ncols = 100, ascii = " =", leave = True)
        
        for i in pbar:
            pbar.set_description(f'Imputation for {data_imput.columns[i]}')
            X_masked = pd.isna(data_imput.iloc[:, i])
            X = data_imput.iloc[:, i]
            X = ma.array(X)
            X.mask = X_masked
            
            if fitting == True:
                kf = KalmanFilter(initial_state_mean=np.nanmean(data_imput.iloc[:, i]), n_dim_obs=1)
                
                (smoothed_state_means, _) = kf.em(X, n_iter=5).smooth(X)
                
                (smoothed_state_means, _) = kf.em(X, n_iter=5).smooth(X)
            
            data_imput.iloc[:, i] = np.where(pd.isna(data_imput.iloc[:, i]) == True, smoothed_state_means.reshape(-1), data_imput.iloc[:, i])
        pbar.close()
        
    if method == "Interpolate": #### Method from 02_model(SV).ipynb
        #### Perform imputation
        data_imput = data.interpolate()
    
    if method == "KNN":
        from sklearn.impute import KNNImputer ### preprocessing -> imputation_KF - 여기 안에 ㄱㄱ

    data_imput[data_imput < 0] = 0
    
    return data_imput

def filtering_longterm_missing(df, threshold):
    df = pd.DataFrame(df)
    df['missing'] = df.isna().astype(int)
    df['group'] = (df['missing'] != df['missing'].shift()).cumsum()
    
    missing_groups = df[df['missing'] == 1].groupby('group').size()
    
    longterm_missing_groups = missing_groups[missing_groups >= threshold].index
    
    start_ = []
    end_ = []
    
    for group in longterm_missing_groups:
        missing_indices = df[df['group'] == group].index
        start_.append(missing_indices[0])
        end_.append(missing_indices[-1])
    
    return start_, end_
