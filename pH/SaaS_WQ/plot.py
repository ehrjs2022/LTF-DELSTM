import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import timedelta

def matrix(df, input_vars = None,
           width_ratios=(15, 1), 
           fontsize=16, labels=None, 
           freq=None, ax=None, fig_size = None):
    
    import shap
    from matplotlib import cm
    from matplotlib import gridspec
    
    input_vars = input_vars
    df = df.T

    height = df.shape[0]
    width = df.shape[1]
    if fig_size is not None:
        figsize = fig_size
    else:    
        figsize=(25, 15)
    
    # z is the color-mask array, g is a NxNx3 matrix. Apply the z color-mask to set the RGB of each pixel.
    z = df.notnull().values
    g = np.zeros((height, width, 4), dtype=np.float32)

    ### adding color bar
    ### set color scheme like heatmap
    for att in input_vars:
        cvals = df.loc[att, :].copy()
        cvals[cvals <= 0] = 0
        
        vmin = np.nanpercentile(cvals, 5) # set color from 5th percentile to 
        vmax = np.nanpercentile(cvals, 95) # 95th percentile

        cvals[cvals > vmax] = vmax
        cvals[cvals < vmin] = vmin

        mm = cm.ScalarMappable(cmap = shap.plots.colors.red_blue) # define colormap
        mm.set_clim(vmin, vmax) # set upper and lower limit value

        g[input_vars.index(att), :, :] = mm.to_rgba(cvals)
        
        if "NTU" in att:
            # print(att)
            g[input_vars.index(att), df.loc[att, :] <= 0, :] = [0.5, 0.5, 0.5, 0.8]
        
        del vmax, vmin, cvals
        
    ### set white color for None value
    g[z < 0.5] = [0.9, 0.9, 0.9, 0.9] # 
    # g[df <= 0] =  ## Need to shift
    
    if ax is None:
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1)
        ax0 = plt.subplot(gs[0])
        
    ## Create the nullity plot.
    ax0.imshow(g, interpolation='none')

    # Remove extraneous default visual elements.
    ax0.set_aspect('auto')
    ax0.grid(visible=False)
    ax0.xaxis.tick_bottom()

    if labels or (labels is None and len(df.index) <= 50):
        ha = 'right'
        ax0.set_yticks(range(0, len(df.index)))
        ax0.set_yticklabels(df.index, rotation=0, ha=ha, fontsize=12)
    else:
        ax0.set_xticks([])

    if freq:
        ts_list = []

        if type(df.columns) == pd.PeriodIndex:
            ts_array = pd.date_range(df.columns.to_timestamp().date[0],
                                     df.columns.to_timestamp().date[-1],
                                     freq='6MS').values
            ts_array = pd.to_datetime(ts_array)
            ts_ticks = pd.date_range(df.columns.to_timestamp().date[0],
                                     df.columns.to_timestamp().date[-1],
                                     freq='6MS').map(lambda t:
                                                    t.strftime('%d %b. %Y'))

        elif type(df.columns) == pd.DatetimeIndex:
            ts_array = pd.date_range(df.columns[0], df.columns[-1],
                                     freq='6MS').values
            ts_array = pd.to_datetime(ts_array)
            ts_ticks = pd.date_range(df.columns[0], df.columns[-1],
                                     freq='6MS').map(lambda t:
                                                    t.strftime('%d %b. %Y'))
        else:
            raise KeyError('Dataframe index must be PeriodIndex or DatetimeIndex.')
        try:
            for value in ts_array:
                ts_list.append(df.columns.get_loc(value))
        except KeyError:
            raise KeyError('Could not divide time index into desired frequency.')

        ts_list.append(width)
        ax0.set_xticks(ts_list)

        ts_ticks = ts_ticks.append(pd.Index([df.columns.date[-1].strftime('%d %b. %Y')])) # for ending date
        ax0.set_xticklabels(ts_ticks, fontsize=int(fontsize / 16 * 20), rotation=45, ha='right', rotation_mode='anchor')

    else:
        ax0.set_xticks([0, df.shape[0] - 1])
        ax0.set_xticklabels([1, df.shape[0]], fontsize=int(fontsize / 16 * 20), rotation=0)

    # Create the inter-column vertical grid.
    in_between_point = [x + 0.5 for x in range(0, height - 1)]
    for in_between_point in in_between_point:
        ax0.axhline(in_between_point, linestyle='-', color='white', linewidth = 2.5)
        

def plot_AD(data_raw,
            am_info,
            data_filtered,
            target,
            folder_nam = "Preprocessing",
            start_date = "2010-01-01",
            end_date = "2024-12-31",
            color_measure = '#B3AAAA',
            color_anomaly = "salmon",
            figsize = (12, 5),
            show = True,
            save = False):
    
    data_raw.index = pd.to_datetime(data_raw.index)
    am_info.index = pd.to_datetime(am_info.index)
    data_filtered.index = pd.to_datetime(data_filtered.index)
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import os
    
    p_dat = data_raw.loc[(data_raw.index >= start_date) & (data_raw.index < end_date), target]
    p_dat_2 = data_filtered.loc[(data_filtered.index >= start_date) & (data_filtered.index < end_date), target]
    
    p_am = am_info.loc[(am_info.index >= start_date) & (am_info.index < end_date), am_info.columns[am_info.columns.str.contains(target.split(".")[0])]]
    p_am = p_am.sum(axis = 1)
    p_am[p_am > 1] = 1
    
    #### Plotting function
    plt.figure(figsize= figsize)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    plt.xticks(rotation=45)
    ax.set_xlim(p_dat.index[0], p_dat.index[-1])

    plt.plot(p_dat.index, p_dat, marker='o', linestyle='-', color=color_measure, 
                markerfacecolor=color_measure, markeredgecolor=color_measure, markersize=2.5)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=45)
    plt.title(f"{target} ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}), detection result")

    ax.plot(p_dat[p_am == 1], linewidth=0, marker='o', markersize=2.8, color=color_anomaly)
    
    ymin, ymax = ax.get_ylim()

    if save == True:
        folder_nam = "Preprocessing"  # Modify this to your desired folder name
        
        if not os.path.exists('./Results/%s'%(folder_nam)):
            os.mkdir('./Results/%s'%(folder_nam))
            
        plt.savefig(f'./Results/{folder_nam}/%s_%s_detection.png'%(target, start_date),
                dpi = 256, transparent = True)

    if show == True:
        plt.show()
    else:
        plt.close()
        
    plt.figure(figsize= figsize)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
    plt.xticks(rotation=45)
    ax.set_xlim(p_dat_2.index[0], p_dat_2.index[-1])

    plt.plot(p_dat_2.index, p_dat_2, marker='o', linestyle='-', color=color_measure, 
                markerfacecolor=color_measure, markeredgecolor=color_measure, markersize=2.5)
    plt.grid(alpha=0.2)
    plt.xticks(rotation=45)
    plt.title(f"{target} ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}), data cleaning result")
    plt.ylim([ymin, ymax])

    if save == True:
        folder_nam = "Preprocessing"  # Modify this to your desired folder name
        
        if not os.path.exists('./Results/%s'%(folder_nam)):
            os.mkdir('./Results/%s'%(folder_nam))
            
        plt.savefig(f'./Results/{folder_nam}/{target}_{start_date}_{end_date}_removed.png', 
                dpi = 256, transparent = True)

    if show == True:
        plt.show()
    else:
        plt.close()


############## Plots for model evaluation        
### Plotting Function from model(SV).ipynb
def plot_ts(y_test, y_pred,
            index, ### plotting period,
            show=True,
            save=False,
            target=None,
            model=None,
            fct_h=None,
            seq_len=None):
    
    y_test = y_test.ravel()
    y_pred = y_pred.ravel()
    
    y_pred[y_pred < 0] = 0
    
    plt.figure(figsize=(10, 4))
    
    # Create a DataFrame to identify continuous segments
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=index)
    
    # Find gaps in the index
    gaps = df.index.to_series().diff().dt.days.ne(1).cumsum()
    
    # Plot each continuous segment separately
    for _, segment in df.groupby(gaps):
        plt.plot(segment.index, segment['y_test'], c='black', linewidth=1)
        plt.plot(segment.index, segment['y_pred'], c='red', linewidth=1)
    
    # Optional: scatter plot to highlight points 
    plt.axhline(y=10, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=30, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    plt.xlim(index[0], index[-1])
    plt.ylim(0 - y_test.max() * 0.05, max(y_test.max() * 1.05, y_pred.max() * 1.05))
    plt.yticks(size=13)
    plt.ylabel('Predicted Turbidity (NTU)', fontsize=13)
    plt.tight_layout()
    
    if save:
        folder_nam = "Model_evaluation"  # Modify this to your desired folder name
        
        import os
        
        if not os.path.exists(f'./Results/{folder_nam}'):
            os.mkdir(f'./Results/{folder_nam}')
        
        plt.savefig(f'./Results/{folder_nam}/{model}_{target}_{fct_h}_{seq_len}_temporal_variation.png', 
                    dpi=256, transparent=True)

    if show:
        plt.show()
    else:
        plt.close()

# Plot Function
def plot_qq(y_test, y_pred, target_val):

    plt.figure(figsize = (6, 6))
    sns.set_style('white')
    axis_limit = max(max(y_test), max(y_pred))
    plt.plot(y_test, y_pred, '.')
    plt.plot([0, axis_limit], [0, axis_limit], 'k-')

    if target_val in ['Synedra', 'ToxicCyano']:
        [plt.axhline(y=i, linestyle='--', lw=1, color='gray', alpha=0.5) for i in [0, 100, 300, 1000]]
        [plt.axvline(x=i, linestyle='--', lw=1, color='gray', alpha=0.5) for i in [0, 100, 300, 1000]]

    if target_val in ['2MIB', 'Geosmin']:
        [plt.axhline(y=i, linestyle='--', lw=1, color='gray', alpha=0.5) for i in [0, 0.010, 0.020, 0.050]]
        [plt.axvline(x=i, linestyle='--', lw=1, color='gray', alpha=0.5) for i in [0, 0.010, 0.020, 0.050]]

    rmse, r2, acc, recall = calculate_score(y_test, y_pred, target_val)
    if target_val in ['Synedra', 'ToxicCyano']: plt.text(axis_limit*0.01, axis_limit*0.76, f'RMSE: {rmse:.1f}', size=14)
    if target_val in ['2MIB', 'Geosmin']: plt.text(axis_limit*0.01, axis_limit*0.76, f'RMSE: {rmse:.4f}', size=14)
    plt.text(axis_limit*0.01, axis_limit*0.83, f'R2: {r2:.2f}', size=14)
    plt.text(axis_limit*0.01, axis_limit*0.90, f'Accuracy: {acc*100:.1f}%', size=14)

    plt.title((f'{target_val}'), size=16)
    plt.xlabel('True') ; plt.ylabel('Pred');
    plt.show()


def plot_pred(pred_vals_final, Model_data_new, target):

    true_vals_final = pd.DataFrame(Model_data_new[target])
    true_vals_final = true_vals_final[true_vals_final.index >= pred_vals_final.index[0]]
    pred_result_df = pred_vals_final.join(true_vals_final)

    true_val = pred_result_df[target]
    pred_val = pred_result_df[pred_result_df[target].isna() == True].iloc[:,0]

    # Plotting
    plt.figure(figsize=(10, 4))
    plt.plot(true_val.index, true_val, color='black', label='Actuals')
    plt.plot(pred_val.index, pred_val, color='red', label='Predictions', marker='o', markersize=3)
    # Connect the end of actuals with the start of predictions
    if not true_val.dropna().empty and not pred_val.dropna().empty:
        plt.plot([true_val.dropna().index[-1], pred_val.dropna().index[0]], [true_val.dropna().iloc[-1], pred_val.dropna().iloc[0]], color='red')

    plt.axhline(y=10, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=30, color='gray', linestyle='--', linewidth=2, alpha=0.5)
    plt.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xlim(true_val.index[0], b.index[-1]+ timedelta(days=1))

    plt.ylim(0 - pred_val.max() * 0.05, max(true_val.max() * 1.05, pred_val.max() * 1.05, 105))
    plt.yticks(size=13)
    plt.ylabel('Predicted Turbidity (NTU)', fontsize=13)

    #Annotate predicted values with alternating vertical offsets
    for i, (x, y) in enumerate(zip(pred_val.index, pred_val)):
        offset = 5 if i % 2 == 0 else -5
        plt.text(x, y + offset, f'{y:.2f}', fontsize=9, color='blue', ha='center')

    plt.tight_layout()
    plt.show()