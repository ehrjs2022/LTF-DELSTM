import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable


###### Model structure
##### LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dr = 0.2):
        super(LSTM, self).__init__()
        self.dropout = nn.Dropout(p = dr)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activ = nn.ELU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.activ(out)
        
        return out
    
##### GRU
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dr = 0.2):
        super(GRU, self).__init__()
        self.dropout = nn.Dropout(p = dr)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activ = nn.ELU()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        out = self.activ(out)
        
        return out

##### Bi-LSTM
class BiLSTM(nn.Module):
    
    torch.manual_seed(42)
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dr = 0.2):
        super(BiLSTM, self).__init__()
        self.dropout = nn.Dropout(p = dr)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  
        self.activ = nn.ELU()
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim)).to(x.device)
        c0 = Variable(torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim)).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        # out = self.activ(out)
        
        return out

   
###### Utility function for modeling TODO separate below functions as another module
def create_sequences(data, time_data, target, seq_length, forecast_horizon, criteria_hour):
    from datetime import timedelta
    period = list(data.index)
    targets = list(data.columns)
    period_x = list(time_data.index)

    start_ = period.index(data.loc[~data[target].isna(), :].index[0])
    end_ = period.index(data.loc[~data[target].isna(), :].index[-1])

    start_x_time = (data.loc[~data[target].isna(), :].index[0] + pd.offsets.Hour(1)).replace(hour=0, minute=0)
    if (data.loc[~data[target].isna(), :].index[0] + pd.offsets.Hour(1)) < start_x_time:
        start_x_time = start_x_time
    else:
        start_x_time = start_x_time + timedelta(days=1)

    start_x = period_x.index(start_x_time)

    end_x_time = data.loc[~data[target].isna(), :].index[-1].replace(hour=criteria_hour, minute=0, second=0, microsecond=0)
    if data.loc[~data[target].isna(), :].index[-1] > end_x_time:
        end_x_time = end_x_time
    else:
        end_x_time = end_x_time - timedelta(days=1)
    end_x = period_x.index(end_x_time)

    d = np.array(data)[start_ : end_+1, :]
    e = np.array(time_data)[start_x : end_x + 1, :]

    xs = []
    ys = []

    for i in range(len(d) - seq_length - forecast_horizon):
        x = e[i*24:(i*24 + (seq_length-1)*24+criteria_hour+1), :]
        y = d[i + 1 + seq_length:i + 1 + seq_length + forecast_horizon, targets.index(target)]
        xs.append(x)
        ys.append(y)
        
    return np.array(xs), np.array(ys) 

def create_sequences_preprocess(Model_data, Sensor_data, target, seq_len, fct_h, criteria_hour, predictors):
    from datetime import timedelta
    all_dates = pd.DatetimeIndex([])
    valid_intervals = []
    current_start = None
    Sensor_dat = Sensor_data[[target] + predictors]
    x_ff = np.empty((0, (seq_len-1)*24+criteria_hour+1, len([target] + predictors)))
    y_ff = np.empty((0, fct_h))
    Model_data_2 = Model_data[[target] + predictors]
    for i in range(len(Model_data_2)):
        if not Model_data_2.iloc[i][[target] + predictors].isna().any():
            if current_start is None:
                current_start = Model_data_2.index[i]
        else:
            if current_start is not None:
                valid_intervals.append((current_start, Model_data_2.index[i-1]))
                current_start = None

    if current_start is not None:
        valid_intervals.append((current_start, Model_data_2.index[-1]))

    for start, end in valid_intervals:
        valid_data = Model_data_2.loc[start:end]
        x, y = create_sequences(valid_data, Sensor_dat, target, seq_len, fct_h, criteria_hour)
        try:
            x_ff = np.concatenate([x_ff, x])
            y_ff = np.concatenate([y_ff, y])
            all_dates = all_dates.append(pd.date_range(start=start, end=end - timedelta(days=fct_h+seq_len)))
        except:
            x = None
            y = None
    
    return x_ff, y_ff, all_dates

def create_sequences_pred(data, target, seq_length):
    
    #### NOTE
    # index for data must be date time index!
    
    period = list(data.index)
    
    start_ = period.index(data.loc[~data[target].isna(), :].index[0])
    end_ = period.index(data.loc[~data[target].isna(), :].index[-1])
    
    d = np.array(data)[start_ : end_+1, :]

    xs = []
    for i in range(len(d) - seq_length + 1):
        x = d[i:(i + seq_length), :]

        xs.append(x)
        
    return np.array(xs)

def data_split(x, y, tr_ratio, vl_ratio, te_ratio):
    samples = len(x)  # Number of samples

    tr_x = x.iloc[:int(np.ceil(samples * tr_ratio))]
    tr_y = y.iloc[:int(np.ceil(samples * tr_ratio))]
    
    vl_x = x.iloc[int(np.ceil(samples * tr_ratio)):int(np.ceil(samples * tr_ratio) + np.ceil(samples * vl_ratio))]
    vl_y = y.iloc[int(np.ceil(samples * tr_ratio)):int(np.ceil(samples * tr_ratio) + np.ceil(samples * vl_ratio))]
    
    if te_ratio > 0:
        te_x = x.iloc[-int(np.ceil(samples * te_ratio)):]
        te_y = y.iloc[-int(np.ceil(samples * te_ratio)):]
    else:
        te_x, te_y = pd.DataFrame(), pd.DataFrame()

    return tr_x, tr_y, vl_x, vl_y, te_x, te_y

def build_model():
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    model = tf.keras.Sequential([
        layers.Dense(2000, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1000, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(500, activation='relu'),
        layers.Dense(1)
        ])
    
    optimizer = tf.keras.optimizers.Adam()
    
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    return model
        

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg=None, known_inputs=None):

        trg_length = trg.size(1) if trg is not None else 1
        hn, cn = self.encoder(src)

        src_mask = torch.ones_like(src)
        src_mask[:, :-trg_length, :] = 0
        masked_src = src * src_mask

        if known_inputs is not None:
            known_mask = torch.ones_like(known_inputs) 
            known_mask[:, :-trg_length, :] = 0
            masked_known_inputs = known_inputs * known_mask
        else:
            masked_known_inputs = None

        output = self.decoder(masked_src, masked_known_inputs, hn, cn)

        return output

class BiLSTMDecoder(nn.Module):
    def __init__(self, input_dim, known_dim, hidden_dim, num_layers, output_dim, dr=0.2):
        super(BiLSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim + known_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(p=dr)
        self.activ = nn.ELU()
############# weather forecast 추가
    def forward(self, x, known_inputs, hn, cn):
        if known_inputs is None:  # No known inputs
            combined_inputs = x
        else:
            combined_inputs = torch.cat((x, known_inputs), dim=-1)
        out, _ = self.lstm(combined_inputs, (hn, cn))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dr=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dr)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        return hn, cn

def get_indices(var_names, var_indices):
    return [var_indices[var] for var in var_names]

def create_mask(input_tensor, padding_idx=0):
    return (input_tensor != padding_idx).float()
