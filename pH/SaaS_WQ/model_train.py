from torch.optim.lr_scheduler import _LRScheduler
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable

##### Set seed    
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
# pbar = tqdm(range(data_imput.shape[1]), ncols = 100, ascii = " =", leave = True)

##### Model training
##### Model training
def train_model(model, tr_loader,
                vl_data, #### Time series dataset
                loss_function,
                optimizer,
                scheduler,
                num_epochs,
                min_, # For inverse transformation TODO change function using scaler function 
                max_,
                patience=200):
    
    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    best_model = None
    
    for epoch in range(num_epochs): ### TODO add pbar
        model.train()
        for x_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            #### Inverse scaling            
            y_pred = y_pred * (max_ - min_) + min_
            y_batch = y_batch * (max_ - min_) + min_
            
            loss = loss_function(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        ##### For early stopping
        model.eval()
        with torch.no_grad():
            x_vl, y_vl = vl_data.x, vl_data.y
            y_pred = model(x_vl)
            
            y_pred = y_pred * (max_ - min_) + min_
            y_vl = y_vl * (max_ - min_) + min_
            
            vl_loss = loss_function(y_pred, y_vl)
            
            if vl_loss < best_loss:
                best_loss = vl_loss
                best_epoch = epoch
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}')
                return best_loss, best_epoch, best_model
    
    return best_loss, best_epoch, best_model


##### For create dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

##### For create dataset
class TimeSeriesDataset2(Dataset):
    def __init__(self, x, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index]

##### For hyperparameter optimization
import optuna
from optuna.pruners import HyperbandPruner
def hyper_param_opt(model_name, target, 
                    x_train, y_train, x_val, y_val,
                    min, max,
                    info, n_trials=200):
    
    study_name = f"{model_name}_{target}_{info}_optimization"
    
    # Create a study object with Hyperband Pruner
    pruner = HyperbandPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )
    
    study.optimize(lambda trial: objective(trial, model_name,
                                           x_train, y_train, x_val, y_val,
                                           min, max), n_trials=n_trials) ### TODO modifying min, max part

    print(f"Best trial for {model_name} on {target}:")
    trial = study.best_trial
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return study, trial, trial.user_attrs["Best_model"] ## For saving optimization results

##### objective function
def objective(trial, model_name,
              x_train, y_train, x_val, y_val,
              min, max):
    
    # Define the hyperparameters to optimize
    hidden_dim = trial.suggest_int('hidden_dim', 16, 128)
    num_layers = trial.suggest_int('num_layers', 1, 5)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.2)

    # Model configuration
    input_dim = x_train.shape[2]
    output_dim = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model based on the model_name
    from SaaS_WQ.model import LSTM, GRU, BiLSTM
    
    if model_name == 'LSTM':
        model = LSTM(input_dim, hidden_dim, num_layers, output_dim, dr=dropout_rate).to(device)
    elif model_name == 'GRU':
        model = GRU(input_dim, hidden_dim, num_layers, output_dim, dr=dropout_rate).to(device)
    elif model_name == 'BiLSTM':
        model = BiLSTM(input_dim, hidden_dim, num_layers, output_dim, dr=dropout_rate).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(x_train, y_train, device)
    val_dataset = TimeSeriesDataset(x_val, y_val, device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training configuration
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    num_epochs = 1000
   
    # Training loop
    vl_loss, _, best_model = train_model(model, train_loader, val_dataset,
                                                loss_function, optimizer, scheduler, num_epochs, min, max)
    
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    trial.set_user_attr("Best_model", best_model)
    
    return vl_loss


####### Scheduler for optimizer - from TODO add source url
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


    

##### 주간데이터 기반 2-MIB, Geosmin, Synedra, Toxicyano
import optuna
from optuna.pruners import HyperbandPruner

def variable_selection(X, y, method='all', backward_threshold=0.05):
    """
    method:
        all : Takes ALL variables
        backward : eliminate one at time
    """
    if method == 'all':
        return X, y
    
    elif method == 'backward':
        cols_backward = X.columns.tolist()

        while True:
            changed = False
            model = sm.OLS(y, sm.add_constant(X[cols_backward])).fit(disp=0)
            worst_pvalue = model.pvalues[1:].max()

            if worst_pvalue > backward_threshold:
                changed = True
                cols_backward.remove(model.pvalues[1:].idxmax())

            if not changed:
                break

        print(f'BACKWARD ELIMINATION : {X.shape[1]} -> {len(cols_backward)} \nYou can Check by `X.columns`')
        return X[cols_backward], y

    else:
        return X, y    
        
def hyper_param_opt_week(model_name, target, x_train, y_train, info, n_trials=5):
    
    study_name = f"{model_name}_{target}_{info}_optimization"
    
# Create a study object with Hyperband Pruner
    pruner = HyperbandPruner()
    
    study = optuna.create_study(
        study_name=study_name,
        load_if_exists=True,
        direction='minimize',
        pruner=pruner
    )
    
    study.optimize(lambda trial: objective_week(trial, model_name, x_train, y_train), n_trials=n_trials)

    print(f"Best trial for {model_name} on {target}:")
    trial = study.best_trial
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return study, trial, trial.user_attrs["Best_model"]

##### objective function
def objective_week(trial, model_name,
                x_train, y_train):
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    # Create the model based on the model_name

    #### Hyperparameter optimization for ANN
    if model_name == 'ANN':
        EPOCHS = 500
        BATCH_SIZE = trial.suggest_int('BATCH_SIZE', 8, 128)
        optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        
        # Define optimizer based on suggestion
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        # Define the model architecture
        model = tf.keras.Sequential([
            Dense(2000, activation='relu'),
            Dropout(0.3),
            Dense(1000, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dense(1)
        ])
        
        model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
        
        # Define callbacks for early stopping and model checkpoint
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=0, save_best_only=True)
        es = EarlyStopping(monitor='val_loss', patience=10)
        
        # Train the model
        history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                            validation_split=0.2, callbacks=[checkpoint, es], verbose = False)
        
        # Get the best model based on validation loss
        best_loss = np.min(history.history['val_loss'])
        best_model = tf.keras.models.load_model('best_model.h5')
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        trial.set_user_attr("Best_model", best_model)


    #### Hyperparameter optimization for tree based ensembles
    else:
        if model_name == 'CV_rf':
            n_estimators = trial.suggest_categorical('n_estimators', [500, 1000])
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            max_depth = trial.suggest_categorical('max_depth', 4, 10)
            criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])

            param_grid_rf = {'n_estimators': [n_estimators],
                            'max_features': [max_features],
                            'max_depth': [max_depth],
                            'criterion': [criterion]
                            }
            
            rf = RandomForestRegressor(random_state=SEED)
            model = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5 ,n_jobs=-1, verbose=1) ### crossvalidation

        elif model_name == 'CV_xgb':
            min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
            gamma = trial.suggest_float('gamma', 0, 0.2)
            subsample = trial.suggest_float('subsample', 0.6, 1)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1)
            max_depth = trial.suggest_int('max_depth', 4, 10)

            param_grid_xgb = {
                'min_child_weight': [min_child_weight],
                'gamma': [gamma],
                'subsample': [subsample],
                'colsample_bytree': [colsample_bytree],
                'max_depth': [max_depth]
            }
            xgb = XGBRegressor(random_state=SEED)
            model = GridSearchCV(estimator=xgb, param_grid= param_grid_xgb, cv=5, n_jobs=-1, verbose=1)

        elif model_name == 'CV_svr':
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly'])
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            C = trial.suggest_int('C', 1, 10000)

            param_grid_svr = {
                'kernel': [kernel],
                'gamma': [gamma],
                'C': [C]
            }
            svr = SVR()
            model = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, n_jobs=-1, verbose=1)

        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        model.fit(x_train, y_train)
        # Train the model and get the best one
        best_loss, best_model = train_model_week(model, x_train, y_train)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        trial.set_user_attr("Best_model", best_model)
    
    return best_loss

##### Model training
def train_model_week(model, x_train, y_train):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer
    import numpy as np
    best_loss = float('inf')
    best_model = None
     
    # Perform 5-fold cross-validation on the training data
    cv_scores = cross_val_score(model, x_train, y_train, cv=5, scoring=make_scorer(mean_squared_error))
    train_loss = np.mean(cv_scores)
    
    # Fit the model on the entire training data
    model.fit(x_train, y_train)

    best_loss = train_loss
    best_model = model 

    return best_loss, best_model

def build_model():
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