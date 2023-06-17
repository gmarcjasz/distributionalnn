import pandas as pd
import numpy as np
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp  
from tensorflow_probability import distributions as tfd
from datetime import datetime, timedelta
import tensorflow.compat.v2.keras as keras
import logging
import sys
import os
import optuna
import time

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

distribution = 'Normal'
paramcount = {'Normal': 2,
              'StudentT': 3,
              'JSU': 4,
              'SinhArcsinh': 4,
              'NormalInverseGaussian': 4,
              'Point': None
}
val_multi = 13 # int for # of re-trains - 1 corresponds to old approach
val_window = 364 // val_multi

if not os.path.exists(f'../trialfiles'):
    os.mkdir(f'../trialfiles')

INP_SIZE = 221
activations = ['sigmoid', 'relu', 'elu', 'tanh', 'softplus', 'softmax']

binopt = [True, False]

cty = 'DE'
storeDBintmp = False

if len(sys.argv) > 1:
    cty = sys.argv[1]
if len(sys.argv) > 2:
    distribution = sys.argv[2]
if len(sys.argv) > 3 and bool(sys.argv[3]):
    storeDBintmp = True

print(cty, distribution)

if cty != 'DE':
    raise ValueError('Incorrect country')
if distribution not in paramcount:
    raise ValueError('Incorrect distribution')

# read data file
data = pd.read_csv(f'../Datasets/{cty}.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
data = data.iloc[:4*364*24] # take the first 4 years - 1456 days

def objective(trial):
    # prepare the input/output dataframes
    Y = np.zeros((1456, 24))
    Yf = np.zeros((364, 24))
    for d in range(1456):
        Y[d, :] = data.loc[data.index[d*24:(d+1)*24], 'Price'].to_numpy()
    # Y = Y[7:, :] # skip first 7 days
    for d in range(364):
        Yf[d, :] = data.loc[data.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    # 
    X = np.zeros((1092+364, INP_SIZE))
    for d in range(7, 1092+364):
        X[d, :24] = data.loc[data.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = data.loc[data.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = data.loc[data.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy() # D-3 price
        X[d, 72:96] = data.loc[data.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy() # D-7 price
        X[d, 96:120] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = data.loc[data.index[(d-7)*24:(d-6)*24], data.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = data.loc[data.index[(d)*24:(d+1)*24], data.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = data.loc[data.index[(d-1)*24:(d)*24], data.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[3]].to_numpy() # D-2 EUA
        X[d, 217] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[4]].to_numpy() # D-2 API2_Coal
        X[d, 218] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[5]].to_numpy() # D-2 TTF_Gas
        X[d, 219] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[6]].to_numpy() # D-2 Brent oil
        X[d, 220] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if trial.suggest_categorical('price_D-1', binopt):
        colmask[:24] = [True] * 24
    if trial.suggest_categorical('price_D-2', binopt):
        colmask[24:48] = [True] * 24
    if trial.suggest_categorical('price_D-3', binopt):
        colmask[48:72] = [True] * 24
    if trial.suggest_categorical('price_D-7', binopt):
        colmask[72:96] = [True] * 24
    if trial.suggest_categorical('load_D', binopt):
        colmask[96:120] = [True] * 24
    if trial.suggest_categorical('load_D-1', binopt):
        colmask[120:144] = [True] * 24
    if trial.suggest_categorical('load_D-7', binopt):
        colmask[144:168] = [True] * 24
    if trial.suggest_categorical('RES_D', binopt):
        colmask[168:192] = [True] * 24
    if trial.suggest_categorical('RES_D-1', binopt):
        colmask[192:216] = [True] * 24
    if trial.suggest_categorical('EUA', binopt):
        colmask[216] = True
    if trial.suggest_categorical('Coal', binopt):
        colmask[217] = True
    if trial.suggest_categorical('Gas', binopt):
        colmask[218] = True
    if trial.suggest_categorical('Oil', binopt):
        colmask[219] = True
    if trial.suggest_categorical('Dummy', binopt):
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xwhole = X.copy()
    Ywhole = Y.copy()
    Yfwhole = Yf.copy()
    metrics_sub = []
    for train_no in range(val_multi):
        start = val_window * train_no
        X = Xwhole[start:1092+start, :]
        Xf = Xwhole[1092+start:1092+start+val_window, :]
        Y = Ywhole[start:1092+start, :]
        Yf = Ywhole[1092+start:1092+start+val_window, :]
        X = X[7:1092, :]
        Y = Y[7:1092, :]
        # begin building a model
        inputs = keras.Input(X.shape[1]) # <= INP_SIZE as some columns might have been turned off
        # batch normalization
        # we decided to always normalize the inputs
        batchnorm = True#trial.suggest_categorical('batch_normalization', [True, False])
        if batchnorm:
            norm = keras.layers.BatchNormalization()(inputs)
            last_layer = norm
        else:
            last_layer = inputs
        # dropout
        dropout = trial.suggest_categorical('dropout', binopt)
        if dropout:
            rate = trial.suggest_float('dropout_rate', 0, 1)
            drop = keras.layers.Dropout(rate)(last_layer)
            last_layer = drop
        # regularization of 1st hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h1_activation = trial.suggest_categorical('regularize_h1_activation', binopt)
        regularize_h1_kernel = trial.suggest_categorical('regularize_h1_kernel', binopt)
        h1_activation_rate = (0.0 if not regularize_h1_activation
                              else trial.suggest_float('h1_activation_rate_l1', 1e-5, 1e1, log=True))
        h1_kernel_rate = (0.0 if not regularize_h1_kernel
                          else trial.suggest_float('h1_kernel_rate_l1', 1e-5, 1e1, log=True))
        # define 1st hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_1', 16, 1024, log=False),
                                    activation=trial.suggest_categorical('activation_1', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
        # regularization of 2nd hidden layer,
        #activation - output, kernel - weights/parameters of input
        regularize_h2_activation = trial.suggest_categorical('regularize_h2_activation', binopt)
        regularize_h2_kernel = trial.suggest_categorical('regularize_h2_kernel', binopt)
        h2_activation_rate = (0.0 if not regularize_h2_activation
                              else trial.suggest_float('h2_activation_rate_l1', 1e-5, 1e1, log=True))
        h2_kernel_rate = (0.0 if not regularize_h2_kernel
                          else trial.suggest_float('h2_kernel_rate_l1', 1e-5, 1e1, log=True))
        # define 2nd hidden layer with regularization
        hidden = keras.layers.Dense(trial.suggest_int('neurons_2', 16, 1024, log=False),
                                    activation=trial.suggest_categorical('activation_2', activations),
                                    # kernel_initializer='ones',
                                    kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                    activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)
        if paramcount[distribution] is None:
            outputs = keras.layers.Dense(24, activation='linear')(hidden)
            model = keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                          # loss=lambda y, rv_y: -rv_y.log_prob(y),
                          loss='mae',
                          metrics='mae')
        else:
            # now define parameter layers with their regularization
            param_layers = []
            param_names = ["loc", "scale", "tailweight", "skewness"]
            for p in range(paramcount[distribution]):
                # regularize_param_kernel = True
                # param_kernel_rate = 0.1
                regularize_param_kernel = trial.suggest_categorical('regularize_'+param_names[p], binopt)
                param_kernel_rate = (0.0 if not regularize_param_kernel
                                     else trial.suggest_float(param_names[p]+'_rate_l1', 1e-5, 1e1, log=True))
                param_layers.append(keras.layers.Dense(
                    24, activation='linear', # kernel_initializer='ones',
                    kernel_regularizer=keras.regularizers.L1(param_kernel_rate))(hidden))
            # concatenate the parameter layers to one
            linear = tf.keras.layers.concatenate(param_layers)
            # define outputs
            if distribution == 'Normal':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.Normal(
                            loc=t[..., :24],
                            scale = 1e-3 + 3 * tf.math.softplus(t[..., 24:])))(linear)
            elif distribution == 'StudentT':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.StudentT(
                            loc=t[..., :24],
                            scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                            df=1 + 3 * tf.math.softplus(t[..., 48:])))(linear)
            elif distribution == 'JSU':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.JohnsonSU(
                            loc=t[..., :24],
                            scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                            tailweight= 1 + 3 * tf.math.softplus(t[..., 48:72]),
                            skewness=t[..., 72:]))(linear)
            elif distribution == 'SinhArcsinh':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.SinhArcsinh(
                            loc=t[..., :24],
                            scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                            tailweight=1e-3 + 3 * tf.math.softplus(t[..., 48:72]),
                            skewness=t[..., 72:]))(linear)
            elif distribution == 'NormalInverseGaussian':
                outputs = tfp.layers.DistributionLambda(
                        lambda t: tfd.NormalInverseGaussian(
                            loc=t[..., :24],
                            scale=1e-3 + 3 * tf.math.softplus(t[..., 24:48]),
                            tailweight=1e-3 + 3 * tf.math.softplus(t[..., 48:72]),
                            skewness=t[..., 72:]))(linear)
            else:
                raise ValueError(f'Incorrect distribution {distribution}')
            model = keras.Model(inputs = inputs, outputs=outputs)
            model.compile(optimizer=keras.optimizers.Adam(trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)),
                          loss=lambda y, rv_y: -rv_y.log_prob(y),
                          metrics='mae')
        # '''
        # define callbacks
        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
        model.fit(X, Y, epochs=1500, validation_data=(Xf, Yf), callbacks=callbacks, batch_size=32, verbose=True)

        metrics = model.evaluate(Xf, Yf) # for point its a list of one [loss, MAE]
        metrics_sub.append(metrics[0])
        # we optimize the returned value, -1 will always take the model with best MAE
    return np.mean(metrics_sub)

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_prob_{distribution.lower()}'
if storeDBintmp:
    storage_name = f'sqlite:////tmp/{study_name}'
else:
    storage_name = f'sqlite:///../trialfiles/{study_name}'
# below calls to create either random sampler and default (tree parzen estimator)
# study can be resumed using a different one
# (or one process can sample randomly, the second one run the TPE sampler at the same time)
# study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=optuna.samplers.RandomSampler())
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=128, show_progress_bar=True)
best_params = study.best_params
print(best_params)
print(study.trials_dataframe())
if storeDBintmp:
    print('Trials DB stored in /tmp')
