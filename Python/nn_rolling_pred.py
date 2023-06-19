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
from multiprocessing import Pool
import json

# Accepts arguments:
#     cty (currently only DE), default: DE
#     distribution (Normal, StudentT, JSU, SinhArcsinh and NormalInverseGaussian), default: Normal

distribution = 'Normal'
paramcount = {'Normal': 2,
              'StudentT': 3,
              'JSU': 4,
              'SinhArcsinh': 4,
              'NormalInverseGaussian': 4,
              'Point': None,
}

INP_SIZE = 221
# activations, neurons and params read from the trials info

cty = 'DE'

if len(sys.argv) > 1:
    cty = sys.argv[1]
if len(sys.argv) > 2:
    distribution = sys.argv[2]

if not os.path.exists(f'../forecasts_probNN_{distribution.lower()}'):
    os.mkdir(f'../forecasts_probNN_{distribution.lower()}')

if not os.path.exists(f'../distparams_probNN_{distribution.lower()}'):
    os.mkdir(f'../distparams_probNN_{distribution.lower()}')

if not os.path.exists(f'../trialfiles'):
    os.mkdir(f'../trialfiles')

print(cty, distribution)

if cty != 'DE':
    raise ValueError('Incorrect country')
if distribution not in paramcount:
    raise ValueError('Incorrect distribution')

# read data file
data = pd.read_csv(f'../Datasets/{cty}.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
# data = data.iloc[:4*364*24] # take the first 4 years - 1456 days

def runoneday(inp):
    params, dayno = inp
    df = data.iloc[dayno*24:dayno*24+1456*24+24]
    # prepare the input/output dataframes
    Y = np.zeros((1456, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(1456):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'Price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days
    # for d in range(1):
    #     Yf[d, :] = df.loc[df.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    X = np.zeros((1456+1, INP_SIZE))
    for d in range(7, 1456+1):
        X[d, :24] = df.loc[df.index[(d-1)*24:(d)*24], 'Price'].to_numpy() # D-1 price
        X[d, 24:48] = df.loc[df.index[(d-2)*24:(d-1)*24], 'Price'].to_numpy() # D-2 price
        X[d, 48:72] = df.loc[df.index[(d-3)*24:(d-2)*24], 'Price'].to_numpy() # D-3 price
        X[d, 72:96] = df.loc[df.index[(d-7)*24:(d-6)*24], 'Price'].to_numpy() # D-7 price
        X[d, 96:120] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[1]].to_numpy() # D load forecast
        X[d, 120:144] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[1]].to_numpy() # D-1 load forecast
        X[d, 144:168] = df.loc[df.index[(d-7)*24:(d-6)*24], df.columns[1]].to_numpy() # D-7 load forecast
        X[d, 168:192] = df.loc[df.index[(d)*24:(d+1)*24], df.columns[2]].to_numpy() # D RES sum forecast
        X[d, 192:216] = df.loc[df.index[(d-1)*24:(d)*24], df.columns[2]].to_numpy() # D-1 RES sum forecast
        X[d, 216] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[3]].to_numpy() # D-2 EUA
        X[d, 217] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[4]].to_numpy() # D-2 API2_Coal
        X[d, 218] = df.loc[df.index[(d-2)*24:(d-1)*24:24], df.columns[5]].to_numpy() # D-2 TTF_Gas
        X[d, 219] = data.loc[data.index[(d-2)*24:(d-1)*24:24], data.columns[6]].to_numpy() # D-2 Brent oil
        X[d, 220] = data.index[d].weekday()
    # '''
    # input feature selection
    colmask = [False] * INP_SIZE
    if params['price_D-1']:
        colmask[:24] = [True] * 24
    if params['price_D-2']:
        colmask[24:48] = [True] * 24
    if params['price_D-3']:
        colmask[48:72] = [True] * 24
    if params['price_D-7']:
        colmask[72:96] = [True] * 24
    if params['load_D']:
        colmask[96:120] = [True] * 24
    if params['load_D-1']:
        colmask[120:144] = [True] * 24
    if params['load_D-7']:
        colmask[144:168] = [True] * 24
    if params['RES_D']:
        colmask[168:192] = [True] * 24
    if params['RES_D-1']:
        colmask[192:216] = [True] * 24
    if params['EUA']:
        colmask[216] = True
    if params['Coal']:
        colmask[217] = True
    if params['Gas']:
        colmask[218] = True
    if params['Oil']:
        colmask[219] = True
    if params['Dummy']:
        colmask[220] = True
    X = X[:, colmask]
    # '''
    Xf = X[-1:, :]
    X = X[7:-1, :]
    # begin building a model
    inputs = keras.Input(X.shape[1]) # <= INP_SIZE as some columns might have been turned off
    # batch normalization
    batchnorm = True#params['batch_normalization'] # trial.suggest_categorical('batch_normalization', [True, False])
    if batchnorm:
        norm = keras.layers.BatchNormalization()(inputs)
        last_layer = norm
    else:
        last_layer = inputs
    # dropout
    dropout = params['dropout'] # trial.suggest_categorical('dropout', [True, False])
    if dropout:
        rate = params['dropout_rate'] # trial.suggest_float('dropout_rate', 0, 1)
        drop = keras.layers.Dropout(rate)(last_layer)
        last_layer = drop
    # regularization of 1st hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h1_activation = params['regularize_h1_activation']
    regularize_h1_kernel = params['regularize_h1_kernel']
    h1_activation_rate = (0.0 if not regularize_h1_activation 
                          else params['h1_activation_rate_l1'])
    h1_kernel_rate = (0.0 if not regularize_h1_kernel 
                      else params['h1_kernel_rate_l1'])
    # define 1st hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_1'], 
                                activation=params['activation_1'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h1_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h1_activation_rate))(last_layer)
    # regularization of 2nd hidden layer, 
    #activation - output, kernel - weights/parameters of input
    regularize_h2_activation = params['regularize_h2_activation']
    regularize_h2_kernel = params['regularize_h2_kernel']
    h2_activation_rate = (0.0 if not regularize_h2_activation 
                          else params['h2_activation_rate_l1'])
    h2_kernel_rate = (0.0 if not regularize_h2_kernel 
                      else params['h2_kernel_rate_l1'])
    # define 2nd hidden layer with regularization
    hidden = keras.layers.Dense(params['neurons_2'], 
                                activation=params['activation_2'],
                                # kernel_initializer='ones',
                                kernel_regularizer=keras.regularizers.L1(h2_kernel_rate),
                                activity_regularizer=keras.regularizers.L1(h2_activation_rate))(hidden)
    if paramcount[distribution] is None:
        outputs = keras.layers.Dense(24, activation='linear')(hidden)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                      loss='mae',
                      metrics='mae')
    else:
        # now define parameter layers with their regularization
        param_layers = []
        param_names = ["loc", "scale", "tailweight", "skewness"]
        for p in range(paramcount[distribution]):
            regularize_param_kernel = params['regularize_'+param_names[p]]
            param_kernel_rate = (0.0 if not regularize_param_kernel 
                                 else params[str(param_names[p])+'_rate_l1'])
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
        model.compile(optimizer=keras.optimizers.Adam(params['learning_rate']),
                      loss=lambda y, rv_y: -rv_y.log_prob(y),
                      metrics='mae')
    # '''
    # define callbacks
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]
    perm = np.random.permutation(np.arange(X.shape[0]))
    VAL_DATA = .2
    trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
    valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
    model.fit(X[trainsubset], Y[trainsubset], epochs=1500, validation_data=(X[valsubset], Y[valsubset]), callbacks=callbacks, batch_size=32, verbose=False)
    
    # metrics = model.evaluate(Xf, Yf) # for point its a list of one [loss, MAE]
    # we optimize the returned value, -1 will always take the model with best MAE

    # pred = model.predict(Xf)[0]
    if paramcount[distribution] is not None:
        dist = model(Xf)
        if distribution == 'Normal':
            getters = {'loc': dist.loc, 'scale': dist.scale}
        elif distribution == 'StudentT':
            getters = {'loc': dist.loc, 'scale': dist.scale, 'df': dist.df}
        elif distribution in {'JSU', 'SinhArcsinh', 'NormalInverseGaussian'}:
            getters = {'loc': dist.loc, 'scale': dist.scale, 
                       'tailweight': dist.tailweight, 'skewness': dist.skewness}
        print(getters)
        params = {k: [float(e) for e in v.numpy()[0]] for k, v in getters.items()}
        print(params)
        json.dump(params, open(os.path.join(f'../distparams_probNN_{distribution.lower()}', datetime.strftime(df.index[-24], '%Y-%m-%d')), 'w'))
        pred = model.predict(np.tile(Xf, (10000, 1)))
        predDF = pd.DataFrame(index=df.index[-24:])
        predDF['real'] = df.loc[df.index[-24:], 'Price'].to_numpy()
        predDF['forecast'] = pd.NA
        predDF.loc[predDF.index[:], 'forecast'] = pred.mean(0)
        # predDF.to_csv(os.path.join('../forecasts', datetime.strftime(df.index[-24], '%Y-%m-%d')))
        np.savetxt(os.path.join(f'../forecasts_probNN_{distribution.lower()}', datetime.strftime(df.index[-24], '%Y-%m-%d')), pred, delimiter=',', fmt='%.3f')
    else:
        predDF = pd.DataFrame(index=df.index[-24:])
        predDF['real'] = df.loc[df.index[-24:], 'Price'].to_numpy()
        predDF['forecast'] = pd.NA
        predDF.loc[predDF.index[:], 'forecast'] = model.predict(Xf)[0]
        pred = model.predict(Xf)
        np.savetxt(os.path.join(f'../forecasts_probNN_{distribution.lower()}', datetime.strftime(df.index[-24], '%Y-%m-%d')), pred, delimiter=',', fmt='%.3f')
    print(predDF)
    return predDF

optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
study_name = f'FINAL_DE_selection_prob_{distribution.lower()}' # 'on_new_data_no_feature_selection'
storage_name = f'sqlite:///../trialfiles/{study_name}'
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
print(study.trials_dataframe())
best_params = study.best_params
print(best_params)

inputlist = [(best_params, day) for day in range(len(data) // 24 - 1456)]
print(len(inputlist))

# for e in inputlist:
#     _ = runoneday(e)

with Pool(max(os.cpu_count() // 4, 1)) as p:
    _ = p.map(runoneday, inputlist)
