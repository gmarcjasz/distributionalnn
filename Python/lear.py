import pandas as pd
import numpy as np
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
# import tensorflow_probability as tfp  
# from tensorflow_probability import distributions as tfd
# import tensorflow.compat.v2.keras as keras
from datetime import datetime, timedelta
import logging
import sys
import os
# import optuna
import time
from multiprocessing import Pool
import sklearn.linear_model
# time.sleep(5)

if not os.path.exists('../lear_forecasts'):
    os.mkdir('../lear_forecasts')

INP_SIZE = 221

binopt = [True, False]

cty = sys.argv[1]

# read data file
data = pd.read_csv(f'../Datasets/{cty}.csv', index_col=0)
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]
# data = data.iloc[:4*364*24] # take the first 4 years - 1456 days

def runoneday(inp):
    cal, dayno = inp
    df = data.iloc[dayno*24:dayno*24+1456*24+24]
    df = df[-(cal+1)*24:]
    # prepare the input/output dataframes
    Y = np.zeros((cal, 24))
    # Yf = np.zeros((1, 24)) # no Yf for rolling prediction
    for d in range(cal):
        Y[d, :] = df.loc[df.index[d*24:(d+1)*24], 'Price'].to_numpy()
    Y = Y[7:, :] # skip first 7 days
    # for d in range(1):
    #     Yf[d, :] = df.loc[df.index[(d+1092)*24:(d+1093)*24], 'Price'].to_numpy()
    X = np.zeros((cal+1, INP_SIZE))
    for d in range(7, cal+1):
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
    Xf = X[-1:, :]
    X = X[7:-1, :]
    predDF = pd.DataFrame(index=df.index[-24:])
    predDF['real'] = df.loc[df.index[-24:], 'Price'].to_numpy()
    predDF['forecast'] = pd.NA
    for h in range(24):
        # begin building a model
        model = sklearn.linear_model.LassoCV(eps=1e-6, n_alphas=100, cv=7)
        model.fit(X, Y[:, h])
        pred = model.predict(Xf)[0]
        predDF.loc[predDF.index[h], 'forecast'] = pred
    # pred = model.predict(np.tile(Xf, (10000, 1)))
    predDF.to_csv(os.path.join('../lear_forecasts', str(cal) + '_' + datetime.strftime(df.index[-24], '%Y-%m-%d')))
    # np.savetxt(os.path.join('forecasts_prob', datetime.strftime(df.index[-24], '%Y-%m-%d')), pred, delimiter=',', fmt='%.4f')
    print(predDF)
    return predDF

# optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))
# study_name = 'on_new_data' # 'on_new_data_no_feature_selection'
# storage_name = f'sqlite:///{study_name}'
# below calls to create either random sampler and default (tree parzen estimator)
# study can be resumed using a different one
# (or one process can sample randomly, the second one run the TPE sampler at the same time)
# study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, sampler=optuna.samplers.RandomSampler())
# study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
# study.optimize(objective, n_trials=1000, show_progress_bar=True)
# best_params = study.best_params
# print(best_params)
# print(study.trials_dataframe())

inputlist = [(int(sys.argv[2]), day) for day in range(len(data) // 24 - 1456)]
print(len(inputlist))

# for e in inputlist:
#     _ = runoneday(e)

with Pool(max(2, os.cpu_count() - 4)) as p:
    _ = p.map(runoneday, inputlist)
