from quantreg import quantreg
from multiprocessing import Pool
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

data = pd.read_csv('lear_forecast.csv', index_col=0) # contains 5 forecast columns and real price
data.loc[data.index[:], 'ones'] = 1.0
data.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in data.index]

if not os.path.exists(os.path.join('..', 'lear_QRA')):
    os.mkdir(os.path.join('..', 'lear_QRA'))

# on avg
def quantreg_day(dno):
    # dno is range(736-182)
    X = data.iloc[dno*24:dno*24+182*24]
    d = datetime.strftime(data.index[dno*24+182*24+1], '%Y-%m-%d')
    Xf = data.iloc[dno*24+182*24:dno*24+183*24][['forecast56', 'forecast84', 'forecast1092', 'forecast1456', 'ones']].to_numpy()
    Y = X['real'].to_numpy()
    X = X[['forecast56', 'forecast84', 'forecast1092', 'forecast1456', 'ones']].to_numpy()
    taus = [i/100 for i in range(1,100)]
    betas = {}
    for t in taus:
        betas[t] = quantreg(X, Y, t)
    prob = np.zeros((len(taus), 24))
    for h in range(24):
        h_res = []
        for t in taus:
            h_res.append(np.dot(Xf[h, :], betas[t]))
        # print(h_res)
        h_res.sort()
        prob[:, h] = h_res
    np.savetxt(os.path.join('..', 'lear_QRA', d), prob, delimiter=',', fmt='%.3f')

with Pool() as p:
    _ = p.map(quantreg_day, list(range(736-182)))
