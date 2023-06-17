from quantreg import quantreg
import os
import numpy as np
import pandas as pd

# read forecasts
filelist = os.listdir('lear_forecasts')
filelist = [e for e in filelist if e.startswith('1092')]
point = np.zeros((736 * 24,))
real = np.zeros((736 * 24,))
for no, fname in enumerate(sorted(filelist)):
    temp = pd.read_csv(f'lear_forecasts/{fname}', index_col=0)
    point[no*24:(no+1)*24] = temp.forecast.to_numpy()
    real[no*24:(no+1)*24] = temp.real.to_numpy()

print(point)

# CODE IMPLEMENTED IN QRA.py
