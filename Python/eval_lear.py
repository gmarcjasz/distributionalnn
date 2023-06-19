import pandas as pd
import os
import numpy as np
from datetime import datetime

filelist = sorted(os.listdir(os.path.join('..', 'lear_forecasts')))
cal_lens = set([int(e.split('_')[0]) for e in filelist])
cal_lens = sorted(list(cal_lens))
cl_dfs = {}
for cl in cal_lens:
    singledfs = []
    subfilelist = [e for e in filelist if e.startswith(str(cl))]# [182:]
    for f in subfilelist:
        temp = pd.read_csv(os.path.join('..', 'lear_forecasts', f), index_col=0)
        temp.index = [datetime.strptime(e, '%Y-%m-%d %H:%M:%S') for e in temp.index]
        singledfs.append(temp)
    
    wholeindex = []
    for e in singledfs:
        for i in e.index:
            wholeindex.append(i)
    
    wholedf = pd.DataFrame(index=wholeindex, columns=['real', 'forecast'])
    for e in singledfs:
        wholedf.loc[e.index[:], 'real'] = e.loc[e.index[:], 'real'].to_numpy()
        wholedf.loc[e.index[:], 'forecast'] = e.loc[e.index[:], 'forecast'].to_numpy()
    
    evaldf = wholedf.iloc[182*24:]
    mae = np.abs((evaldf.real - evaldf.forecast).to_numpy()).mean()
    rmse = np.sqrt(((evaldf.real - evaldf.forecast).to_numpy()**2).mean())
    print(f'CAL LEN: {cl}')
    print(f'MAE: {mae}\t\tRMSE: {rmse}\tbased on {len(evaldf)/24} days')
    # wholedf.to_csv(f'lear_forecast_{cl}.csv')
    cl_dfs[cl] = wholedf

wholedf = cl_dfs[cal_lens[0]]
wholedf.columns = ['real', f'forecast{cal_lens[0]}']
for cl in cal_lens[1:]:
    temp = cl_dfs[cl]
    wholedf.loc[temp.index[:], f'forecast{cl}'] = temp.loc[temp.index[:], 'forecast'].to_numpy()
wholedf.loc[wholedf.index[:], 'forecast_averaged'] = wholedf.drop(columns='real').mean(axis='columns').to_numpy()
wholedf.to_csv(f'lear_forecast.csv')
wholedf = wholedf.iloc[182*24:]
mae = np.abs((wholedf.real - wholedf.forecast_averaged).to_numpy()).mean()
rmse = np.sqrt(((wholedf.real - wholedf.forecast_averaged).to_numpy()**2).mean())
print(f'ENSEMBLE OF 4 LENGTHS')
print(f'MAE: {mae}\t\tRMSE: {rmse}\tbased on {len(wholedf)/24} days')
