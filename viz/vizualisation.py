import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def print_anomalies(df, path_save, anomaly_time_start, anomaly_time_end):
    fig, ax = plt.subplots(len(anomaly_time_end), figsize=(15,10))
    length = np.timedelta64(4, 'h')

    for i in range(len(anomaly_time_start)):
        start = anomaly_time_start[i] - length
        end = anomaly_time_start[i] + length
        mask = (df[' Timestamp']>start) & (df[' Timestamp']<end)
        x = df.loc[mask, ' Timestamp']
        y = df.loc[mask, 'LIT101']
        ax[i].plot(x,y)
        ax[i].axvspan(anomaly_time_start[i], anomaly_time_end[i], color='#FEDCC2')

    plt.savefig(path_save + 'LIT101_anomalies.svg', format='svg', dpi=1000)

path_save = "/home/jdelas/projects/def-fcuppens/jdelas/figures/"
path = "/home/jdelas/projects/def-fcuppens/jdelas/datasets/"

anomaly_time_start = [np.array('2015-12-28T11:22:00', dtype=np.datetime64),
                        np.array('2015-12-29T18:30:00', dtype=np.datetime64),
                        np.array('2015-12-31T15:47:40', dtype=np.datetime64),
                        np.array('2016-01-01T22:16:01', dtype=np.datetime64)]
anomaly_time_end = [np.array('2015-12-28T11:28:22', dtype=np.datetime64),
                        np.array('2015-12-29T18:42:00', dtype=np.datetime64),
                        np.array('2015-12-31T16:07:10', dtype=np.datetime64),
                        np.array('2016-01-01T22:25:00', dtype=np.datetime64)]

if __name__ == "__main__":
    df_LIT101 = pd.read_pickle(path + 'df_LIT101.pkl')
    print('start')
    print_anomalies(df_LIT101, path_save, anomaly_time_start, anomaly_time_end)
    print('end')

    

