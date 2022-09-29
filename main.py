import pandas as pd
import numpy as np
import time
from envmt import preprocessing

path = "/home/jdelas/projects/def-fcuppens/jdelas/datasets/" #local data files
preprocessed = True #Is the data already preprocessed ?

if __name__ == "__main__":
    if not preprocessed :

        df_normal = pd.read_excel(path + "SWaT_Dataset_Normal_v1.xlsx", header=1)
        df_attacks = pd.read_excel(path + "SWaT_Dataset_Attack_v0.xlsx", header=1)

        # 
        df, df_train, df_test = preprocessing.basic(df_normal, df_attacks)

        # Select only LIT101
        anomaly_time_start = [np.array('2015-12-28T11:22:00', dtype=np.datetime64),
                            np.array('2015-12-29T18:30:00', dtype=np.datetime64),
                            np.array('2015-12-31T15:47:40', dtype=np.datetime64),
                            np.array('2016-01-01T22:16:01', dtype=np.datetime64)]
        anomaly_time_end = [np.array('2015-12-28T11:28:22', dtype=np.datetime64),
                            np.array('2015-12-29T18:42:00', dtype=np.datetime64),
                            np.array('2015-12-31T16:07:10', dtype=np.datetime64),
                            np.array('2016-01-01T22:25:00', dtype=np.datetime64)]

        df_train_LIT101, df_test_LIT101 = preprocessing.select(df, anomaly_time_start, anomaly_time_end)
    
    else :
        s = time.time()
        df_train = pd.read_pickle(path + 'df_train.pkl')
        df_test = pd.read_pickle(path + 'df_test.pkl')
        df_train_LIT101 = pd.read_pickle(path + 'df_train_LIT101.pkl')
        df_test_LIT101 = pd.read_pickle(path + 'df_test_LIT101.pkl')
        e = time.time()
        print("loading time : ", e-s)

    print(df_train.shape)