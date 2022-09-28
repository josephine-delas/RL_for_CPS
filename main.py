import pandas as pd
import numpy as np
from envmt import preprocessing

# Import data from local files
path = "/home/jdelas/projects/def-fcuppens/jdelas/datasets/"
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

print(df.shape)