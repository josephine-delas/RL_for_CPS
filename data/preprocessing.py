import pandas as pd
import numpy as np
import time 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import  confusion_matrix

def basic(df_normal, df_attacks, date_lim = '2016-01-01T00:00:00'):
    '''
    Performs basic preprocessing in order to train models on SWaT dataset

    Args : 
        - df_normal, df_attacks : raw datasets as downloaded on SWaT drive
        - date_lim : string of form YYYY-MM-DDThh:mm:ss. Data before the limit will be used for training, the rest for testing

    Returns : 
        - df, df_test, df_train : datasets trained and preprocessed
    '''

    # Creating one unique dataset
    df_normal.columns = df_attacks.columns
    df = pd.concat([df_normal, df_attacks])
    df.reset_index(inplace=True)
    df.drop('index', axis=1, inplace=True)

    # Changing boolean columns (1-2 values become 0-1 values)
    df_to_modify_1 = df.select_dtypes(include='int64')
    columns_to_modify_1 = df_to_modify_1.columns
    df[columns_to_modify_1] -=1

    # Min/Max normalization for float64 columns
    df_to_modify_2 = df.select_dtypes(include='float64')
    columns_to_modify_2 = df_to_modify_2.columns
    scaler = MinMaxScaler()
    df[columns_to_modify_2] = pd.DataFrame(scaler.fit_transform(df_to_modify_2))

    # Formatting the label column
    df['Attack'] = df['Normal/Attack'].map({'A ttack':1, 'Attack':1, 'Normal':0})
    df['Normal'] = df['Normal/Attack'].map({'A ttack':0, 'Attack':0, 'Normal':1})
    df.drop('Normal/Attack', axis=1, inplace=True)

    # Formatting timestamp
    df[' Timestamp'] = pd.to_datetime(df[' Timestamp'])
    df_test = df[df[' Timestamp'] >= np.array(date_lim, dtype=np.datetime64)]
    df_train = df[df[' Timestamp'] < np.array(date_lim, dtype=np.datetime64)]

    return(df, df_test, df_train)

def select(df, anomaly_time_start, anomaly_time_end, column='LIT101', time_start = np.array('2015-12-28T00:00:00', dtype=np.datetime64), time_sep = np.array('2016-01-01T00:00:00', dtype=np.datetime64)):
    '''
    Selects only one columns from the SWaT dataset, reajusting the labels.
    
    Args : 
        - df : the full SWaT preprocessed dataframe
        - anomaly_time_start, anomaly_time_end : arrays of np.datetimes delimiting the attacks
        - column : string 
        - time_start, time_sep : np.datetime64 ; beginning of the data, separation between train and test

    Returns : 
        - df_train_LIT101, df_test_LIT101 : train and test datasets for the wanted column
    '''
    df_LIT101 = df[[' Timestamp', column, 'Attack', 'Normal']]

    # Adjusting the labels
    df_LIT101.loc[:,'Normal']=1
    df_LIT101.loc[:,'Attack']=0
    for i in range(len(anomaly_time_start)):
        mask = (df_LIT101[' Timestamp'] >= anomaly_time_start[i]) & (df_LIT101[' Timestamp'] <= anomaly_time_end[i])
        df_LIT101.loc[mask, 'Normal']=0
        df_LIT101.loc[mask, 'Attack']=1
    
    #Splitting dataset
    df_train_LIT101 = df_LIT101[df_LIT101[' Timestamp'] < time_sep]
    df_train_LIT101 = df_train_LIT101[df_LIT101[' Timestamp'] >=  time_start]
    df_test_LIT101 = df_LIT101[df_LIT101[' Timestamp'] >= time_sep]

    return(df_LIT101, df_train_LIT101, df_test_LIT101)

if __name__ == "__main__":
    '''
    Save preprocessed dataset for faster execution
    '''

    # Load raw data
    a = time.time()
    path = "/home/jdelas/projects/def-fcuppens/jdelas/datasets/"
    df_normal = pd.read_excel(path + "SWaT_Dataset_Normal_v1.xlsx", header=1)
    df_attacks = pd.read_excel(path + "SWaT_Dataset_Attack_v0.xlsx", header=1)
    b = time.time()
    print("Loading time : ", b-a)
    
    # preprocess dataframes
    df, df_test, df_train = basic(df_normal, df_attacks)

    # Select only LIT101
    anomaly_time_start = [np.array('2015-12-28T11:22:00', dtype=np.datetime64),
                        np.array('2015-12-29T18:30:00', dtype=np.datetime64),
                        np.array('2015-12-31T15:47:40', dtype=np.datetime64),
                        np.array('2016-01-01T22:16:01', dtype=np.datetime64)]
    anomaly_time_end = [np.array('2015-12-28T11:28:22', dtype=np.datetime64),
                        np.array('2015-12-29T18:42:00', dtype=np.datetime64),
                        np.array('2015-12-31T16:07:10', dtype=np.datetime64),
                        np.array('2016-01-01T22:25:00', dtype=np.datetime64)]

    df_LIT101, df_train_LIT101, df_test_LIT101 = select(df, anomaly_time_start, anomaly_time_end)
    
    df.to_pickle(path + 'df.pkl')
    df_train.to_pickle(path + 'df_train.pkl')
    df_test.to_pickle(path + 'df_test.pkl')
    df_LIT101.to_pickle(path + 'df_LIT101.plk')
    df_train_LIT101.to_pickle(path + 'df_train_LIT101.pkl')
    df_test_LIT101.to_pickle(path + 'df_test_LIT101.pkl')
