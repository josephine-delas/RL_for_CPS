import numpy as np
import pandas as pd


class data_cls:
    '''
    In this case we want batches of batch_size states, each state is the conscatenation 
    of self.state_size (30) consecutive observations. The states should not be 
    taken consecutively, so use a shuffle of the index array.
    The labels for each state are calculated as follow : Anomaly if at least 
    one of the state's observations is an attack, else Normal
    '''
    def __init__(self,train_test, state_size, df_train, df_test, **kwargs):
        self.index = 0
        self.train_test = train_test
        self.train_df = df_train
        self.test_df = df_test
        self.state_size = state_size

        if self.train_test == 'train':
            self.df = self.train_df
        else:
            self.df = self.test_df
        
        # List of indexes for batch sampling 
        self.index_list = np.arange(self.df.shape[0]-state_size+1)
        np.random.shuffle(self.index_list)

        self.attack_types = ['Normal','Attack']
        formated = True  

    def reset(self):
        '''
        Reset index list
        '''
        self.index_list = np.arange(self.df.shape[0]-self.state_size+1)
        np.random.shuffle(self.index_list)
            
    def get_shape(self):
        self.data_shape = self.df.shape
        return self.data_shape
    
    def get_batch(self, batch_size=32):
        '''
        Batches are not always of the same size ! The last one might be smaller
        batch shape : (batch_size, state_size, n_features)
        labels shape : (batch_size)
        '''
        end = min(self.index+batch_size, self.index_list.size) # In case the last batch overrides the end of the index list
        indexes = self.index_list[self.index:end]   

        self.index = end if (end<self.index_list.size) else 0

        batch = []
        labels = []
        for i in indexes:
          state = self.df.iloc[range(i,i+self.state_size)]
          label = state['Attack'].max()
          state = state.drop([' Timestamp', 'Attack', 'Normal'], axis=1)
          batch.append(state)
          labels.append(label)
        
        return np.array(batch), np.array(labels)
    
    def get_full(self):
        labels=[]
        batch=[]

        for i in range(len(self.df.index)-self.state_size+1):
          state = self.df.iloc[range(i,i+self.state_size)]
          label = state['Attack'].max()
          state = state.drop([' Timestamp', 'Attack', 'Normal'], axis=1)
          batch.append(state)
          labels.append(label)
        return np.array(batch),np.array(labels)