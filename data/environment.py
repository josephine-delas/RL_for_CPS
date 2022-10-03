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

class ReplayMemory(object):
    '''
    Basic replay memory
    '''
    def __init__(self, observation_size, batch_size, max_size):
        '''
        observation size  = (1, features)
        '''
        self.observation_size = observation_size
        self.num_observed = 0
        self.batch_size = batch_size
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros((self.max_size, observation_size[1])),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16),
                 'reward'   : np.zeros(self.max_size * 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16),
               }

    def observe(self, states, action, reward, done):
        # je pense qu'il faut changer pour les autres (actions, reward, etc.) et changer les arguments là ou la méthode est appelée
        index = self.num_observed % self.max_size
        if index + self.batch_size < self.max_size : 
          self.samples['obs'][index:index+self.batch_size] = states
          self.samples['action'][index:index+self.batch_size] = action
          self.samples['reward'][index:index+self.batch_size] = reward
          self.samples['terminal'][index:index+self.batch_size] = done
          self.num_observed += self.batch_size
        else :
          decalage = index + self.batch_size - self.max_size
          np.roll(self.samples['obs'], decalage, axis=0)
          self.samples['obs'][0:self.batch_size,:] = states
          np.roll(self.samples['action'], decalage, axis=0)
          self.samples['action'][0:self.batch_size] = action
          np.roll(self.samples['reward'], decalage, axis=0)
          self.samples['reward'][0:self.batch_size] = reward
          np.roll(self.samples['terminal'], decalage, axis=0)
          self.samples['terminal'][0:self.batch_size] = done   

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices], dtype=np.float32)
        # s.shape : (minibatch_size, features)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1], dtype=np.float32) 

        a      = self.samples['action'][sampled_indices]
        r      = self.samples['reward'][sampled_indices]
        done   = self.samples['terminal'][sampled_indices]

        return (s, a, r, s_next, done)

class RLenv(data_cls):
    '''
    Adversarial environment for Intrusion Detection
    '''
    def __init__(self,train_test, state_size, batch_size, iterations_episode, adversarial=False, attacker_agent = None, **kwargs):
        data_cls.__init__(self,train_test, state_size, **kwargs)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = batch_size
        self.iterations_episode = iterations_episode
        self.adversarial = adversarial
        self.attacker_agent = attacker_agent
        if self.adversarial:
          self.attack_df = self.df[self.df['Attack']==1][state_size: -state_size]
          self.normal_df = self.df[self.df['Attack']==0][state_size: -state_size]

    def _update_state(self):        
        '''
        Gets states and labels from df
        '''
        self.states,self.labels = data_cls.get_batch(self, self.batch_size)
        
        # Update statistics
        self.def_true_labels += np.sum(self.labels) #number of true attack labels

    def reset(self):
        '''
        Initializes stats and returns first state
        '''
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types),dtype=int) # [n_true_normal, n_true_attack]
        self.def_estimated_labels = np.zeros(len(self.attack_types),dtype=int) # [n_estimated_normal, n_estimated_attack]
        
        self.state_numb = 0
        
        data_cls.reset(self) # Reload and random index
        self.states,self.labels = data_cls.get_batch(self,batch_size=self.batch_size)
        
        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states
   
    def act(self, actions):
        '''
        Action taken without adversarial environment (without attacker agent)
        actions : shape (batch_size) -> like self.labels
        '''

        true_pos = actions*self.labels
        true_neg = (1-actions)*(1-self.labels)
        false_pos = actions * (1-self.labels)
        false_neg = (1-actions)*self.labels
        self.reward = 1*true_pos + 1*true_neg - 1*false_pos - 1*false_neg
        
        self.def_estimated_labels += np.bincount(actions, minlength=len(self.attack_types))
        self.def_true_labels += np.asarray([len(self.labels)-np.sum(self.labels), np.sum(self.labels)])

        if self.adversarial:
          att_reward = actions!=self.labels
          attack_actions = np.random.randint(2, size=(self.batch_size))#self.attacker_agent.act(self.states)
          self.states, self.labels = self.get_states(self.batch_size, attack_actions)
        
        else:
          att_reward = None
          attack_actions = None
          self.states, self.labels= self.get_batch(batch_size = self.batch_size)

        self.done=np.zeros(len(actions)) # Continuous task
        return self.states, self.reward, self.done, att_reward, attack_actions

    def get_states(self, batch_size, attacker_actions):
        '''
        Gets batch of states with the required label (the attacker agent chooses which labels to feed to the defender)
        '''
        batch = []
        labels = []

        for a in attacker_actions:
          if a : 
            index = self.attack_df.sample(1).index.values
          else :
            index = self.normal_df.sample(1).index.values
          offset = np.random.randint(self.state_size)
          state = self.df.loc[range(int(index)-offset, int(index)-offset+self.state_size)]
          label = state['Attack'].max()
          state = state.drop([' Timestamp', 'Attack', 'Normal'], axis=1)
          batch.append(state)
          labels.append(label)
        
        return np.array(batch), np.array(labels)