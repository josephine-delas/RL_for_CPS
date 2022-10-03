import numpy as np
from data.environment import ReplayMemory
from agent.dqn import QNetwork, SimpleNetwork
from agent.policy import Epsilon_greedy


class Agent(object):  
    '''
    Double Deep Q-Network agent (target network and )
    '''
        
    def __init__(self, actions,obs_size, batch_size, policy="EpsilonGreedy", attacker = False, **kwargs):
        '''
        obs_size : (timesteps, features)
        '''
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        self.attacker = attacker
        
        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001) # Très faible gamma car pas de corrélation
        self.minibatch_size = kwargs.get('minibatch_size', 32) # Pourquoi pas plus ? 
        self.epoch_length = kwargs.get('epoch_length', 10000)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        self.ExpRep = kwargs.get('ExpRep',True) # Experience replay
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, batch_size, kwargs.get('mem_size', 100))
        
        self.ddqn_time = 100 #update target network every 100 steps
        self.ddqn_update = self.ddqn_time

        if attacker : 
            self.model_network = SimpleNetwork(self.obs_size, learning_rate = kwargs.get('learning_rate',.2))
            self.target_model_network = SimpleNetwork(self.obs_size, learning_rate = kwargs.get('learning_rate',.2))
            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        else : 
            self.model_network = QNetwork(self.obs_size, learning_rate = kwargs.get('learning_rate',.2))
            self.target_model_network = QNetwork(self.obs_size, learning_rate = kwargs.get('learning_rate',.2))
            self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network,len(actions),
                                         self.epsilon,self.min_epsilon,
                                         self.decay_rate,self.epoch_length)
        
        
    def learn(self, states, actions,next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done        

    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
            # Each is an array of shape (minibatch_size, ..)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
        
        #if self.attacker :
        #  states = states[:,0,:]
        #  next_states = next_states[:,0,:]
        # Compute Q targets
        Q_prime = self.target_model_network.predict(next_states,self.minibatch_size)
        next_actions = np.argmax(Q_prime, axis=1)
        sx = np.arange(len(next_actions))

        # Compute Q(s,a)
        Q = self.model_network.predict(states,self.minibatch_size)
        # Q-learning update
        # target = reward + gamma * max_a'{Q(next_state,next_action))}
        targets = rewards.reshape(Q[sx,actions].shape) + \
                  self.gamma * Q_prime[sx,next_actions] * \
                  (1-done.reshape(Q[sx,actions].shape))   
        Q[sx,actions] = targets  
        
        if self.attacker : 
          states = states[:,0,:]
        loss = self.model_network.model.train_on_batch(states,Q)#inputs,targets        
        
        # timer to ddqn update
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            # self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
            self.target_model_network.model.set_weights(self.model_network.model.get_weights()) 
        
        return loss    

    def act(self, states):
        '''
        Gets actions from states
        '''
        actions = self.policy.get_actions(states)
        return actions