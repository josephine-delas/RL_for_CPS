from functools import total_ordering
import pandas as pd
import numpy as np
import time
import json
from data import preprocessing
from data.environment import RLenv
from agent.agent import Agent
from aux import train
from viz.visualisation import print_results

model_name = 'model'
Train = True
Test = False

path_data = "/home/jdelas/projects/def-fcuppens/jdelas/datasets/" #local data files
path_save_fig = "/home/jdelas/projects/def-fcuppens/jdelas/figures/"
path_save_models = "/home/jdelas/projects/def-fcuppens/jdelas/models/"
preprocessed = True #Is the data already preprocessed ?

'''
Hyperparameters
'''
# Training
batch_size = 128 # Train batch
minibatch_size = 128 # batch of memory ExpRep
ExpRep = True
state_size = 60 # Number of timesteps in each state
iterations_episode = 200 # Epoch length (number of iteration per episode)
adversarial = True
num_episodes = 20 # Train for 20 epochs
# Learning
epsilon = 1 # exploration
min_epsilon = 0.01 # min value for exploration
gamma = 0.001 # Very small gamma because no correlation beween states
decay_rate = 0.99 # Learning rate decay 
learning_rate = .2


if __name__ == "__main__":
    ####################
    # Load data 
    ####################
    print('Loading...')
    if not preprocessed :

        df_normal = pd.read_excel(path_data + "SWaT_Dataset_Normal_v1.xlsx", header=1)
        df_attacks = pd.read_excel(path_data + "SWaT_Dataset_Attack_v0.xlsx", header=1)

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
        df_train = pd.read_pickle(path_data + 'df_train.pkl')
        df_test = pd.read_pickle(path_data + 'df_test.pkl')
        df_train_LIT101 = pd.read_pickle(path_data + 'df_train_LIT101.pkl')
        df_test_LIT101 = pd.read_pickle(path_data + 'df_test_LIT101.pkl')
        e = time.time()
        print("Loading time : ", e-s)

    ####################
    # Initialization
    ####################
    print('Initialisation...')
    obs_size = (state_size, df_train_LIT101.shape[1]-2 - 1) # shape of the states (n_timesteps, features)

    agent = Agent(['Normal', 'Attack'], obs_size, batch_size, "EpsilonGreedy", 
              epoch_length = iterations_episode,
              epsilon = epsilon,
              min_epsilon = min_epsilon,
              decay_rate = decay_rate,
              gamma = gamma,
              minibatch_size = minibatch_size,
              mem_size = 1000,
              learning_rate=learning_rate,
              ExpRep=ExpRep)
    total_reward_chain = []

    attacker_agent = None
    if adversarial : 
        attacker_agent = Agent(['Normal', 'Attack'], obs_size, batch_size, "EpsilonGreedy", 
                    attacker=True,
                    epoch_length = iterations_episode,
                    epsilon = epsilon,
                    min_epsilon = min_epsilon,
                    decay_rate = decay_rate,
                    gamma = gamma,
                    minibatch_size = minibatch_size,
                    mem_size = 1000,
                    learning_rate=learning_rate,
                    ExpRep=ExpRep)
    
    env = RLenv('train', state_size, batch_size, iterations_episode, adversarial=adversarial, attacker_agent = attacker_agent)  
    
    ####################
    # Training
    ####################
    if Train : 
        print('Training...')
        print("-------------------------------------------------------------------------------")
        print("Total epoch: {} | Iterations in epoch: {}"
            "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                            iterations_episode,minibatch_size,
                            num_episodes*iterations_episode*batch_size))
        print("-------------------------------------------------------------------------------")
        print("Dataset shape: {}".format(env.data_shape))
        print("-------------------------------------------------------------------------------")
        print("Agent parameters: Num_actions={} | gamma={} | "
            "epsilon={} | Convolution units ={} |"
            " Kernel size={}|".format(2, gamma, epsilon, 32, 8))
        print("-------------------------------------------------------------------------------")

        reward_chain, loss_chain = train(agent, env, batch_size=batch_size, minibatch_size=minibatch_size,
                                        attacker_agent=attacker_agent, num_episodes=num_episodes, adversarial=adversarial,
                                        iterations_episode=iterations_episode, ExpRep=ExpRep)

    ####################
    # Saving
    ####################
        agent.model_network.model.save_weights(path_save_models + '/' + model_name + '.h5', overwrite=True)
        with open(path_save_models + '/' + model_name + '.json', "w") as outfile:
            json.dump(agent.model_network.model.to_json(), outfile)
        
        print_results(reward_chain, loss_chain, path_save_fig, 'training_' + model_name + '.pdf', max_reward = 0)

    ###################
    # Testing
    ###################
