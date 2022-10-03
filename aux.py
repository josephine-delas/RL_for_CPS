import time
import numpy as np

def train(agent, env, batch_size = 32, minibatch_size = 32, attacker_agent = None, num_episodes=1, adversarial = False, iterations_episode = 100, ExpRep = False):
    reward_chain, loss_chain = []

    for epoch in range(num_episodes):
        start_time = time.time()
        loss = 0.
        total_reward_by_episode = 0
        tp, fp, tn, fn = 0 

        # Reset enviroment (random order for the states)
        states = env.reset()
        done = False
        if adversarial :
            att_actions = attacker_agent.act(states)
            states, labels = env.get_states(batch_size, att_actions)
    
        # Steps in one epoch
        for i_iteration in (range(iterations_episode)):
            act_time = time.time()  
            actions = agent.act(states) # shape(batch_size)

            # Statistics
            tp += (actions*env.labels).sum()
            tn += ((1-actions)*(1-env.labels)).sum()
            fp += (actions * (1-env.labels)).sum()
            fn += ((1-actions)*env.labels).sum()

            next_states,rewards, done, att_reward, next_att_actions= env.act(actions)
            agent.learn(states, actions, next_states, rewards, done)

            if adversarial:
                attacker_agent.learn(states, att_actions, next_states, att_reward, done)

            act_end_time = time.time()
            
            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch*iterations_episode + i_iteration >= minibatch_size:
                loss += agent.update_model()
            elif not ExpRep:
                loss += agent.update_model()

            update_end_time = time.time()

            # Updates
            states = next_states
            att_actions = next_att_actions
            total_reward_by_episode += np.sum(rewards,dtype=np.int32)

        # Update user view
        reward_chain.append(total_reward_by_episode) 
        loss_chain.append(loss)

        end_time = time.time()
        print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
                "|Loss {:4.4f} | Reward in ep {:03d}/{:03d}|"
                .format(epoch+1, num_episodes,(end_time-start_time), 
                loss,total_reward_by_episode,batch_size*iterations_episode))
        print("|TP {:04d} | TN {:04d} | FP {:04d}| FN {:04d}|".format(tp, tn, fp, fn))
        print("|Def Estimated: {}| True Labels: {}".format(env.def_estimated_labels,
            env.def_true_labels))
    
    return reward_chain, loss_chain
