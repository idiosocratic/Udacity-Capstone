# nearest neighbor majority vote implementation


import numpy as np
import gym
import random


class NnSelcMemAgent(object):

    def __init__(self, action_space):
    
        self.action_space = action_space
        assert isinstance(action_space, gym.spaces.discrete.Discrete), 'Yo, not our space!'
        
        # hyperparameters
        self.epsilon = 0.87  # exploration percentage
        self.epsilon_decay = 0.98 # exploration decay
        self.number_of_neighbors = 7 # number of closest states to vote on our actions
        self.highest_episode_rewards = 0  # keep record of highest episode, to decide what memories to keep
        self.did_we_do_well_threshold = 0.77 # percentage of highest score considered doing well
        self.iteration = 0 # how many actions have we taken
        self.time_before_exploit = 337 # how much knowledge to build before using it
        self.max_memory = 4e4  # maximum memory to retain
        
        
        self.state_action_rewards_memory = [] # selective memory for our (state,action) tuples

    
    """
       This function compares the episode rewards we received with our did_we_do_well_threshold parameter and
       returns a bool letting us know if the threshold was met.
    """
    def did_we_do_well(self, episode_rewards):
    
        if episode_rewards > self.highest_episode_rewards * self.did_we_do_well_threshold:
            
                return True
        
        return False
            
            
    """
       This function adds a (state, action, episode rewards) tuple to our memory.
    """
    def add_to_mem(self, episode_state_action_rewards_list):
        
        for s_a_r in episode_state_action_rewards_list:
        
            self.state_action_rewards_memory.append(s_a_r)
    
        
    """
       This function uses the helper function get_closest_states to get the closest number of neighbors specified
       by the number_of_neighbors parameter, and takes a majority vote of their actions.
    """
    def get_best_action(self, state):
    
        nearest_s_a_r_tuples = self.get_closest_states(state)
        
        action_list = [] # get the actions from similar states
        
        for s_a_r in nearest_s_a_r_tuples:
        
            action_list.append(s_a_r[1])
                    
        averaged_action = np.average(action_list) # get majority vote           
                    
        action = None # initialize
            
        if averaged_action >= 0.5: # round
            
            action = 1 
                
        if averaged_action < 0.5: # round
            
            action = 0    
                
        assert not (action == None) # assert have action
        
        return action
            
    
    # Get the L2 distance using Numpy's built-in function.
    def get_L2_distance(self, state1, state2):
            
        return np.linalg.norm(state1-state2)
        
        
    """
       This function goes through all the states in our memory and returns the amount of closet neighbors 
       specified by the number_of_neighbors parameter.
    """
    def get_closest_states(self, state):  
        
        sorted_s_a_r_tuples = []
        
        for s_a_r in self.state_action_rewards_memory:
        
            dist = self.get_L2_distance(s_a_r[0], state)
            
            sorted_s_a_r_tuples.append((dist,s_a_r))
    
        sorted_s_a_r_tuples = sorted(sorted_s_a_r_tuples, key = lambda x: x[0])  # sort by closest distance
        
        nearest_s_a_r_tuples = []
        
        for i in range(self.number_of_neighbors):  # how many neighbors are we using
        
            nearest_s_a_r_tuples.append(sorted_s_a_r_tuples[i][1])  # only keeping the state-action pairs
        
        return nearest_s_a_r_tuples
              
        
    """
       This function uses the iteration parameter to let us know if we've reached the time_before_exploit 
       threshold.
    """
    def should_we_exploit(self):    
    
        if self.iteration > self.time_before_exploit: 
            
            return True  # we have enough knowledge
    
        return False  # not enough knowledge to exploit 
    
    
    # Function for decaying our exploration rate.
    def decay_epsilon(self):
    
        self.epsilon *= self.epsilon_decay
        
        
    # Function for updating our highest acheived rewards.
    def update_highest_reward(self,episode_rewards):  
    
        if episode_rewards > self.highest_episode_rewards:
        
            self.highest_episode_rewards = episode_rewards


    """
       This function goes through our memory and prunes old (state, action, episode rewards) tuples were the 
       episode rewards no longer fit our definition of having done well. Because our highest reward is being 
       updated, the threshold for having done well changes over time.
    """
    def prune_memory(self):

        pruning_list = []
        
        for index, memory in enumerate(self.state_action_rewards_memory):

            if memory[2] < (self.highest_episode_rewards * self.did_we_do_well_threshold):

                pruning_list.append(index)


        pruning_list = sorted(pruning_list, reverse = True)
        

        for leaf in pruning_list:

            cut = self.state_action_rewards_memory.pop(leaf)
        
    
  
        

env = gym.make('CartPole-v0')
wondering_gnome = NnSelcMemAgent(env.action_space)        
            
episode_rewards_list = []            
            
for i_episode in xrange(1000):
    observation = env.reset()
    
    episode_rewards = 0
    episode_state_list = []
    episode_state_action_list = []
    episode_state_action_rewards_list = []
    
    for t in xrange(200):
        #env.render()
        
        current_state = observation  
        
        action = env.action_space.sample() # initialize action randomly
        
        # should we override action 
        if wondering_gnome.should_we_exploit():
            
            random_fate = np.random.random()
             
            if random_fate > wondering_gnome.epsilon: # epsilon greedy implementation
           
                action = wondering_gnome.get_best_action(current_state) # overwrite random action
                    
        
        observation, reward, done, info = env.step(action)
        
        episode_rewards += reward
        
        wondering_gnome.iteration += 1
        
        episode_state_action_list.append((current_state, action))
        
 
        if done:
            
            print "Episode finished after {} timesteps".format(t+1)
            break

    print "Episode: " + str(i_episode) + ", Rewards: " + str(episode_rewards) + ", Epsilon: " + str(wondering_gnome.epsilon)
    episode_rewards_list.append(episode_rewards)
    print "running average: " + str(np.average(episode_rewards_list[-100:]))

    wondering_gnome.recent_avg = np.average(episode_rewards_list[-100:])

    wondering_gnome.update_highest_reward(episode_rewards)


    for state_action in episode_state_action_list:

        episode_state_action_rewards_list.append((state_action[0], state_action[1], episode_rewards))
    
    
    if wondering_gnome.did_we_do_well(episode_rewards):
       
        if len(wondering_gnome.state_action_rewards_memory) < wondering_gnome.max_memory:

            wondering_gnome.add_to_mem(episode_state_action_rewards_list)


    if wondering_gnome.should_we_exploit():

        wondering_gnome.decay_epsilon()  # decay epsilon only once we start exploiting


    if i_episode % 10 == 0:

        wondering_gnome.prune_memory()

if np.average(episode_rewards_list[-100:]) >= 195:

    print "Solved!"
    assert False


