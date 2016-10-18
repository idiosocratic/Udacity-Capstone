# nearest neighbor state-space discretizing implementation


import numpy as np
import gym
import random


class NN_Archetype_Agent(object):

    def __init__(self, action_space):
    
        
        # hyperparameters
        self.epsilon = 1.0  # exploration percentage
        self.epsilon_decay = 0.90 # exploration decay
        self.iteration = 0 # how many actions have we taken
        self.time_before_exploit = 137 # how much knowledge to build before using it
        self.archetype_distance = 0.17  # how close can two observations be before we consider them the same
    
        # memory for our (state, action, value) tuples, aka our archetypes
        self.state_action_value_memory = []
    

    """
       This function goes through every archetype memory and checks to see if the current state is in range
       of any present. If not, the state is added as a new archetype, but if there is already an archetype in 
       range the two states values are compared and the one with the higher value is kept in memory.
    """
    def add_to_mem_if_needed(self, state_action_value):
        
        archetype_present = False
        
        archetype_2_replace = None
        
        current_state = state_action_value[0]
        
        current_value = state_action_value[2]
        
        for index, s_a_v in enumerate(self.state_action_value_memory):
        
            if not archetype_present:  # search memory for a archetype fit
        
                if self.get_L2_distance(current_state, s_a_v[0]) < self.archetype_distance:
    
                    archetype_present = True  # found fitting archetype
    
                    if current_value > s_a_v[2]:
    
                        archetype_2_replace = index  # index of lower value archetype
    
    
        if not archetype_present:  # we don't have this archetype yet, add it
        
            self.state_action_value_memory.append(state_action_value)
        
        if not archetype_2_replace == None:  # we need to replace an old archetype for a better one

            pruned_memory = self.state_action_value_memory.pop(archetype_2_replace)
    
            self.state_action_value_memory.append(state_action_value)
    
    """
       This function goes through every archetype in memory to find the one closest to the given state. It then
       returns the action taken by that archetype as the action to take.
    """
    def get_best_action(self, state):
    
        closest_distance = 1e4  # initialize big distance
        
        best_action = None
    
        for state_action_value in self.state_action_value_memory:
    
            dist = self.get_L2_distance(state, state_action_value[0])
    
            if dist < closest_distance:  # use action from closest archetype

                closest_distance = dist
            
                best_action = state_action_value[1]
            
        assert not best_action == None
        
        return best_action



    def get_L2_distance(self, state1, state2):
        
        return np.linalg.norm(state1-state2)  # L2 distance function provided by numpy
        
        
        
    """
       This function checks the iteration number against the time_before_exploit parameter to let us know if we
       are ready to start exploiting our archetype memory.
    """
    def should_we_exploit(self):    
    
        if self.iteration > self.time_before_exploit: 
            
            return True  # we have enough knowledge
    
        return False  # not enough knowledge to exploit 
    
    
        
    def decay_epsilon(self):
        
        """ 
            we will continue to see new state-spaces so while the need to explore may taper off, we
            will need to maintain some level of exploration
        """
        if self.epsilon > 0.1:

            self.epsilon *= self.epsilon_decay  # decay epsilon
  
        

env = gym.make('CartPole-v0')
wondering_gnome = NN_Archetype_Agent(env.action_space)
            
episode_rewards_list = []            
            
for i_episode in xrange(1000):
    observation = env.reset()
    
    episode_rewards = 0
    episode_state_action_list = []
    episode_state_action_value_list = []
    
    for t in xrange(200):
        #env.render()
        
        current_state = observation  
        
        action = env.action_space.sample() # initialize action randomly
        
        # should we override action 
        if wondering_gnome.should_we_exploit():
            
            random_fate = np.random.random()  # get random number in range (0, 1)
             
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
    print "Running average of last 100 episodes: " + str(np.average(episode_rewards_list[-100:]))


    discounted_value = episode_rewards

    for state_action in episode_state_action_list:
        
        episode_state_action_value_list.append((state_action[0], state_action[1], discounted_value))

        discounted_value -= 1

    print "Size of Mem: " + str(len(wondering_gnome.state_action_value_memory))


    for state_action_value in episode_state_action_value_list:

        wondering_gnome.add_to_mem_if_needed(state_action_value)


    if wondering_gnome.should_we_exploit():

        wondering_gnome.decay_epsilon()


if np.average(episode_rewards_list[-100:]) >= 195:

    print "Solved!"
    assert False
