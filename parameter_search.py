# a parameter search agent
import gym
import numpy as np

"""
   This agent was inspired by Kevin Frans blog: http://kvfrans.com/simple-algoritms-for-solving-cartpole/
   There he introduces the concept as well as the lines used to generate random parameters and multiply them 
   with the environments inputs.
   
   I've used the following pieces of code in homeage, and because they are very concise:
   
   parameters = np.random.rand(4) * 2 - 1
   
   action = 0 if np.matmul(parameters,observation) < 0 else 1
   
   His implementation does not, however, solve Cartpole by OpenAI's standards and the reason is because, as
   discussed in the report, the function needed is not a linear function. Such a solution only approximates one
   area of the state space and thus they must be combined until they saturate the space.
"""


class ParameterSearchAgent(object):
    
    def __init__(self, action_space):
        
        # hyperparameters
        self.epsilon = 0.6 # exploration rate
        self.have_parameters = False  # do we have parameters yet
        self.pruning_interval = 10  # interval dictating how often to prune low scoring parameters
        self.pruning_threshold = 190 # threshold for pruning lower scoring parameters every few episodes
        
        # memory
        """
            list of (initial observation, parameters, average score) tuples. 
            average score has form (average score, episodes used)
        """
        self.parameters_avg_score_memory = []


    """
       This function goes through our stored parameters and returns those who's initial state is closest to our 
       own. It also returns the index of those parameters from our memory list so that we can update the 
       average score of the parameters after using them.
    """
    def get_best_parameters_for_state(self, state):
    
        closest_dist = 1e4  # initialize large distance
    
        best_parameters = None
        
        pararmeters_index = None
    
        for index, parameters in enumerate(self.parameters_avg_score_memory):
        
            stored_state = parameters[0]
            
            current_parameters = parameters[1]
        
            dist = np.linalg.norm(state-stored_state)  # get L2 distance

            if dist < closest_dist:

                closest_dist = dist

                best_parameters = current_parameters

                pararmeters_index = index


        assert not best_parameters == None
    
        return best_parameters, pararmeters_index


    """
        This function adds a new set of parameters to our memory.
    """
    def add_to_parameters_memory(self, initial_state_parameters_score):

        initial_state = initial_state_parameters_score[0]
        
        parameters = initial_state_parameters_score[1]
        
        average_score = (initial_state_parameters_score[2], 1) # initialize score with one episode of experience

        new_parameters = (initial_state, parameters, average_score)

        self.parameters_avg_score_memory.append(new_parameters)


    """
        This function updates the average score for the parameters whose index is given using the newest episode
        rewards, the old average, and the number of times the parameters have been used.
    """
    def update_avg_parameters_score(self, new_score, pararmeters_index):

        current_score = self.parameters_avg_score_memory[pararmeters_index][2][0]

        episodes_seen = self.parameters_avg_score_memory[pararmeters_index][2][1] + 1

        updated_average = ( (current_score * (episodes_seen - 1)) + new_score ) / episodes_seen

        entry_state = self.parameters_avg_score_memory[pararmeters_index][0]

        entry_parameters = self.parameters_avg_score_memory[pararmeters_index][1]

        pop_old_memory = self.parameters_avg_score_memory.pop(pararmeters_index)
            
        new_entry = (entry_state, entry_parameters, (updated_average, episodes_seen))
            
        self.parameters_avg_score_memory.append(new_entry)


    """
       This function prunes the parameters whose average score have fallen below the pruning_threshold parameter
       value.
    """
    def prune_memory(self):

        pruning_list = []
        
        print "Length of Memory: " + str(len(self.parameters_avg_score_memory))
        print self.parameters_avg_score_memory

        for index, parameters in enumerate(self.parameters_avg_score_memory):

            average_score = parameters[2][0]

            if average_score < self.pruning_threshold:

                pruning_list.append(index)
                    
                print "Pruning: " + str(parameters)


        for leaf in sorted(pruning_list, reverse = True):

            cut = self.parameters_avg_score_memory.pop(leaf)






episode_rewards_list = []  # list of episode rewards from when we're exploiting our knowledge

solved = False  # have we possibly solved the environment
solved_counter = 100  # counter to test if solved

env = gym.make('CartPole-v0')
wondering_gnome = ParameterSearchAgent(env.action_space)

for i_episode in xrange(1000):

    observation = env.reset()

    initial_observation = observation  # keep track of starting state
    
    episode_rewards = 0          # keep track of episode rewards


    if not solved:

        exploring = True
    
    
    if wondering_gnome.have_parameters: # don't start exploiting until we have some good parameters
    
        random_fate = np.random.random()
    
        if random_fate > wondering_gnome.epsilon:  # epsilon greedy implementation

            exploring = False


    parameters = None
    pararmeters_index = None  # if using parameters from memory

    if exploring: # get random parameters, this form allows us to randomly generate parameters between -1 and 1

        parameters = np.random.rand(4) * 2 - 1


    if not exploring:
                
        parameters, pararmeters_index = wondering_gnome.get_best_parameters_for_state(initial_observation)



    for t in xrange(200):
        #env.render()
       
        # pick action
        action = 0 if np.matmul(parameters,observation) < 0 else 1


        observation, reward, done, info = env.step(action) # act in gym environment and get reward
    
        episode_rewards += reward
        
        if done:
            
            print "Episode finished after {} timesteps".format(t+1)
            break
            
    print "Episode " + str(i_episode) + " rewards: " + str(episode_rewards)
    print "Exploring: " + str(exploring)
    print "Epsilon:" + str(wondering_gnome.epsilon)
    print "Running Average of Last 100 Episodes: " + str(np.average(episode_rewards_list[-100:]))
    print "Number of episodes in mem: " + str(len(episode_rewards_list))

    if not exploring:  # interested only in how well we are doing when not exploring

        episode_rewards_list.append(episode_rewards)


    if (episode_rewards > 197) & exploring:
      
        parameter_entry = (initial_observation, parameters, episode_rewards)
      
        wondering_gnome.add_to_parameters_memory(parameter_entry) # add good parameters to memory
     
        wondering_gnome.have_parameters = True


    if not pararmeters_index == None:

        wondering_gnome.update_avg_parameters_score(episode_rewards, pararmeters_index)


    if i_episode % wondering_gnome.pruning_interval == 0:

        print "Pruning Memory..."
        wondering_gnome.prune_memory()
    
        if len(wondering_gnome.parameters_avg_score_memory) == 0:
    
            wondering_gnome.have_parameters = False


    if (len(episode_rewards_list) > 30) and (np.average(episode_rewards_list[-30:]) > 195):

        print "Possibly solved, turning off exploration."
        print "Countdown = " + str(solved_counter)
        
        solved = True
        solved_counter -= 1

        if solved_counter == 0:
            
            if np.average(episode_rewards_list[-100:]) > 195:

                print "SOLVED! SOLVED! SOLVED! SOLVED! SOLVED! SOLVED! SOLVED! SOLVED! SOLVED! SOLVED!"
                assert False

            else:

                print "Environment not solved, retrying..."
                solved = False



