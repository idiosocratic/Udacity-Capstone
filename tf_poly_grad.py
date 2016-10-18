
#### Libraries
# Standard library
import random
from collections import deque

# Third-party libraries
import numpy as np
import tensorflow as tf
import gym


class PolyGradAgent(object):
    
    def __init__(self, action_space):
        
        ### model hyperparameters

        self.epsilon = 0.9  # how much do we explore initially
        #        self.epsilon_decay_rate = 0.99  # rate by which exploration decreases
        self.high_score = 0  # keep track of highest score obtained thus far
        self.did_well_threshold = 0.77  # how close we need to be to our high score to have "done well"
        self.net_trained = False  # has our neural net had any training

        ### finite state_action memory for episodes where we did well
        self.memory_len = 10000
        self.state_action_mem = deque(maxlen = self.memory_len)
            
        self.last_100_episode_scores = deque(maxlen = 100) # keep track of average score from last 100 episodes


    """
       Function for getting the next training batch whose size is determined by the input variable.
    """
    def get_next_batch(self, batch_size):
            
        if len(self.state_action_mem) < batch_size:
        
            assert False, "Not enough memory for batch size!"
            
        rand_batch = random.sample(self.state_action_mem, batch_size)  # get random sample
    
        _states = []
        _actions = []
    
        for state_action in rand_batch:  # format random sample
        
            _states.append(state_action[0])
            _actions.append(state_action[1])
    
        batch_lists = (_states, _actions)

        return batch_lists


    """
       Function for letting us know if we did well based on the rewards received this episode and the 
       did_well_threshold parameter.
    """
    def did_we_do_well(self, episode_rewards):

        if episode_rewards > self.did_well_threshold * self.high_score:

            return True

        return False


    """
       Function for adding state-action pairs from episodes were we've "done well". The function translates
       the actions into the one-hot encoding needed to train the network.
    """
    def add_to_memory(self, episode_state_action_list):

        ## only use next line when opting for fast convergence epsilon strategy
        if not (len(self.state_action_mem) == self.memory_len):

            for state_action in episode_state_action_list:

                _state = state_action[0]
                _action = None  # initialize action
            
                if state_action[1] == 1:  # format one-hot action vector
            
                    _action = [0, 1]
            
                if state_action[1] == 0:  # format one-hot action vector
            
                    _action = [1, 0]

                self.state_action_mem.append((_state, _action))


    """
        Function that lets us know if our network has had any training.
    """
    def are_we_exploiting(self):  # do we have enough memory to start training our neural net

        if not self.net_trained:

            return False

        return True


    """
        Function that updates our highest score acheived thus far.
    """
    def update_high_score(self, episode_rewards):

        if episode_rewards > self.high_score:

            self.high_score = episode_rewards


    """
        Customized epsilon decay function with two different strategies. One must always be commented out.
    """
    def decay_epsilon(self):
    
    ## decaying epsilon strategy for fast solution convergence, average solution ~700 episodes
    ## fails to converge 1/8 episode from bad initialization
    
        if len(self.state_action_mem) > 0:
            self.epsilon = 0.9
        if len(self.state_action_mem) > 0.05 * self.memory_len:
            self.epsilon = 0.8
        if len(self.state_action_mem) > 0.1 * self.memory_len:
            self.epsilon = 0.7
        if len(self.state_action_mem) > 0.2 * self.memory_len:
            self.epsilon = 0.6
        if len(self.state_action_mem) > 0.3 * self.memory_len:
            self.epsilon = 0.5
        if len(self.state_action_mem) > 0.4 * self.memory_len:
            self.epsilon = 0.4
        if len(self.state_action_mem) > 0.5 * self.memory_len:
            self.epsilon = 0.3
        if len(self.state_action_mem) > 0.6 * self.memory_len:
            self.epsilon = 0.2
        if len(self.state_action_mem) > 0.7 * self.memory_len:
            self.epsilon = 0.1
    
    
    
    ## decaying epsilon strategy for eventual solution convergence, average solution ~6000 episodes

#        if self.high_score > 50:
#            self.epsilon = 0.5
#        if self.high_score > 100:
#            self.epsilon = 0.4
#        if self.high_score > 150:
#            self.epsilon = 0.3
#        if self.high_score == 200:
#            self.epsilon = 0.1

    



### set up tensorflow neural net

def weight_variable(shape):  # function that returns a network weight variable
    
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):  # function that returns a network bias variable
    
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


sess = tf.InteractiveSession()  # initialize tensorflow session


state = tf.placeholder(tf.float32,[None,4])  # placeholder for current state
actions = tf.placeholder(tf.float32,[None,2])  # placeholder for our actions
w1 = weight_variable([4,40])  # 1st layer of network weights
b1 = bias_variable([40])  # 1st layer of network bises
h1 = tf.nn.relu(tf.matmul(state,w1) + b1)  # RELU transformation

keep_prob = tf.placeholder(tf.float32)
h1_drop = tf.nn.dropout(h1, keep_prob) # dropout variable

w2 = weight_variable([40,2])  # 2nd layer of network weights
b2 = bias_variable([2])  # 2nd layer of network biases
output = tf.nn.softmax(tf.matmul(h1_drop,w2) + b2)  # softmax transformation

cross_entropy = tf.reduce_mean(-tf.reduce_sum(actions * tf.log(output), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(actions,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Add ops to save and restore all the variables.
#saver = tf.train.Saver()

sess.run(tf.initialize_all_variables())


### Save the variables to disk. (In case we want to store our model)
#save_path = saver.save(sess, "/Users/idiosocratic/desktop/env/capstone/model.ckpt")
#print("Model saved in file: %s" % save_path)


env = gym.make('CartPole-v0')
wondering_gnome = PolyGradAgent(env.action_space)

for i_episode in xrange(10000):
    observation = env.reset()
    episode_rewards = 0
    episode_state_action_list = []
    
    #episode_state_target_list = []
    
    exploiting = wondering_gnome.are_we_exploiting()
    
    for t in xrange(200):
        env.render()
        #print observation
        
        state_for_mem = observation
        
        current_state =  np.expand_dims(observation, axis=0)
        
        action = env.action_space.sample()  # initialize action randomly
        
        if exploiting:

            random_fate = np.random.random()
        
            if random_fate > wondering_gnome.epsilon:  # e-greedy implementation
            
                raw_output = output.eval(feed_dict = {state: current_state, keep_prob: 1.0})
        
                action = np.argmax(raw_output)
        
        
        observation, reward, done, info = env.step(action)
        
        episode_state_action_list.append((state_for_mem, action))
        
        episode_rewards += reward
        
        if done:
            
            #print "Episode finished after {} timesteps".format(t+1)
            break

    print "Episode: " + str(i_episode) + ", " + "Rewards: " + str(episode_rewards)
    print "Epsilon: "
    print wondering_gnome.epsilon
    print "Mem length: "
    print len(wondering_gnome.state_action_mem)

    wondering_gnome.last_100_episode_scores.append(episode_rewards)
    print "Running average of last 100 episodes: " + str(np.average(wondering_gnome.last_100_episode_scores))

    if np.average(wondering_gnome.last_100_episode_scores) >= 195.0:
        print "Solved!"
        assert False

    if wondering_gnome.did_we_do_well(episode_rewards):  # add episode to our memory if we did well

        wondering_gnome.add_to_memory(episode_state_action_list)


    wondering_gnome.update_high_score(episode_rewards)  # update high score


    if exploiting:

        wondering_gnome.decay_epsilon()


    batch_size = 100

    if len(wondering_gnome.state_action_mem) > batch_size:

        batch = wondering_gnome.get_next_batch(batch_size)
        
        train_step.run(feed_dict={state: batch[0], actions: batch[1], keep_prob: 0.75})

        wondering_gnome.net_trained = True


print "Size of our memory: "
print len(wondering_gnome.state_action_mem)

""" some data
623
581
743
721

2341
null
null
1153
null


620
540
null
null
818
null
"""


