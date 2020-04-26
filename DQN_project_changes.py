import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten,LSTM
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import tensorflow as tf
from collections import deque
import time
import random
from tqdm import tqdm
import os
# import cv2
import re
np.warnings.filterwarnings('ignore')

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '8x32'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 300

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

#-----------------------------------------------------------------------------------------------------------
# CREATING STATE AND ACTION SPACE , ALSO CREATING VIRTUAL REWARDS FOR TESTING
#-----------------------------------------------------------------------------------------------------------

def return_all_possible_states(state_index,range_val,default_state,state_name):
    state_dict = []
    start = range_val[0]
    end = range_val[1]
    for i in range(start,end+1):
        default_state[state_index] = i
        state_dict.append(list(default_state))
    return state_dict

def all_possible_states(state_dict,state_list):
    possible_states = []
    for key in state_list.keys():
        states = state_dict.get(key)
        for state in states:
            if state not in possible_states:
                possible_states.append(state)
    return possible_states

state_list = {"MaxRequestWorkers":0,"KeepAliveTimeOut":1,"MinSpareThreads":2,"MaxSpareThreads":3,"ThreadsPerChild":4,"MaxConnectionsPerChild":5,"ServerLimit":6,"StartServers":7}

action_space = action_space = [["MaxRequestWorkers",[(1,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["KeepAliveTimeOut",[(0,0,0),(1,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MinSpareThreads",[(0,0,0),(0,0,0),(1,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxSpareThreads",[(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["ThreadsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxConnectionsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,0,0),(0,0,0)]],
                ["ServerLimit",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0),(0,0,0)]],
                ["StartServers",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(1,0,0)]],
                ["MaxRequestWorkers",[(0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["KeepAliveTimeOut",[(0,0,0),(0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MinSpareThreads",[(0,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxSpareThreads",[(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["ThreadsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxConnectionsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,0,0),(0,0,0)]],
                ["ServerLimit",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0),(0,0,0)]],
                ["StartServers",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,1,0)]],
                ["MaxRequestWorkers",[(0,0,1),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["KeepAliveTimeOut",[(0,0,0),(0,0,1),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MinSpareThreads",[(0,0,0),(0,0,0),(0,0,1),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxSpareThreads",[(0,0,0),(0,0,0),(0,0,0),(0,0,1),(0,0,0),(0,0,0),(0,0,0),(0,0,0)]],
                ["ThreadsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,1),(0,0,0),(0,0,0),(0,0,0)]],
                ["MaxConnectionsPerChild",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,1),(0,0,0),(0,0,0)]],
                ["ServerLimit",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,1),(0,0,0)]],
                ["StartServers",[(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,0),(0,0,1)]]]


#-----------------------------------------------------------------------------------------------------------
# Own Tensorboard class
#-----------------------------------------------------------------------------------------------------------

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def _write_logs(self, logs, index):
        for name,value in logs.items():                                                                                                                                                                                                                               
            with self.writer.as_default():
                tf.summary.scalar(name, value,step=index)
        self.writer.flush() 

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)
#-----------------------------------------------------------------------------------------------------------
# CREATING THE APACHE ENVIRONMENT
#-----------------------------------------------------------------------------------------------------------
class Apache_environment:

    #Getting the default performance using the initial values
    os.system("ab -n 1000 -c 100 -r https://54.191.74.174/ >/home/output.txt 2>&1")
    search = open("/home/output.txt")
    string = "Requests per second:"
    line = ""
    req_per_second = ""
    for line in search.readlines():
        if string in line:
            req_per_second = re.sub('[^\d\.]', '', line)
            break 
    def_perf = req_per_second

    def reset(self):
        return [256,2,128,192,64,5000,4,2]

    def step(self,action,current_state):
        parameter = action_space[action][0]
        param_index = state_list.get(parameter)
        if param_index == 1: #if keepAliveTimeOut is increased, dont have to change other param values
            current_state[param_index] += 1
        elif param_index == 4: #if ThreadsPerChild is modified, then its dependent param "MaxRequestWorkers" has to be modified accordingly
            current_state[param_index] += 1
            if current_state[param_index] > 64:
                current_state[param_index] = 64
            else:
                current_state[0] = current_state[param_index]*current_state[6]
        elif param_index == 7: #if StartServers is changed, make sure its >= ServerLimit, and its dependent param "MinSpareThreads" needs to be changed accordingly
            current_state[param_index] += 1
            if current_state[param_index] >  current_state[6]: #since ServerLimit is modified, it dependent param "MaxSpareThreads" needs to be modified
                current_state[6] =  current_state[6] + abs(current_state[param_index] - current_state[6])
                current_state[3] = current_state[6]*current_state[param_index]

            #since ServerLimit is modified, it dependent param "MaxSpareThreads" needs to be modified
            current_state[2] = current_state[param_index]*current_state[param_index]
        

        new_state = current_state
        
        reward = self.get_reward(current_state)
        return new_state, reward

    def get_simulated_reward(self):
        # def_perf = 7.01
        # reward = virtual_rewards[all_states.index(cur_state)]
        reward = np.random.randint(10,100)
        return reward - Apache_environment.def_perf

    def get_reward(self,cur_state):
        value_dict = { 'MaxRequestWorkers':cur_state[0],
        'KeepAliveTimeOut' : cur_state[1],
        'MinSpareThreads' : cur_state[2],
        'MaxSpareThreads' : cur_state[3],
        'ThreadsPerChild' : cur_state[4],
        'MaxConnectionsPerChild' : cur_state[5],
        'ServerLimit' : cur_state[6],
        'StartServers' : cur_state[7]}

        line_numbers = [23,25,20,21,22,24,18,19]
        i = 0
        for key,value in value_dict.items():
            command = "sed -i '" + str(line_numbers[i]) + "s/" + str(key) + "*/" + str(key) + "    " + str(value) + "/'  /opt/bitnami/apache2/conf/bitnami/httpd.conf"
            os.system(command)
            i += 1 

        os.system("sudo /opt/bitnami/ctlscript.sh restart apache")
        os.system("ab -n 1000 -c 100 -r https://54.191.74.174/ >/home/output.txt 2>&1")
        search = open("/home/output.txt")
        string = "Requests per second:"
        for line in search.readlines():
            if string in line:
                req_per_second = re.sub('[^\d\.]', '', line)
                break 
        
        return float(req_per_second) - Apache_environment.def_perf

#-----------------------------------------------------------------------------------------------------------
#DEFINING THE DEEP-Q NETWORK AGENT
#-----------------------------------------------------------------------------------------------------------
class DQNAgent:
    def __init__(self):

        # main model  # gets trained every step
        self.model = self.create_model()

        # Target model this is what we .predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Dense(32,kernel_initializer='random_normal', input_dim=8))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(32,kernel_initializer='random_normal'))
        model.add(Activation("relu"))
        model.add(Dropout(0.2))

        model.add(Dense(24, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        print(model.summary)
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # def get_qs(self, state, step):
    #     return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    
     # Trains main network every step during episode
    def train(self, terminal_state, step):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
    
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
        
         # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        my_state = np.asarray(state)
        return self.model.predict(np.array(my_state).reshape(-1, *my_state.shape))
        # print("the predicted value is:",self.model.predict(np.asarray(state)))
        # return self.model.predict(np.asarray(state))


#-----------------------------------------------------------------------------------------------------------
#OFFLINE TRAINING SESSION FOR THE DEEP-Q NETWORK 
#-----------------------------------------------------------------------------------------------------------
agent = DQNAgent()
env = Apache_environment()
# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    # episode_reward = 0
    episode_reward = []
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, 24)

        # new_state, reward, done = env.step(action)
        new_state, reward= env.step(action,current_state)

        # Transform new continous state to new discrete state and count reward
        episode_reward.append(reward)

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1
        if step == 1500:
            done = True

    min_reward = min(episode_reward)
    max_reward = max(episode_reward)
    average_reward = sum(episode_reward)/1500
    agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
    if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)