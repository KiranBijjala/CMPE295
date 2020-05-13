import numpy as np
import keras.backend.tensorflow_backend as backend
from keras.models import Sequential, load_model
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
import subprocess

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'DQNAgent'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 600

# Exploration settings
EPSILON = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.90
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

#-----------------------------------------------------------------------------------------------------------
# CREATING STATE AND ACTION SPACE , ALSO CREATING VIRTUAL REWARDS FOR TESTING
#-----------------------------------------------------------------------------------------------------------

state_list = {"MaxRequestWorkers":0,"KeepAliveTimeOut":1,"MinSpareThreads":2,
                "MaxSpareThreads":3,"ThreadsPerChild":4,"MaxConnectionsPerChild":5,"ServerLimit":6,"StartServers":7}

#100 => INCREASE,   010 => DECREASE,   001 => NEUTRAL
action_space = [["KeepAliveTimeOut","100"],["MinSpareThreads","100"],["MaxSpareThreads","100"],["ThreadsPerChild","100"],
                ["MaxConnectionsPerChild","100"],["ServerLimit","100"],["StartServers","100"],["KeepAliveTimeOut","010"],
                ["MinSpareThreads","010"],["MaxSpareThreads","010"],["ThreadsPerChild","010"],["MaxConnectionsPerChild","010"],
                ["ServerLimit","010"],["StartServers","010"],["KeepAliveTimeOut","001"],["MinSpareThreads","001"],
                ["MaxSpareThreads","001"],["ThreadsPerChild","001"],["MaxConnectionsPerChild","001"],
                ["ServerLimit","001"],["StartServers","001"]]


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
    return_code = subprocess.run("ab -n 1000 -c 100 -r https://54.213.208.22/ >/home/output.txt 2>&1",shell=True)
    line = ""
    req_per_second = ""
    if return_code.returncode == 0:
        search = open("/home/output.txt")
        string = "Requests per second:"
        for line in search.readlines():
            if string in line:
                req_per_second = re.sub('[^\d\.]', '', line)
                break 
    def_perf = float(req_per_second)
    total_req = 1000
    concurrent_req = 100

    def reset(self):
        #Whenever the environment is reset, reset the param values to default values and change the config file to default param values
        default_param_values = [256,2,128,192,64,5000,4,2]
        value_dict = { 'MaxRequestWorkers':default_param_values[0],
        'KeepAliveTimeOut' : default_param_values[1],
        'MinSpareThreads' : default_param_values[2],
        'MaxSpareThreads' : default_param_values[3],
        'ThreadsPerChild' : default_param_values[4],
        'MaxConnectionsPerChild' : default_param_values[5],
        'ServerLimit' : default_param_values[6],
        'StartServers' : default_param_values[7]}

        line_numbers = [23,25,20,21,22,24,18,19]
        i = 0
        for key,value in value_dict.items():
            command = "sed -i '" + str(line_numbers[i]) + "s/" + str(key) + ".*/" + str(key) + "    " + str(value) + "/'  /opt/bitnami/apache2/conf/bitnami/httpd.conf"
            run = subprocess.run(command,shell=True)
            if run.returncode != 0:
                print("Error code: Couldnt execute sed command")
                exit()
            i += 1 
        return default_param_values

    def step(self,action,current_state,episode,test=False):
        parameter = action_space[action][0]
        action_to_perform = action_space[action][1]
        param_index = state_list.get(parameter)
        
        #When the action is to INCREASE 
        if action_to_perform == "100":
            if param_index == 1:
                current_state[param_index] += 1
            elif param_index == 2:
                current_state[param_index] += 1
                if current_state[param_index] > current_state[3]:
                    current_state[3] += abs(current_state[3] - current_state[param_index]) 
            elif param_index == 3:
                current_state[param_index] += 1
            elif param_index == 4:
                if current_state[param_index] + 1 > 64:
                    current_state[0] = 64
                else:
                    current_state[param_index] += 1
                current_state[0] = current_state[param_index]*current_state[6]
            elif param_index == 5:
                current_state[param_index] += 250
            elif param_index == 6:
                current_state[param_index] += 1
                current_state[0] = current_state[4]*current_state[param_index]
                if current_state[param_index]*current_state[7] > current_state[3]:
                    current_state[3] = current_state[param_index]*current_state[7]
            elif param_index == 7:
                current_state[param_index] += 1
                if current_state[param_index] > current_state[6]:
                    current_state[6] += abs(current_state[6] - current_state[param_index])
                    current_state[0] = current_state[4]*current_state[param_index]
                    if current_state[param_index]*current_state[7] > current_state[3]:
                        current_state[3] = current_state[param_index]*current_state[7]
                if current_state[param_index]*current_state[param_index] > current_state[2]:
                    current_state[3] = current_state[param_index]*current_state[param_index]
        
        #When the action is to DECREASE
        if action_to_perform == "010":
            if param_index == 1:
                current_state[param_index] -= 1
                if current_state[param_index] == 0:
                    current_state[param_index] = 1
            elif param_index == 2:
                current_state[param_index] -= 1
            elif param_index == 3:
                if current_state[param_index] - 1 < current_state[2]:
                    current_state[param_index] = current_state[2]
                else:
                    current_state[param_index] -= 1
            elif param_index == 4:
                current_state[param_index] -= 1
                if current_state[param_index] <= 0:
                    current_state[param_index] = 1
                current_state[0] = current_state[param_index]*current_state[6]
            elif param_index == 5:
                if current_state[param_index] - 250 <= 0:
                    current_state[param_index] = 250
                else:
                    current_state[param_index] -= 250
            elif param_index == 6:
                if current_state[param_index] - 1 < current_state[7]:
                    current_state[param_index] = current_state[7]
                elif current_state[param_index] - 1 == 0:
                    current_state[param_index] = 1
                else:
                    current_state[param_index] -= 1
                current_state[0] = current_state[param_index]*current_state[4]
            elif param_index == 7:
                if current_state[param_index] - 1 <= 0:
                    current_state[param_index] = 1
                else:
                    current_state[param_index] -= 1

        if current_state[2] > current_state[3]:
            current_state[3] = current_state[2] 

        #When the action_to_perform is "001" which is NEUTRAL, then we make no changes and the current_state is directly assigned as the
        #new_state
        print("The current state is: ",current_state)

        new_state = current_state
             
        if test==True:
            Apache_environment.total_req = np.random.randint(1000,5000)
            Apache_environment.concurrent_req = np.random.randint(10,1000)
            print("Total requests: ",Apache_environment.total_req, "Concurrent requests: ",Apache_environment.concurrent_req)
            reward = self.get_reward(current_state,Apache_environment.total_req,Apache_environment.concurrent_req)
        else:
            if episode%30 == 0:
                Apache_environment.total_req = np.random.randint(1000,5000)
                Apache_environment.concurrent_req = np.random.randint(10,1000)

            reward = self.get_reward(current_state,Apache_environment.total_req,Apache_environment.concurrent_req)

        return new_state, reward,Apache_environment.total_req,Apache_environment.concurrent_req

    def get_reward(self,cur_state,total_requests,concurrent_requests):
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
            command = "sed -i '" + str(line_numbers[i]) + "s/" + str(key) + ".*/" + str(key) + "    " + str(value) + "/'  /opt/bitnami/apache2/conf/bitnami/httpd.conf"
            run = subprocess.run(command,shell=True)
            if run.returncode != 0:
                print("Error code: Couldnt execute sed command")
                exit()
            i += 1 

        return_val = subprocess.run("sudo /opt/bitnami/ctlscript.sh restart apache",shell=True)
        if return_val.returncode == 0:
            ab_command = "ab -n " +str(total_requests) + " -c " + str(concurrent_requests) + " -r https://54.213.208.22/ >/home/output.txt 2>&1"
            return_code = subprocess.run(ab_command,shell=True)
            if return_code.returncode == 0:
                search = open("/home/output.txt")
                string = "Requests per second"
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
        model.add(Dense(32,kernel_initializer='random_normal', input_dim=10))
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

        model.add(Dense(20, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        print(model.summary)
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # def get_qs(self, state, step):
    #     return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
    
     # Trains main network every step during episode
    def train(self, terminal_state, step, total_requests,concurrent_requests):

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

        X = [] #states
        y = [] #Q -values

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
            
            #Appending the total_requests and the concurrent_requests depicting the current load on the system
            current_state.append(total_requests)
            current_state.append(concurrent_requests)

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
#ONLINE TRAINING SESSION FOR THE DEEP-Q NETWORK 
#-----------------------------------------------------------------------------------------------------------
def train_dqn_agent(Agent,environment):
    agent = Agent
    env = environment
    best_reward = -float('inf')
    final_best_rewards = []
    epsilon = EPSILON
    total_requests = 0
    concurrent_requests = 0
    # Iterate over episodes
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode
        # Restarting episode - reset episode reward and step number
        # episode_reward = 0
        episode_reward = []
        step = 1
        total_reward = 0
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
                action = np.random.randint(0, 20)

            # new_state, reward, done = env.step(action)
            new_state, reward, total_requests, concurrent_requests = env.step(action,current_state,episode,False)

            #Calculating the total reward in each episode
            total_reward += reward
            # Transform new continous state to new discrete state and count reward
            episode_reward.append(reward)

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step,total_requests,concurrent_requests)

            current_state = new_state
            step += 1
            if step == 75:
                done = True

        min_reward = min(episode_reward)
        max_reward = max(episode_reward)
        agent.tensorboard.update_stats(total_reward=total_reward,reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)
        # if min_reward >= MIN_REWARD:
        #         agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
        if total_reward > best_reward:
            final_best_rewards.append(best_reward)
            best_reward = total_reward
            agent.model.save(f'models/{MODEL_NAME}__best_reward__{best_reward:_>7.2f}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    agent.target_model.save(f'models/{MODEL_NAME}final.model')
    return max(final_best_rewards)


#-----------------------------------------------------------------------------------------------------------
#TESTING THE DEEP-Q NETWORK 
#-----------------------------------------------------------------------------------------------------------
def test(model, env,episode):
    done = False
    current_state = env.reset()
    total_reward = 0.0
    while not done:
        action = np.argmax(model.predict(np.array(np.asarray(current_state)).reshape(-1, *current_state.shape)))
        next_state, reward, done = env.step(action,current_state,episode,True)
        current_state = next_state
        total_reward += reward
    return total_reward



if __name__ == "__main__":
    env = Apache_environment()
    agent = DQNAgent()

    #Getiing back the best reward from the training of the DQNAgent on the environment
    best_reward = train_dqn_agent(agent, env)
    print("Best reward after Training is: ",best_reward)
    # #Loading the model based on the best_reward obtained from the saved models
    # LOAD_MODEL = f'models/{MODEL_NAME}__best_reward__{best_reward:_>7.2f}.model'
    # test_model = load_model(LOAD_MODEL)

    # for episode in range(2):
    #     reward = test(test_model,env,episode)
    #     print("Test Episode#:{} reward:{}".format(episode,reward) )


