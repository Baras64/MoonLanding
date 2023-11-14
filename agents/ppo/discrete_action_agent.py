import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # -1:cpu, 0:first gpu
import pylab
import numpy as np
import gym
import tensorflow as tf
from tensorboardX import SummaryWriter
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import copy

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass
    
from model import Actor_Model, Critic_Model


class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env, env_name):
    # def __init__(self, env_name):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = env
        # self.env_name = env_name       
        # self.env = gym.make(env_name)
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 30000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle=False
        self.Training_batch = 1000
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        
        self.Actor_name = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_PPO_Critic.h5"
        
    def act(self, state):
        """ example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_size, p=prediction)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot, prediction

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r

    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages
        #discounted_r = self.discount_rewards(rewards)
        #advantages = np.vstack(discounted_r - values)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        pylab.plot(advantages,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom PPO loss function we unpack it
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
        
    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.title(self.env_name+" PPO training cycle", fontsize=18)
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            self.lr *= 0.95
            K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run(self): # train only when episode is finished
        state, _ = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            while not done:
                self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                # next_state, reward, done, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _  = self.env.step(action)
                done = terminated or truncated
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    
                    self.replay(states, actions, rewards, predictions, dones, next_states)

                    pseudo_state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state, _ = pseudo_state
                    state = np.reshape(state, [1, self.state_size[0]])

            if self.episode >= self.EPISODES:
                break
        self.env.close()

    def run_batch(self): # train every self.Training_batch episodes
        state, _ = self.env.reset()
        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, predictions, dones = [], [], [], [], [], []
            for t in range(self.Training_batch):
                self.env.render()
                # Actor picks an action
                action, action_onehot, prediction = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                # next_state, reward, done, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _  = self.env.step(action)
                done = terminated or truncated
                # Memorize (state, action, reward) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action_onehot)
                rewards.append(reward)
                dones.append(done)
                predictions.append(prediction)
                # Update current state
                state = np.reshape(next_state, [1, self.state_size[0]])
                score += reward
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)

                    pseudo_state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state, _ = pseudo_state
                    state = np.reshape(state, [1, self.state_size[0]])
                    
            self.replay(states, actions, rewards, predictions, dones, next_states)
            if self.episode >= self.EPISODES:
                break
        self.env.close()  

        
    # def run_multiprocesses(self, num_worker = 4):
    #     works, parent_conns, child_conns = [], [], []
    #     for idx in range(num_worker):
    #         parent_conn, child_conn = Pipe()
    #         work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
    #         work.start()
    #         works.append(work)
    #         parent_conns.append(parent_conn)
    #         child_conns.append(child_conn)

    #     states =        [[] for _ in range(num_worker)]
    #     next_states =   [[] for _ in range(num_worker)]
    #     actions =       [[] for _ in range(num_worker)]
    #     rewards =       [[] for _ in range(num_worker)]
    #     dones =         [[] for _ in range(num_worker)]
    #     predictions =   [[] for _ in range(num_worker)]
    #     score =         [0 for _ in range(num_worker)]

    #     state = [0 for _ in range(num_worker)]
    #     for worker_id, parent_conn in enumerate(parent_conns):
    #         state[worker_id] = parent_conn.recv()

    #     while self.episode < self.EPISODES:
    #         predictions_list = self.Actor.predict(np.reshape(state, [num_worker, self.state_size[0]]))
    #         actions_list = [np.random.choice(self.action_size, p=i) for i in predictions_list]

    #         for worker_id, parent_conn in enumerate(parent_conns):
    #             parent_conn.send(actions_list[worker_id])
    #             action_onehot = np.zeros([self.action_size])
    #             action_onehot[actions_list[worker_id]] = 1
    #             actions[worker_id].append(action_onehot)
    #             predictions[worker_id].append(predictions_list[worker_id])

    #         for worker_id, parent_conn in enumerate(parent_conns):
    #             next_state, reward, done, _ = parent_conn.recv()

    #             states[worker_id].append(state[worker_id])
    #             next_states[worker_id].append(next_state)
    #             rewards[worker_id].append(reward)
    #             dones[worker_id].append(done)
    #             state[worker_id] = next_state
    #             score[worker_id] += reward

    #             if done:
    #                 average, SAVING = self.PlotModel(score[worker_id], self.episode)
    #                 print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
    #                 self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
    #                 self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
    #                 score[worker_id] = 0
    #                 if(self.episode < self.EPISODES):
    #                     self.episode += 1
                        
    #         for worker_id in range(num_worker):
    #             if len(states[worker_id]) >= self.Training_batch:
    #                 self.replay(states[worker_id], actions[worker_id], rewards[worker_id], predictions[worker_id], dones[worker_id], next_states[worker_id])
                    
    #                 states[worker_id] = []
    #                 next_states[worker_id] = []
    #                 actions[worker_id] = []
    #                 rewards[worker_id] = []
    #                 dones[worker_id] = []
    #                 predictions[worker_id] = []

    #     # terminating processes after while loop
    #     works.append(work)
    #     for work in works:
    #         work.terminate()
    #         print('TERMINATED:', work)
    #         work.join()
            

    def test(self, test_episodes = 100):
        self.load()
        for e in range(100):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                self.env.render()
                action = np.argmax(self.Actor.predict(state)[0])
                # state, reward, done, _ = self.env.step(action)
                next_state, reward, terminated, truncated, _  = self.env.step(action)
                done = terminated or truncated
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    print("episode: {}/{}, score: {}".format(e, test_episodes, score))
                    break
        self.env.close()