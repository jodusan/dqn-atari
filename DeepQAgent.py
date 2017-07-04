import numpy as np
from keras.losses import MSE
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from collections import deque

from utils import preprocess_frame
from ReplayBuffer import ReplayBuffer


class DeepQAgent:
    def __init__(self, replay_queue_size, env, sample_amount, buffer_start_size):
        self.replay_queue = ReplayBuffer(replay_queue_size)
        self.main_network = self.get_model()
        self.target_network = self.get_model()
        self.env = env
        self.sample_amount = sample_amount
        self.buffer_start_size = buffer_start_size
        self.reward_log = deque(100*[0], 100)

        self.init_buffer()

    def get_model(self):
        model = Sequential()
        model.add(Conv2D(16, 8, strides=(4, 4), activation='relu', input_shape=(80, 80, 4)))
        model.add(Conv2D(32, 4, strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(4))
        model.compile(loss=MSE,
                      optimizer=Adam(lr=0.002),
                      metrics=['accuracy', MSE])
        return model

    def select_action(self):
        pass

    def predict(self, obs):
        return self.main_network.predict(obs, verbose=False)[0]

    def gen_target(self, obs):
        return self.target_network.predict(obs, verbose=False)[0]

    def init_buffer(self):
        obs_list = [preprocess_frame(self.env.reset())] * 5
        for i in range(self.buffer_start_size):
            obs_list.pop(0)

            action = self.env.action_space.sample()
            obs_p, r, done, _ = self.env.step(action)
            obs_list.append(preprocess_frame(obs_p))

            if done:
                self.env.reset()

            transition = (np.stack(obs_list, -1), action, r, done)
            self.replay_queue.append(transition)

    def train_main(self, x, y):
        return self.main_network.train_on_batch(x, y)[2]

    def sample_minibatch(self):
        return self.replay_queue.sample_random(self.sample_amount)

    def update_target_network(self):
        for main_layer, target_layer in zip(self.main_network.layers, self.target_network.layers):
            target_layer.set_weights(main_layer.get_weights())

    def report(self, total_steps, steps, rewards, episode):
        self.reward_log.append(rewards)
        print('Episode: {} Total steps: {}, steps: {}, reward: {} mean-100: '
              '{} epsilon: {}'.format(episode, total_steps, steps, rewards,
                                      np.mean(self.reward_log), self.eps))
