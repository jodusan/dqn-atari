import numpy as np


class ReplayBuffer:

    def __init__(self, size):
        self._size = size
        self._storage = [None] * size
        self._index = -1
        self._curr_length = 0

    def __len__(self):
        return self._curr_length

    def append(self, item):
        self._index = (self._index + 1) % self._size
        self._storage[self._index] = item
        if self._curr_length < self._size:
            self._curr_length += 1

    def sample_random(self, sample_amount):
        rand_indxs = np.random.choice(range(self._curr_length),
                                      sample_amount,
                                      replace=False)
        #return [self._storage[ind] for ind in rand_indxs]
        obs = np.empty([len(rand_indxs), 80, 80, 5])
        actions = np.empty([len(rand_indxs)], dtype=np.uint8)
        rewards = np.empty([len(rand_indxs)], dtype=np.uint8)
        dones = np.empty([len(rand_indxs)], dtype=np.bool)
        for i, ind in enumerate(rand_indxs):
            obs[i] = self._storage[ind][0]
            actions[i] = self._storage[ind][1]
            rewards[i] = self._storage[ind][2]
            dones[i] = self._storage[ind][3]

        return obs, actions, rewards, dones


