import gym
import numpy as np
from utils import preprocess_frame
from DeepQAgent import DeepQAgent

BATCH_SIZE = 32
BUFFER_START_SIZE = 40
BUFFER_SIZE = 100
TARGET_UPDATE = 50
NUM_EPISODES = int(2e5)
MAX_STEPS = int(1e6)
GAMMA = 0.99
EPSILON_DECAY_STEPS = int(1e6)
MIN_EPS = 0.1


env = gym.make("Breakout-v0")
total_reward = 0
eps = 1

agent = DeepQAgent(BUFFER_SIZE, env, BATCH_SIZE, BUFFER_START_SIZE)

total_steps = 0
for ep in range(NUM_EPISODES):

    step = 0
    done = False
    obs_list = [preprocess_frame(env.reset())] * 5

    ep_reward = 0

    # Loop until MAX_STEPS reached or env returns done (checked at the bottom)
    while step < MAX_STEPS:
        step += 1
        total_steps += 1

        obs_list.pop(0)

        if np.random.random() < eps:
            action = env.action_space.sample()
        else:
            obs_nd = np.stack(obs_list, -1)
            action = np.argmax(agent.predict(obs_nd[None,]))

        obs_p, r, done, _ = env.step(action)
        obs_list.append(preprocess_frame(obs_p))

        total_reward += r
        ep_reward += r

        if done and r == 1:
            r = 200
        elif done:
            r = -200

        transition = (np.stack(obs_list, -1), action, r, done)
        agent.replay_queue.append(transition)

        minibatch = agent.sample_minibatch()

        #observations, actions, rewards, dones = minibatch
        observations = minibatch[0]
        actions = minibatch[1]
        rewards = minibatch[2]
        dones = minibatch[3]



        targets = agent.gen_target(observations[:, :, :, 0:4])
        targets_p = agent.gen_target(observations[:, :, :, 1:])

        print(targets_p.shape)

        arranged = np.arange(len(targets))
        targets[arranged, actions] = rewards + (dones==False) * GAMMA * np.max(targets_p, axis=1)

        agent.train_main(observations[:, :, :, 0:4], targets)

        if total_steps % TARGET_UPDATE == 0:
            agent.update_target_network()

        if done:
            agent.report(total_steps, step, ep_reward, ep)









