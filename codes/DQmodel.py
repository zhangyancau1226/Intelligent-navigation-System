import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random
import collections


action_list = ['forward', 'back', 'left', 'right']
map_list5 = [0, 0, 2, 0, 0,
       0, -1, 0, 0, 0,
       0, -1, 0, 0, 0,
       0, 0, 0, -1, 0,
       1, 0, 0, 0, -1]

map_list10 = [0, 0, 2, 0, 3, 0, -1, 0, 0, 0,
       -1, -1, 0, 0, 0, 0, 0, 0, 0, -1,
       0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       0, -1, 0, 0, 3, 0, 0, 0, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       3, 0, -1, 0, 0, 0, -1, 0, 0, -1,
       1, 0, 0, 0, -1, -1, 0, 0, 0, 0]


class AgentAi:
    def __init__(self, map_size, action_size, gamma, min_epsilon):
        self.state_size = map_size**2
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1
        self.min_epsilon = min_epsilon
        self.d_rate = 0.001
        self.behavior_model = Sequential()
        self.behavior_model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.behavior_model.add(Dense(self.action_size, activation='linear'))
        self.behavior_model.compile(loss='mse', optimizer='Adam')
        self.target_model = Sequential()
        self.target_model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.target_model.add(Dense(self.action_size, activation='linear'))
        self.target_model.compile(loss='mse', optimizer='Adam')
        self.update_model()

    def update_model(self):
        self.target_model.set_weights(self.behavior_model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.behavior_model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        min_batch = self.memory
        if batch_size < len(self.memory):
            min_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in min_batch:
            target = self.behavior_model.predict(state)
            target[0][action] = reward
            if not done:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # target_f = self.model.predict(state)
            # target_f[0][action] = target
            self.behavior_model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.d_rate


def check_movable(pos, action, map_size):
    if pos % map_size == 0 and action == 2:
        return False
    if (pos-map_size+1) % map_size == 0 and action == 3:
        return False
    if action == 0:
        pos -= map_size
    if action == 1:
        pos += map_size
    if pos < 0 or pos >= map_size**2:
        return False
    return True


# Let's keep doning this, yeah
def take_action(map, action, reward, map_size):
    agent = map.index(1)
    if action == action_list.index('forward'):
        if check_movable(agent, action, map_size):
            map[agent] = 0
            agent -= map_size
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] == 2:
                reward += 100
                return map, reward, True
            if map[agent] == 3:
                reward += 20
                map[agent] = 1
                return map, reward, False
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('back'):
        if check_movable(agent, action, map_size):
            map[agent] = 0
            agent += map_size
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] == 2:
                reward += 100
                return map, reward, True
            if map[agent] == 3:
                reward += 20
                map[agent] = 1
                return map, reward, False
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('left'):
        if check_movable(agent, action, map_size):
            map[agent] = 0
            agent -= 1
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] == 2:
                reward += 100
                return map, reward, True
            if map[agent] == 3:
                reward += 20
                map[agent] = 1
                return map, reward, False
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('right'):
        if check_movable(agent, action, map_size):
            map[agent] = 0
            agent += 1
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] == 2:
                reward += 100
                return map, reward, True
            if map[agent] == 3:
                reward += 20
                map[agent] = 1
                return map, reward, False
        else:
            reward -= 1
            return map, reward, False


agent = AgentAi(10, 4, 0.9, 0.1)
episode = 100000
for e in range(episode):

    state = [0, 0, 2, 0, 3, 0, -1, 0, 0, 0,
       -1, -1, 0, 0, 0, 0, 0, 0, 0, -1,
       0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       0, -1, 0, 0, 3, 0, 0, 0, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
       0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
       3, 0, -1, 0, 0, 0, -1, 0, 0, -1,
       1, 0, 0, 0, -1, -1, 0, 0, 0, 0]
    reward = 0

    for step in range(100):
        ndstate = np.reshape(state, [1, 100])
        action = agent.act(ndstate)
        next_state, reward, done = take_action(state,action,reward,10)
        ndnext_state = np.reshape(next_state, [1, 100])
        agent.remember(ndstate, action, reward, ndnext_state, done)
        state = next_state
        if done:
            agent.update_model()
            print("episode: {}/{}, step: {}, score: {}".format(e, episode, step, reward))
            break
    agent.replay(32)
