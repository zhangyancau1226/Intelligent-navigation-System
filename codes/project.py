from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
import math
import os
import random
import sys
import time
import malmoutils

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random
import collections



action_list = ['forward', 'back', 'left', 'right']
map_list = [0, 0, 2, 0, 0,
       0, -1, 0, 0, 0,
       0, -1, 0, 0, 0,
       0, 0, 0, -1, 0,
       1, 0, 0, 0, 0]

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


map_list102 = [0, 0, 0, 0, 3, 0, -1, 0, 0, 0,
            -1, -1, 0, 0, 0, 0, 0, 0, 0, -1,
             0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            0, -1, 0, 0, 3, 0, 0, 2, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            3, 0, -1, 0, 0, 0, -1, 0, 0, -1,
            1, 0, 0, 0, -1, -1, 0, 0, 0, 0]

def print_map(grid):
    for i in range(10):
        print(str(grid[10*i+0]) + " " + str(grid[10*i+1]) + " " + str(grid[10*i+2]) + " " + str(grid[10*i+3]) + " " + str(grid[10*i+4])+ " " + str(grid[10*i+5])+ " " + str(grid[10*i+6])+ " " + str(grid[10*i+7])+ " " + str(grid[10*i+8])+ " " + str(grid[10*i+9]))


def load_grid(world_state):
    """
    Used the agent observation API to get a 21 X 21 grid box around the agent (the agent is in the middle).

    Args
        world_state:    <object>    current agent world state

    Returns
        grid:   <list>  the world grid blocks represented as a list of blocks (see Tutorial.pdf)
    """
    while world_state.is_mission_running:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)
            grid = observations.get(u'floorAll', 0)
            break
    return grid

class AgentAi:
    def __init__(self, map_size, action_size, gamma, min_epsilon):
        self.state_size = map_size**2
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 0.1
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

    print("Action is " + action_list[action])
    print("State is ")
    print_map(map)
    print("Reward is " + str(reward))
    print("Agent is " + str(agent))

    if action == action_list.index('forward'):
        if check_movable(agent, action, map_size):
            agent_host.sendCommand('movesouth 1')
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
            agent_host.sendCommand('movenorth 1')
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
            agent_host.sendCommand('moveeast 1')
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
            agent_host.sendCommand('movewest 1')
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
# agent.target_model.load_weights('weights/1200target2.h5')
# agent.behavior_model.load_weights('weights/1200behave2.h5')
#agent.model.load_weights('new40003.h5')

agent_host = MalmoPython.AgentHost()

# Find the default mission file by looking next to the schemas folder:
schema_dir = None
try:
    schema_dir = os.environ['MALMO_XSD_PATH']
except KeyError:
    print("MALMO_XSD_PATH not set? Check environment.")
    exit(1)
mission_file = os.path.abspath(os.path.join(schema_dir, '..', 
    'sample_missions', 'avengers.xml')) # Integration test path
if not os.path.exists(mission_file):
    mission_file = os.path.abspath(os.path.join(schema_dir, '..', 
        'Sample_missions', 'avengers.xml')) # Install path
if not os.path.exists(mission_file):
    print("Could not find avengers.xml under MALMO_XSD_PATH")
    exit(1)

agent_host.addOptionalStringArgument('mission_file',
    'Path/to/file from which to load the mission.', mission_file)
malmoutils.parse_command_line(agent_host)


# -- set up the mission -- #
mission_file = agent_host.getStringArgument('mission_file')
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    my_mission = MalmoPython.MissionSpec(mission_xml, True)
my_mission.removeAllCommandHandlers()
my_mission.allowAllDiscreteMovementCommands()
my_mission.requestVideo( 640, 360 )

my_mission.setViewpoint( 1 )
my_mission_record = MalmoPython.MissionRecordSpec()

my_clients = MalmoPython.ClientPool()
my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available


max_retries = 3
agentID = 0
expID = 'Avengers AI'

# for retry in range(max_retries):
#             try:
#                 agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, 1) )
#                 break
#             except RuntimeError as e:
#                 if retry == max_retries - 1:
#                     print("Error starting mission:",e)
#                     exit(1)
#                 else:
#                     time.sleep(2.5)


# Loop until mission starts:
# print("Waiting for the mission", (1), "to start ",)
# world_state = agent_host.getWorldState()
# while not world_state.has_mission_begun:
#     #sys.stdout.write(".")
#     time.sleep(0.1)
#     world_state = agent_host.getWorldState()
#     for error in world_state.errors:
#         print("Error:",error.text)

# print()
# print("Mission", (1), "running.")



episode = 50000
for e in range(episode):

    time.sleep(0.1)
    state = [0, 0, 0, 0, 3, 0, -1, 0, 0, 0,
            -1, -1, 0, 0, 0, 0, 0, 0, 0, -1,
             0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            0, -1, 0, 0, 3, 0, 0, 2, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            0, -1, 3, 0, 0, 0, 0, 3, 0, -1,
            0, 0, 0, -1, 0, 0, -1, 0, 0, -1,
            3, 0, -1, 0, 0, 0, -1, 0, 0, -1,
            1, 0, 0, 0, -1, -1, 0, 0, 0, 0]

    reward = 0
    for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_clients, my_mission_record, agentID, "%s-%d" % (expID, 1) )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2.5)

    print("Waiting for the mission", (1), "to start ",)
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        #sys.stdout.write(".")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)

    print()
    print("Mission", (1), "running.")

    for step in range(100):
        # time.sleep(1)
        print(step)
        ndstate = np.reshape(state, [1, 100])
        action = agent.act(ndstate)
        next_state, reward, done = take_action(state,action,reward,10)
        time.sleep(0.2)
        ndnext_state = np.reshape(next_state, [1, 100])
        agent.remember(ndstate, action, reward, ndnext_state, done)
        state = next_state
        if done:
            agent.update_model()
            time.sleep(0.3) # (let the Mod reset)
            print("episode: {}/{}, score: {}".format(e, episode, reward))
            # -- clean up -- #
            time.sleep(0.3) # (let the Mod reset)
            break
    
    time.sleep(0.2) # (let the Mod reset)
    agent.replay(32)
    time.sleep(0.2) # (let the Mod reset)
    print("Killing character")
    for retry in range(max_retries):
            try:
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                agent_host.sendCommand('movesouth 1')
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error moving character:",e)
                    exit(1)
                else:
                    time.sleep(2.5)
    # agent_host.sendCommand('movesouth 6')
    time.sleep(1) # (let the Mod reset)