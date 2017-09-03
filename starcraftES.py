import sys
import random
import cPickle as pickle
import numpy as np
from evostra import EvolutionStrategy

import gflags as flags
from pysc2.env import sc2_env
from pysc2.lib import actions as sc2_actions
from pysc2.env import environment
from pysc2.lib import features
from pysc2.lib import actions

# Starcraft Hyperparameters
step_mul = 8
steps = 400
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

FLAGS = flags.FLAGS

class Model(object):

    def __init__(self):
        self.weights = [np.random.randn(4096, 20), np.random.randn(20, 4), np.random.randn(1, 20)]

    def predict(self, inp):
        #out = np.expand_dims(inp.flatten(), 0)
        out = inp.flatten()
        out = np.dot(out, self.weights[0]) + self.weights[-1]
        out = np.dot(out, self.weights[1])
        return out[0]
        #return random.choice([0,1,2,3])

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

class Agent:

    AGENT_HISTORY_LENGTH = 1
    NUM_OF_ACTIONS = 4
    POPULATION_SIZE = 20
    EPS_AVG = 1
    SIGMA = 0.1
    LEARNING_RATE = 0.03
    INITIAL_EXPLORATION = 0.0
    FINAL_EXPLORATION = 0.0
    EXPLORATION_DEC_STEPS = 100000

    def __init__(self):
        FLAGS(sys.argv)
        self.model = Model()
        self.env =  sc2_env.SC2Env("CollectMineralShards", step_mul=step_mul, visualize=True, game_steps_per_episode = 1600)
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
        self.exploration = self.INITIAL_EXPLORATION
        
    def get_predicted_action(self, sequence):
        #return self.model.predict(np.array(sequence))
        prediction = self.model.predict(np.array(sequence))
        return np.argmax(prediction)
        #x = np.argmax(prediction)
        #return 119 if x == 1 else None    


    def get_observation(self):
        state = self.env.getGameState()
        return np.array(state.values())
    
    def train(self, iterations):
        self.es.run(iterations, print_step=1)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in xrange(self.EPS_AVG):
            path_memory = np.zeros((64,64))
            obs = self.env.reset()
            # Select all marines first
            obs = self.env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
            
            player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
            #sequence = [obs]*self.AGENT_HISTORY_LENGTH
            done = False
            
            screen = player_relative + path_memory
            #sequence = screen
            player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            player = [int(player_x.mean()), int(player_y.mean())]

            if(player[0]>32):
                screen = shift(LEFT, player[0]-32, screen)
            elif(player[0]<32):
                screen = shift(RIGHT, 32 - player[0], screen)

            if(player[1]>32):
                screen = shift(UP, player[1]-32, screen)
            elif(player[1]<32):
                screen = shift(DOWN, 32 - player[1], screen)

            while not done:
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = random.choice([0,1,2,3])
                else:
                    #action = self.get_predicted_action(sequence)
                    action = self.get_predicted_action(screen)

                coord = [player[0], player[1]]
                path_memory_ = np.array(path_memory, copy=True)
                reward = 0 

                if(action == 0): #UP
                    if(player[1] >= 16):
                      coord = [player[0], player[1] - 16]
                      path_memory_[player[1] - 16 : player[1], player[0]] = -1
                    elif(player[1] > 0):
                      coord = [player[0], 0]
                      path_memory_[0 : player[1], player[0]] = -1
            
                elif(action == 1): #DOWN
                    if(player[1] <= 47):
                      coord = [player[0], player[1] + 16]
                      path_memory_[player[1] : player[1] + 16, player[0]] = -1
                    elif(player[1] > 47):
                      coord = [player[0], 63]
                      path_memory_[player[1] : 63, player[0]] = -1
            
                elif(action == 2): #LEFT
                    if(player[0] >= 16):
                      coord = [player[0] - 16, player[1]]
                      path_memory_[player[1], player[0] - 16 : player[0]] = -1
                    elif(player[0] < 16):
                      coord = [0, player[1]]
                      path_memory_[player[1], 0 : player[0]] = -1
                 
                elif(action == 3): #RIGHT
                    if(player[0] <= 47):
                      coord = [player[0] + 16, player[1]]
                      path_memory_[player[1], player[0] : player[0] + 16] = -1
                    elif(player[0] > 47):
                      coord = [63, player[1]]
                      path_memory_[player[1], player[0] : 63] = -1
                       
                path_memory = np.array(path_memory_)

                if _MOVE_SCREEN not in obs[0].observation["available_actions"]:
                    obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

                new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]                
                obs = self.env.step(actions=new_action)
                new_screen = player_relative + path_memory

                player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
                player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                player = [int(player_x.mean()), int(player_y.mean())]
                if(player[0]>32):
                    new_screen = shift(LEFT, player[0]-32, new_screen)
                elif(player[0]<32):
                    new_screen = shift(RIGHT, 32 - player[0], new_screen)

                if(player[1]>32):
                    new_screen = shift(UP, player[1]-32, new_screen)
                elif(player[1]<32):
                    new_screen = shift(DOWN, 32 - player[1], new_screen)
                
                reward = obs[0].reward
                reward += random.choice([0.0001, -0.0001])
                total_reward += reward
                #sequence = sequence[1:]
                #sequence.append(screen)
                screen = new_screen
                done = obs[0].step_type == environment.StepType.LAST

                if done:
                    obs = self.env.reset()
                    player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
                    screen = player_relative + path_memory
                    player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
                    player = [int(player_x.mean()), int(player_y.mean())]

                    if(player[0]>32):
                      screen = shift(LEFT, player[0]-32, screen)
                    elif(player[0]<32):
                      screen = shift(RIGHT, 32 - player[0], screen)

                    if(player[1]>32):
                      screen = shift(UP, player[1]-32, screen)
                    elif(player[1]<32):
                      screen = shift(DOWN, 32 - player[1], screen)

                    self.env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

        return total_reward/self.EPS_AVG


    def play(self, episodes):
        self.env.display_screen = True
        self.model.set_weights(self.es.weights)
        for episode in xrange(episodes):
            self.env.reset_game()
            obs = self.get_observation()
            sequence = [obs]*self.AGENT_HISTORY_LENGTH
            done = False
            score = 0
            while not done:
                action = self.get_predicted_action(sequence)
                reward = self.env.act(action)
                obs = self.get_observation()
                sequence = sequence[1:]
                sequence.append(obs)
                done = self.env.game_over()
                if self.game.getScore() > score:
                    score = self.game.getScore()
                    print "score: %d" % score
        self.env.display_screen = False

    def load(self, filename='weights.pkl'):
        with open(filename,'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)


UP, DOWN, LEFT, RIGHT = 'up', 'down', 'left', 'right'

def shift(direction, number, matrix):
  ''' shift given 2D matrix in-place the given number of rows or columns
      in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
  '''
  if direction in (UP):
    matrix = np.roll(matrix, -number, axis=0)
    matrix[number:,:] = -2
    return matrix
  elif direction in (DOWN):
    matrix = np.roll(matrix, number, axis=0)
    matrix[:number,:] = -2
    return matrix
  elif direction in (LEFT):
    matrix = np.roll(matrix, -number, axis=1)
    matrix[:,number:] = -2
    return matrix
  elif direction in (RIGHT):
    matrix = np.roll(matrix, number, axis=1)
    matrix[:,:number] = -2
    return matrix
  else:
    return matrix


# TODO
# model
# path_memory
