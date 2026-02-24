# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    
    def breadth_first_search(self, initial_state, goal, game_state, verboden_wegen):
        agenda = util.Queue()
        agenda.push((initial_state, []))
        closed = set()
        muren = game_state.data.layout.walls
    

        while True:
            if agenda.is_empty():
                return 10000
            current_state, actions = agenda.pop()
        
            if current_state in goal:
                return len(actions)
            
            if current_state  not in closed:
                closed.add(current_state)
                x, y = current_state
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    next_state = (int(x + dx), int(y + dy))
                    if not muren[next_state[0]][next_state[1]] and next_state not in verboden_wegen:
                        list_actions = list(actions)
                        list_actions.append(next_state)
                        agenda.push((next_state, list_actions))


    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.get_score(successor)
        my_pos = successor.get_agent_state(self.index).get_position()


        distance_to_safety = self.get_maze_distance(self.start, my_pos)
        features['distance_to_safety'] = distance_to_safety

        verboden_wegen = []
        distance_to_defender = float("inf")
        defenders = self.get_opponents(successor)
        for defender in defenders:
            enemy_position = successor.get_agent_state(defender).get_position()
           
            if enemy_position is not None and not successor.get_agent_state(defender).is_pacman:
                verboden_wegen.append(enemy_position)
                distance = self.get_maze_distance(my_pos, enemy_position)
                if distance < distance_to_defender:
                   distance_to_defender = distance

        if len(food_list) > 0:
            closest_food = self.breadth_first_search(my_pos, food_list, game_state, verboden_wegen)
            features['distance_to_food'] = closest_food

        if distance_to_defender == float("inf"):
            features['distance_defender'] = 10
        else: 
            if distance_to_defender <= 1:
                features['distance_to_safety'] = -1000
            else:
                features['distance_defender'] = distance_to_defender 

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features['Aantal_acties'] = len(successor.get_legal_actions(self.index))

        return features

    def get_weights(self, game_state, action):
        carrying = game_state.get_agent_state(self.index).num_carrying

        if carrying < 5:
            return {'successor_score': 100, 'distance_to_food': -3, 'distance_to_safety' : 0, 'distance_defender' : 3, 'stop': -100,
                'reverse': -2, 'Aantal_acties' : 1}
        else: 
            return {'successor_score': 50, 'distance_to_food': 0, 'distance_to_safety' : -20, 'distance_defender' : 10, 'stop': -100,
                'reverse': -2, 'Aantal_acties' : 1}


class DefensiveAgent(ReflexCaptureAgent):

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        midden_x = game_state.data.layout.walls.width // 2
        midden_y = game_state.data.layout.walls.height // 2

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            if self.red:
                grens = midden_x - 1
            else:
                grens = midden_x
            
            waiting_point = self.get_maze_distance(my_pos, (grens, midden_y))
            features['ga_naar_wachtpunt'] = waiting_point
            
        

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'ga_naar_wachtpunt' : -2}
