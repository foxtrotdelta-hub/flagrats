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
        self.eaten_dot = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

        walls = game_state.data.layout.walls
        mid_x = walls.width // 2
        border_x = mid_x - 1 if self.red else mid_x
        self.border_points = [(border_x, y) for y in range(walls.height) if not walls[border_x][y]]

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
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
    
    def is_winning(self, game_state):
        score = self.get_score(game_state)
        return score > 0 if self.red else score < 0

    def breadth_first_search(self, initial_state, goal, game_state, forbidden_paths):
        agenda = util.Queue()
        agenda.push((initial_state, []))
        closed = set()
        walls = game_state.data.layout.walls

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
                    if not walls[next_state[0]][next_state[1]] and next_state not in forbidden_paths:
                        list_actions = list(actions)
                        list_actions.append(next_state)
                        agenda.push((next_state, list_actions)) 

    def determine_agents_role(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman]

        if my_state.scared_timer > 0:
            return 'Offense'
        
        if len(invaders) == 0:
            if self.is_winning(game_state) and game_state.data.timeleft < 300:
                return'Defense'
            return 'Offense'
        
        teammates = self.get_team(game_state)
        teammate = [i for i in teammates if i != self.index][0]
        teammate_state = game_state.get_agent_state(teammate)
        my_pos = my_state.get_position()
        teammate_pos = teammate_state.get_position()

        if my_pos == self.start:
            return 'Defense'
        if teammate_pos == self.start:
            return 'Offense'
        if my_state.num_carrying < 4 and teammate_state.num_carrying >= 4:
            return 'Offense'
        
        existing_invaders = [invader for invader in invaders if invader.get_position() is not None]
        if len(existing_invaders) > 0:
            my_min_distance = min([self.get_maze_distance(my_pos, invader.get_position()) for invader in existing_invaders])
            teammate_min_position = min([self.get_maze_distance(teammate_pos, invader.get_position()) for invader in existing_invaders])
            if my_min_distance < teammate_min_position:
                return 'Defense'
            if my_min_distance > teammate_min_position:
                return 'Offense'
            
        if self.index < teammate:
            return 'Defense'
        else:
            return 'Offense'
    
    def get_forbidden_paths(self, game_state, successor, my_pos):
        forbidden_paths = []
        distance_to_defender = float("inf")
        defenders = self.get_opponents(game_state)

        for defender in defenders:
            enemy_state = successor.get_agent_state(defender)
            enemy_position = enemy_state.get_position()

            if enemy_position is None:
                previous_state = self.get_previous_observation()
                if previous_state is not None:
                    previous_enemy_state = previous_state.get_agent_state(defender)
                    if previous_enemy_state.get_position() is not None:
                        enemy_position = previous_enemy_state.get_position()
                        enemy_state = previous_enemy_state
           
            if enemy_position is not None and not enemy_state.is_pacman and enemy_state.scared_timer == 0:
                forbidden_paths.append(enemy_position)
                x, y = enemy_position
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    forbidden_paths.append((int(x + dx), int(y + dy)))
                distance = self.get_maze_distance(my_pos, enemy_position)
                if distance < distance_to_defender:
                   distance_to_defender = distance
        return forbidden_paths, distance_to_defender
    
    def calculate_distance_to_food(self, game_state, my_pos, food_list, forbidden_paths):
        if len(food_list) == 0:
            return 0
        teammates = self.get_team(game_state)
        teammate = [mate for mate in teammates if mate != self.index][0]
        mid_y = game_state.data.layout.walls.height // 2
        
        if self.index > teammate:
            target_food = [food for food in food_list if food[1] >= mid_y]
        else:
            target_food = [food for food in food_list if food[1] < mid_y]

        if len(target_food) == 0:
            target_food = food_list
        
        return self.breadth_first_search(my_pos, target_food, game_state, forbidden_paths)
    
    def calculate_distance_to_capsule(self, games_state, my_pos, forbidden_paths):
        capsules = self.get_capsules(games_state)
        return self.breadth_first_search(my_pos, capsules, games_state, forbidden_paths)
    
    def calculate_distance_to_waitinpoint(self, game_state, my_pos):
        if self.eaten_dot is not None:
                point_to_go = self.eaten_dot
        else:
            mid_x = game_state.data.layout.walls.width // 2
            mid_y = game_state.data.layout.walls.height // 2 
            border_x = mid_x -1 if self.red else mid_x
            point_to_go = (border_x, mid_y)
            walls = game_state.data.layout.walls
            if walls[border_x][mid_y]:
                point_to_go = (border_x, mid_y - 1)
        
        return self.get_maze_distance(my_pos, point_to_go)

        
    def get_offensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()

        features['successor_score'] = -len(food_list) 

        my_pos = successor.get_agent_state(self.index).get_position()
        current_position = game_state.get_agent_state(self.index).get_position()

        features['suicide'] = 1 if my_pos == self.start and current_position != self.start else 0

        if successor.get_agent_state(self.index).is_pacman:
            features['distance_to_safety'] = min([self.get_maze_distance(my_pos, grens_punt) for grens_punt in self.border_points])
        else:
            features['distance_to_safety'] = 0
    
        forbidden_paths, distance_to_defender = self.get_forbidden_paths(game_state, successor, my_pos)

        features['distance_to_food'] = self.calculate_distance_to_food(game_state, my_pos, food_list, forbidden_paths)
        features['distance_to_capsule'] = self.calculate_distance_to_capsule(game_state, my_pos, forbidden_paths)

        if distance_to_defender == float("inf"):
            features['distance_defender'] = 10
        else: 
            if distance_to_defender <= 1:
                features['distance_defender'] = -10000
            else:
                features['distance_defender'] = distance_to_defender 

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        features['Aantal_acties'] = len(successor.get_legal_actions(self.index))

        return features
    
    def get_offensive_weights(self, game_state, action):
        carrying = game_state.get_agent_state(self.index).num_carrying
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        dangerous_enemies = [enemy for enemy in enemies if not enemy.is_pacman and enemy.get_position() is not None and enemy.scared_timer == 0 ]
        my_pos = game_state.get_agent_state(self.index).get_position()
        chased = False
        capsules = self.get_capsules(game_state)


        if len(enemies) > 0:
            distances = [self.get_maze_distance(my_pos, enemie.get_position()) for enemie in dangerous_enemies]
            if len(distances) > 0 and min(distances) <= 5:
                chased = True

        if chased:
            distance_to_safety = min([self.get_maze_distance(my_pos, grens_punt) for grens_punt in self.border_points])
            if len(capsules) > 0:
                dist_to_closest_capsule = min([self.get_maze_distance(my_pos, capsule) for capsule in capsules])
                if distance_to_safety < dist_to_closest_capsule and carrying > 0:
                    return {'successor_score': 0, 'distance_to_food': 0, 'distance_to_safety' : -80, 'distance_defender' : 100, 'stop': -100,
                    'reverse': -2, 'Aantal_acties' : 20, 'distance_to_capsule' : 0, 'suicide' : -9999}
                else: return {'successor_score': 0, 'distance_to_food': 0, 'distance_to_safety' : 0, 'distance_defender' : 100, 'stop': -100,
                    'reverse': -2, 'Aantal_acties' : 20, 'distance_to_capsule' : -80, 'suicide' : -9999}
                
        if carrying >= 6 and not chased:
            return {'successor_score': 50, 'distance_to_food': 0, 'distance_to_safety' : -5, 'distance_defender' : 10, 'stop': -100,
                'reverse': -2, 'Aantal_acties' : 1,  'distance_to_capsule' : 0}
        if carrying > 0 and game_state.data.timeleft < 100:
            return {'successor_score': 50, 'distance_to_food': 0, 'distance_to_safety' : -20, 'distance_defender' : 10, 'stop': -100,
                'reverse': -2, 'Aantal_acties' : 1,  'distance_to_capsule' : 0}
        else: 
            return {'successor_score': 100, 'distance_to_food': -3, 'distance_to_safety' : 0, 'distance_defender' : 3, 'stop': -100,
                'reverse': -2, 'Aantal_acties' : 1,  'distance_to_capsule' : 0}
        
    def get_defensive_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        current_position  = game_state.get_agent_position(self.index)

        if self.eaten_dot == current_position:
            self.eaten_dot = None

        current_enemies = [game_state.get_agent_state(i) for i in self.get_opponents(successor)]
        current_invaders = [a for a in current_enemies if a.is_pacman and a.get_position() is not None]

        if len(current_invaders) > 0:
            self.eaten_dot = None
        else:
            previous_state = self.get_previous_observation()
            if previous_state is not None:
                previous_food = self.get_food_you_are_defending(previous_state).as_list()
                current_food = self.get_food_you_are_defending(game_state).as_list()

                if len(previous_food) > len(current_food):
                        eaten_dots = list(set(previous_food) - set(current_food))
                        if eaten_dots:
                            self.eaten_dot = eaten_dots[0]

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        
        if len(invaders) > 0:
            features['invader_distance'] = min([self.get_maze_distance(my_pos,invader.get_position()) for invader in invaders])
        else:
            features['go_to_waiting_point'] = self.calculate_distance_to_waitinpoint(game_state, my_pos)

        features['avoid_ghost'] = 0
        if my_state.is_pacman:
            defenders = [agent for agent in enemies if not agent.is_pacman and agent.scared_timer is None and agent.get_position() is not None]
            if len(defenders) > 0:
                minimum_distance = min([self.get_maze_distance(my_pos, agent.get_position) for agent in defenders])
                if minimum_distance <= 1:
                    features['avoid_ghost'] = 10000
                else: features['avoid_ghost'] = 10

        

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features
    
    def get_defensive_weights(self, game_state, action):
        if game_state.get_agent_state(self.index).scared_timer > 0:
            return {'num_invaders': 0, 'on_defense': 100, 'invader_distance': 10, 'stop': -100, 'reverse': -2, 'go_to_waiting_point' : 0, 'avoid_ghost' : -1}
        else:
            return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'go_to_waiting_point' : -2, 'avoid_ghost' : -1}
        

class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
            if self.determine_agents_role(game_state) == 'Defense':
                return self.get_defensive_features(game_state, action)
            
            return self.get_offensive_features(game_state, action)
            
        
    def get_weights(self, game_state, action):
            if self.determine_agents_role(game_state) == 'Defense':
                return self.get_defensive_weights(game_state, action)
            
            return self.get_offensive_weights(game_state, action)
        
         

class DefensiveAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        
            if self.determine_agents_role(game_state) == 'Defense':
                return self.get_defensive_features(game_state, action)
        
            return self.get_offensive_features(game_state, action)
            
                
        
        
    def get_weights(self, game_state, action):
            if self.determine_agents_role(game_state) == 'Defense':
                return self.get_defensive_weights(game_state, action)
            
            return self.get_offensive_weights(game_state, action)
            
        
       

    