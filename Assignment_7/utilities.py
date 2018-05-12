import random

'''
This package provies utility functions
'''

# Contants for min/max velocity
MAX_VELOCITY = 5
MIN_VELOCITY = -5

# Control actions
# No Accel, N, S, E, W, NE, SE, NW, SW
actions = [(0, 0), (0, 1), (0, -1),
           (1, 0), (-1, 0), (1, 1),
           (1, -1), (-1, 1), (-1, -1)]


def get_actions():
    '''
    This method returns all the available actions
    '''
    return actions


def get_random_action():
    '''
    This method returns a random action
    '''
    return random.sample(actions, 1)


def get_min_velocity():
    '''
    This method returns the minimum velocity
    '''
    return MIN_VELOCITY


def get_max_velocity():
    '''
    This method returns the maximum velocity
    '''
    return MAX_VELOCITY


def regulate_velocity(velocity):
    '''
    This method regulates the velocity
    between MIN_VELOCITY and MAX_VELOCITY
    '''
    if velocity > MAX_VELOCITY:
        return MAX_VELOCITY
    if velocity < MIN_VELOCITY:
        return MIN_VELOCITY
    return velocity


def find_crash(ahead):
    '''
    This method determines if a wall
    is present
    '''
    try:
        return ahead.index("#")
    except:
        return None


def get_reward_value(pattern):
    '''
    This method returns the reward based
    on the state
    '''
    if pattern == "F":
        return 0
    return 1


def search_and_add(search_list, value, add_list, idx):
    if search_list.count(value) > 0:
        for entry in range(len(search_list)):
            if search_list[entry] == value:
                add_list.append((entry, idx))

def transpose(data_list):
    '''
    Transpose the datalist from rows to columns or
    columns to rows
    '''
    return list(map(list, zip(*data_list)))
