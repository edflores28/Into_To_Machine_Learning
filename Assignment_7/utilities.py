import random

# Contants for min/max velocity
MAX_VELOCITY = 5
MIN_VELOCITY = -5

# Control actions
# No Accel, N, S, E, W, NE, SE, NW, SW
actions = [(0, 0), (0, 1), (0, -1),
           (1, 0), (-1, 0), (1, 1),
           (1, -1), (-1, 1), (-1, -1)]


def get_actions():
    return actions


def get_random_action():
    return random.sample(actions, 1)


def get_min_velocity():
    return MIN_VELOCITY


def get_max_velocity():
    return MAX_VELOCITY
