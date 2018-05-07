import q_table
import utilities
import random


class Q_Learn:
    def __init__(self, track, epsilon=0.1, eta=0.5, gamma=0.9):
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma
        self.total_states = track.get_track_dimensions()
        self.q_table = q_table.Q_Table(self.total_states)
        self.action = None
        self.track = track

    def __get_max_q(self, state, iskey):
        actions = self.q_table.get_q_values(state)
        max_key = max(actions, key=actions.get)
        if iskey:
            return max_key
        else:
            return actions[max_key]

    def __get_action(self):
        # If the random generated number is less than
        # the epsilon value pick an action at random
        if random.random() < self.epsilon:
            self.action = utilities.get_random_action()
        # Otherwise get the best action from the Q Table
        else:
            key = self.track.get_position() + self.track.get_velocity()
            self.action = self.__get_max_q(key, True)
        # If the random number is less than 0.2 then
        # the action (acceleration) is not applied
        if random.random() <= 0.2:
            self.action = (0, 0)

    def __calculate(self):
        current_key = self.track.get_position() + self.track.get_velocity()
        current_q = self.q_table.get_q_value(current_key, self.action)
        new_key = tuple(map(lambda x, y: x+y, current_key, self.action+(0, 0)))
        next_reward = self.track.get_reward((new_key[0], new_key[1]))
        next_max_q = self.__get_max_q(new_key, False)
        print(next_max_q, next_reward, current_q)
        new_q = current_q + self.eta*(next_reward + self.gamma*next_max_q - current_q)
        self.q_table.set_q_value(current_key, self.action, new_q)
        self.track.update_velocity(self.action)
        self.track.update_position()
    def learn(self):
        self.__get_action()
        self.__calculate()
