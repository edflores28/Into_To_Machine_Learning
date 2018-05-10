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

    def __get_action(self):
        # If the random generated number is less than
        # the epsilon value pick an action at random
        if random.random() < self.epsilon:
            self.action = utilities.get_random_action()[0]
            #print("EP RAND", self.action)
        # Otherwise get the best action from the Q Table
        else:
            #print("Finding best", self.track.get_position(), self.track.get_velocity())
            key = self.track.get_position() + self.track.get_velocity()
            self.action = self.q_table.get_min_q(key, True)
            #print("action:", self.action)
        # If the random number is less than 0.2 then
        # the action (acceleration) is not applied
        if random.random() <= 0.2:
            self.action = (0, 0)
            #print("RANDOM", self.action)

    def __calculate(self):
        print("act", self.action)
        current_key = self.track.get_position() + self.track.get_velocity()
        #print("C", current_key)
        current_q = self.q_table.get_q_value(current_key, self.action)
        new_key = tuple(map(lambda x, y: x+y, current_key, self.action+(0, 0)))
        #print("N",new_key)
        next_reward = self.track.get_reward((new_key[0], new_key[1]))
        next_max_q = self.q_table.get_min_q(new_key, False)
        new_q = current_q + self.eta*(next_reward + self.gamma*next_max_q - current_q)
        self.q_table.set_q_value(current_key, self.action, new_q)
        self.track.update_velocity(self.action)
        self.track.update_position()
        self.track.determine_state()
    def learn(self):
        for i in range(5):
            self.__get_action()
            self.__calculate()

        self.q_table.print()
