import q_table
import utilities
import random
import matplotlib.pyplot as plt
import numpy


class Learn:
    def __init__(self, track, epsilon=0.2, alpha=0.1, gamma=0.5 is_qlearn):
        '''
        Initialization
        '''
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.total_states = track.get_track_dimensions()
        self.q_table = q_table.Q_Table(self.total_states)
        self.action = None
        self.track = track
        self.is_qlearn = qlearn

    def __get_action(self):
        '''
        This method based on the epsilon value
        generates an action
        '''
        # If the random generated number is less than
        # the epsilon value pick an action at random
        if random.random() < self.epsilon:
            self.action = utilities.get_random_action()[0]
        # Otherwise get the best action from the Q Table
        else:
            key = self.track.get_current_key()
            self.action = self.q_table.get_min_q(key, True)
        # If the random number is less than 0.2 then
        # the action (acceleration) is not applied
        if random.random() <= 0.2:
            self.action = (0, 0)

    def __calculate(self):
        '''
        This method interacts with the environment
        and updates the Q Table based on the selected
        algorithm
        '''
        # Get the q value for the current state
        current_key = self.track.get_current_key()
        current_q = self.q_table.get_q_value(current_key, self.action)
        # Update the velocity and get the next position
        # on the track
        self.track.update_velocity(self.action)
        self.track.inter_position_update()
        # Get the key, reward, and q value for the next position
        next_key = self.track.get_intermediate_key()
        next_reward = self.track.get_intemeriate_reward()
        next_q = self.q_table.get_min_q(next_key, False)
        # Calculate the new q for the current state
        # and update it
        new_q = current_q + self.alpha*(next_reward + self.gamma*next_q - current_q)
        self.q_table.set_q_value(current_key, self.action, new_q)
        # update the environment
        self.track.environment_update()

    def learn(self):
        episodes = 1000000
        seconds = []
        total = 0
        second = 0
        #print("##################################")
        while total < episodes:
            self.__get_action()
            self.__calculate()
            second += 1
            #print(second)
            if self.track.get_state() == "F":
                #print("##################################")
                total += 1
                self.track.episode_reset()
                seconds.append(second)
                second = 0
        #print(seconds)
        #self.q_table.print()
        xrange = [i for i in range(episodes)]
        x = numpy.array(xrange)
        y = numpy.array(seconds)
        plt.plot(x, y)
        plt.show()
