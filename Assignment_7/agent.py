import q_table
import utilities
import random
import matplotlib.pyplot as plt
import numpy

MAX_PRINT = 10

class MDP:
    def __init__(self, track, is_qlearn, episodes, epsilon=0.1, alpha=0.5, gamma=0.9):
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
        self.is_qlearn = is_qlearn
        self.episodes = episodes
        self.is_print = True

    def algorithm_action(self):
        '''
        This method gets the action
        based on the flag
        '''
        if not self.is_qlearn:
            self.__get_action()

    def __calculate(self, next_q):
        return self.current_q + self.alpha*(self.next_reward + (self.gamma*next_q) - self.current_q)

    def __get_action(self):
        '''
        This method based on the epsilon value
        generates an action
        '''
        # If the random generated number is less than
        # the epsilon value pick an action at random
        if random.random() < self.epsilon:
            self.action = utilities.get_random_action()[0]
            if self.is_print:
                print("A random action was generated\n")
        # Otherwise get the best action from the Q Table
        else:
            key = self.track.get_current_key()
            self.action = self.q_table.get_max_q(key, True)
            if self.is_print:
                print("An action was retreived from the Q Table\n")
        # If the random number is less than 0.2 then
        # the action (acceleration) is not applied
        if random.random() <= 0.2:
            self.action = (0, 0)
            if self.is_print:
                print("Action Reset\n")
        if self.is_print:
            print("The action is", self.action, "\n")

    def __common_processing(self):
        # Get the q value for the current state
        self.current_key = self.track.get_current_key()
        self.current_q = self.q_table.get_q_value(self.current_key, self.action)
        # Update the velocity and get the next position
        # on the track
        self.track.update_velocity(self.action)
        self.track.inter_position_update()
        # Get the key, reward, and q value for the next position
        self.next_key = self.track.get_intermediate_key()
        self.next_reward = self.track.get_intemeriate_reward()

    def __q_learn(self):
        '''
        This method interacts with the environment
        and updates the Q Table based on Q learning algorithm
        '''
        # Obtain an action
        self.__get_action()
        # Perform common processing
        self.__common_processing()
        # Get the next q value
        next_q = self.q_table.get_max_q(self.next_key, False)
        # Calculate the new q for the current state
        # and update it
        new_q = self.__calculate(next_q)
        self.q_table.set_q_value(self.current_key, self.action, new_q)
        if self.is_print:
            print("Finished updating the Q value", self.current_key)
        # update the environment
        self.track.environment_update()

    def __sarsa(self):
        '''
        This method interacts with the environment
        and updates the Q Table based on SARSA algorithm
        '''
        # Perform common processing
        self.__common_processing()
        # Save off the previous action
        prev_action = self.action
        # Get a new action
        self.__get_action()
        # Get the next q value
        next_q = self.q_table.get_q_value(self.next_key, self.action)
        # Calculate the new q for the current state
        # and update it
        new_q = self.__calculate(next_q)
        self.q_table.set_q_value(self.current_key, prev_action, new_q)
        # update the environment
        self.track.environment_update()

    def __simulate(self, total_episodes):
        seconds = []
        rewards = []
        total = 0
        second = 0
        total_rewards = 0
        self.algorithm_action()
        while total < total_episodes:
            if self.is_qlearn:
                self.__q_learn()
            else:
                self.__sarsa()
            # Turn off printing
            if second == 200:
                self.is_print = False
                self.track.set_print(False)
            total_rewards += self.next_reward
            second += 1
            if self.track.get_state() == "F":
                total += 1
                self.track.episode_reset()
                seconds.append(second)
                rewards.append(total_rewards)
                total_rewards = 0
                second = 0
                self.algorithm_action()
        self.is_print = True
        self.track.set_print(True)
        return seconds, rewards

    def learn(self):
        seconds, rewards = self.__simulate(self.episodes)

        print("Learning finished.")
        print("The fastest time was", min(seconds), "at episode", seconds.index(min(seconds)))
        # xrange = [i for i in range(self.episodes)]
        # x = numpy.array(xrange)
        # y = numpy.array(seconds)
        # yr = numpy.array(rewards)
        #
        # text = "Q Learning" if self.is_qlearn else "SARSA"
        # plt.subplot(2, 1, 1)
        # plt.plot(x, y)
        # plt.title(text+" Algorithm")
        # plt.ylabel('Time')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x, yr)
        # plt.xlabel('Episodes')
        # plt.ylabel('Rewards')
        # plt.show()

    def test(self):
        episodes = 20
        seconds, rewards = self.__simulate(episodes)
        print("Testing finished.")
        print("The fastest time was", min(seconds), "at episode", seconds.index(min(seconds)))
        # xrange = [i for i in range(episodes)]
        # x = numpy.array(xrange)
        # y = numpy.array(seconds)
        # yr = numpy.array(rewards)
        #
        # text = "Q Learning" if self.is_qlearn else "SARSA"
        # plt.subplot(2, 1, 1)
        # plt.plot(x, y)
        # plt.title(text+" Algorithm")
        # plt.ylabel('Time')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x, yr)
        # plt.xlabel('Episodes')
        # plt.ylabel('Rewards')
        # plt.show()
