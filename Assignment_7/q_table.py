import copy
import utilities


class Q_Table:
    def __init__(self, dimension):
        '''
        Initialization
        '''
        self.Q = {}
        # Obtain the action list
        actions = utilities.get_actions()
        # Default value list
        values = [0.0 for i in range(len(actions))]
        # Create a dictionary
        q_v_tmp = dict(zip(actions, values))
        # Obtain the min and max velocity
        min_vel = utilities.get_min_velocity()
        max_vel = utilities.get_max_velocity()
        # Create temporary velocity lists
        tmp_a = [i for i in range(min_vel, max_vel+1)]
        tmp_b = [list(zip([i]*len(tmp_a), tmp_a)) for i in range(min_vel, max_vel+1)]
        # Create the final velocity list
        velocity = []
        for entry in tmp_b:
            velocity += entry
        # Create the Q Table
        # Iterate through the rows and columns
        for row in range(dimension[0]):
            for column in range(dimension[1]):
                # Iterate through the velocity list
                # and create the Q table key and set
                # the default values
                for value in velocity:
                    key = (row, column) + value
                    self.Q[key] = copy.deepcopy(q_v_tmp)

    def get_q_value(self, state, action):
        '''
        This method returns the Q value
        for the given state and action
        '''
        return self.Q[state][action]

    def get_q_values(self, state):
        '''
        This method returns all the q values
        based on the state
        '''
        return self.Q[state]

    def set_q_value(self, state, action, value):
        '''
        This method set the Q value for the
        given state action and value
        '''
        self.Q[state][action] = value

    def get_min_q(self, state, iskey):
        '''
        This method find the action with the
        minimum Q value. Based on iskey
        the action or the q value is returned
        '''
        actions = self.get_q_values(state)
        min_key = min(actions, key=actions.get)
        if iskey:
            return min_key
        else:
            return actions[min_key]

    def print(self):
        '''
        Utility funtion that prints out the
        Q table entries
        '''
        for key in self.Q.keys():
            if max(self.Q[key].values()) > 0.0 or min(self.Q[key].values()) < 0.0:
                print(key, self.Q[key])
