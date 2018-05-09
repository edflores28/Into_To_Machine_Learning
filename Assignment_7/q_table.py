import copy
import utilities


class Q_Table:
    def __init__(self, dimension):
        self.Q = {}
        actions = utilities.get_actions()
        values = [0.0 for i in range(len(actions))]
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
        try:
            return self.Q[state][action]
        except:
            print("EXCEPT:", state, action)

    def get_q_values(self, state):
        return self.Q[state]

    def set_q_value(self, state, action, value):
        self.Q[state][action] = value

    def print(self):
        for key in self.Q.keys():
            temp = max(self.Q[key].values())
            if temp > 0.0:
                print(key, self.Q[key])
            temp = min(self.Q[key].values())
            if temp < 0.0:
                print(key, self.Q[key])
