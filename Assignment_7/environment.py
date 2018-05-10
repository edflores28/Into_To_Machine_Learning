import preprocess
import random
import copy
import utilities


class Track:
    def __init__(self, filename, is_start):
        # Read the file
        self.states = preprocess.read_file(filename)
        # Set the length of the states
        self.y = int(self.states[0][0])
        # Set the width of the envronemnt
        self.x = int(self.states[0][1])
        # Determines which crashing policy to use
        self.is_start = False #TODO is_start
        # Remove the first entry since it contains L,W
        del self.states[0]
        # Create a list of starting points
        self.start = []
        # Create a list of starting points
        self.finish = []
        # Iterate through each row and convert the string
        # to a list and also find the starting points
        for row in range(len(self.states)):
            # Convert the string to a list
            self.states[row] = list(self.states[row][0])
            # If "S" is found then add the coordinates to the list
            if self.states[row].count("S") > 0:
                for entry in range(len(self.states[row])):
                    if self.states[row][entry] == "S":
                        self.start.append((entry, row))
            # If "F" is found then add the coordinates to the list
            if self.states[row].count("F") > 0:
                for entry in range(len(self.states[row])):
                    if self.states[row][entry] == "F":
                        self.finish.append((entry, row))
        # Copy the states and set the rewards
        self.rewards = copy.deepcopy(self.states)
        for row in range(len(self.rewards)):
            for col in range(len(self.rewards[row])):
                self.rewards[row][col] = self.__convert_reward(self.rewards[row][col])
        # Set the car start position
        self.car_start = random.sample(self.start, 1)[0]
        self.car_pos = self.car_start
        self.prev_pos = "S"
        self.car_state = "S"
        self.car_velocity = (0, 0)  # North
        self.finish_x = [32, 33, 34, 35]
        self.finish_y = 1
        print("X:", self.x, "Y:", self.y)
        X = utilities.transpose(self.states)
        for entry in X:
            print(entry)
        X = utilities.transpose(X)
        for entry in X:
            print(entry)
    def __convert_reward(self, value):
        if value == "#":
            return 1
        if value == "F":
            return 0
        return 1

    def __regulate_velocity(self, value):
        value = 5 if value > 5 else value
        value = -5 if value < -5 else value
        return value

    def __get_xy_crash(self):
        x = self.car_pos[0] if self.car_pos[0] < self.x else self.x - 1
        x = self.car_pos[0] if self.car_pos[0] > 0 else 0
        y = self.car_pos[1] if self.car_pos[1] < self.y else self.y - 1
        y = self.car_pos[1] if self.car_pos[1] > 0 else 0
        return x, y
    def __update_crash_position(self):
        print("Car Crashed resetting")
        if self.is_start:
            self.car_state = "S"
            self.car_pos = self.car_start
        else:
            dists = [0, 0]
            x, y = self.__get_xy_crash()
            print(self.states[y])
            x_lst = self.states[y].index(".")
            tr_states = utilities.transpose(self.states)
            print(tr_states[x])
            y_lst = tr_states[x].index(".")
            x_dist = x_lst - x
            y_dist = y_lst - y
            dists = [x_dist, y_dist]
            minimum = min(dists)
            index = dists.index(minimum)
            print(x, y, x_lst, y_lst, x_dist, y_dist)
            if index == 0:
                self.car_pos = (x, y_lst)
            else:
                self.car_pos = (x_lst, y)

        if self.prev_pos in self.car_start:
            self.car_pos = self.car_start

        self.car_state = "S"
        self.car_vel = (0, 0)

    def __determine_finish(self):
        if self.car_pos[0] in self.finish_x and self.car_pos[1] <= self.finish_y:
            self.car_state = "F"
        else:
            self.car_state = "#"

    def determine_state(self):
        temp = None
        # Determine whether the position is within the environment.
        if 0 <= self.car_pos[0] < self.x and 0 <= self.car_pos[1] < self.y:
            temp = self.states[self.car_pos[1]][self.car_pos[0]]
        # If temp is still None that means the position is outside
        # the environment
        print("TEMP", temp)
        if temp is None:
            # Check to see if the finish line was crossed
            self.__determine_finish()
        else:
            self.car_state = temp
        # Check to see if crashed
        if self.car_state == "#":
            self.__update_crash_position()

    def update_velocity(self, acceleration):
        car_vel = list(tuple(map(sum, zip(self.car_velocity, acceleration))))
        car_vel[0] = self.__regulate_velocity(car_vel[0])
        car_vel[1] = self.__regulate_velocity(car_vel[1])
        self.car_velocity = tuple(car_vel)
        print("Vel/Pos", self.car_velocity, self.car_pos)

    def update_position(self):
        self.prev_pos = self.car_pos
        # Correct the Y coordinate since a negative value
        # represents north and positive south
        velocity = [self.car_velocity[0], -self.car_velocity[1]]
        self.car_pos = tuple(map(lambda x, y: x+y, self.car_pos, velocity))
        print(self.car_pos, "POS")
    def get_position(self):
        return self.car_pos

    def get_state(self):
        return self.car_state

    def get_velocity(self):
        return self.car_velocity

    def get_track_dimensions(self):
        return [self.y, self.x]

    def get_reward(self, position):
        return(self.rewards[position[0]][position[1]])
