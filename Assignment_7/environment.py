import preprocess
import random
import copy


class Track:
    def __init__(self, filename):
        # Read the file
        self.states = preprocess.read_file(filename)
        # Set the length of the states
        self.rows = int(self.states[0][0])
        # Set the width of the envronemnt
        self.columns = int(self.states[0][1])
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
                        self.start.append((row, entry))
            # If "F" is found then add the coordinates to the list
            if self.states[row].count("F") > 0:
                for entry in range(len(self.states[row])):
                    if self.states[row][entry] == "F":
                        self.finish.append((row, entry))
        # Copy the states and set the rewards
        self.rewards = copy.deepcopy(self.states)
        for row in range(len(self.rewards)):
            for col in range(len(self.rewards[row])):
                self.rewards[row][col] = self.__convert_reward(self.rewards[row][col])
        # Set the car start position
        self.car_start = random.sample(self.start, 1)[0]
        self.car_pos = self.car_start
        self.car_state = "S"
        self.car_velocity = (1, 0)  # North
        self.finish = [32, 33, 34, 35]
        print("Start:",self.car_pos)
    def __convert_reward(self, value):
        if value == "#":
            return -1
        if value == "F":
            return 0
        return -1

    def __regulate_velocity(self, value):
        value = 5 if value > 5 else value
        value = -5 if value < -5 else value
        return value

    def __determine_crash(self):
        # TODO need to check finish line
        if self.car_state == "#":
            #print("Car Crashed resetting")
            self.car_state = "S"
            self.car_pos = self.car_start
            #print("Car start", self.car_pos)
            self.car_vel = (0, 0)

    def update_velocity(self, acceleration):

        # Since the environment is indexed by (y,x) and
        # the accel is (x,y) reverse accel to (y.x)

        car_vel = list(tuple(map(sum, zip(self.car_velocity, acceleration))))
        car_vel[0] = self.__regulate_velocity(car_vel[0])
        car_vel[1] = self.__regulate_velocity(car_vel[1])
        self.car_velocity = tuple(car_vel)
        #print("Vel/Pos", self.car_velocity, self.car_pos)
    def update_position(self):
        velocity = [-self.car_velocity[1], self.car_velocity[0]]
        self.car_pos = tuple(map(lambda x, y: x+y, self.car_pos, velocity))
        #print("New position:", self.car_pos)
        if 0 <= self.car_pos[0] < self.rows and 0 <= self.car_pos[1] < self.columns:
            self.car_state = self.states[self.car_pos[0]][self.car_pos[1]]
            #print("hurr")
        # elif self.car_pos[0] <= 1 and self.car_pos[1] in self.finish:
        #     self.car_pos
        else:
            #print("error")
            self.car_state = "#"
        self.__determine_crash()

    def get_position(self):
        return self.car_pos

    def get_state(self):
        return self.car_state

    def get_velocity(self):
        return self.car_velocity

    def get_track_dimensions(self):
        return [self.rows, self.columns]

    def get_reward(self, position):
        return(self.rewards[position[0]][position[1]])
