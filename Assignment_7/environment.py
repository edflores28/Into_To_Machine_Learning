import preprocess
import random
import copy
import utilities


class Track:
    def __init__(self, filename, is_start):
        '''
        Initialization
        '''
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
            utilities.search_and_add(self.states[row], "S", self.start, row)
            # If "F" is found then add the coordinates to the list
            utilities.search_and_add(self.states[row], "F", self.finish, row)
        # Copy the states and set the rewards
        self.rewards = copy.deepcopy(self.states)
        for row in range(len(self.rewards)):
            for col in range(len(self.rewards[row])):
                self.rewards[row][col] = utilities.get_reward_value(self.rewards[row][col])
        # Set initial parameters
        self.car_start = random.sample(self.start, 1)[0]
        self.car_pos = self.car_start
        self.prev_pos = "S"
        self.car_state = "S"
        self.car_velocity = (0, 0)
        self.finish_x = [32, 33, 34, 35]
        self.finish_y = 1

    def __get_bounded_value(self, value, max_val):
        '''
        This method limits the given value to 0
        and the given max value
        '''
        if value > max_val:
            return max_val - 1
        if value < 0:
            return 0
        return value

    def __is_finish(self):
        '''
        This method determines if the finish line
        was crossed
        '''
        # TODO R track may be different
        if self.inter_pos[0] in self.finish_x and self.inter_pos[1] <= self.finish_y:
            return True
        else:
            return False

    def __determine_diagnol_crash(self):
        '''
        This method checks the corners of the current
        position to check if there is a wall
        '''
        diagnol_list = []
        # Check the NW NE SW and SE areas of the position
        # and add the states to the list
        diagnol_list.append(self.states[self.car_pos[1]+1][self.car_pos[0]+1])
        diagnol_list.append(self.states[self.car_pos[1]+1][self.car_pos[0]-1])
        diagnol_list.append(self.states[self.car_pos[1]-1][self.car_pos[0]+1])
        diagnol_list.append(self.states[self.car_pos[1]-1][self.car_pos[0]-1])
        # Check to see if there is a wall and update the position
        if "#" in diagnol_list:
            self.car_pos = self.prev_pos

    def __negative_lookahead(self, lookahead, new_pos, is_y):
        '''
        This method examines the environment from the old to new
        position for a wall and sets the position of the
        first wall
        This is used for the W and S movements
        '''
        # Reverse the list
        reversed_list = lookahead[::-1]
        # Obtain the position
        car_pos = self.car_pos[0] if not is_y else self.car_pos[1]
        # Calculate the new and old positions based
        # on the reversed list
        new = len(lookahead) - new_pos
        old = len(lookahead) - car_pos
        # Only examine what we from old to new
        reversed_list = reversed_list[old:new]
        # Find the wall and get the index
        state_idx = utilities.find_crash(reversed_list)
        # If that state_idx is None then there was
        # no wall encountered
        if state_idx is None:
            return new_pos, 0
        # Otherwise return the position of the wall
        else:
            return (car_pos - state_idx - 1), 1

    def __positive_lookahead(self, lookahead, new_pos, is_y):
        '''
        This method examines the environment from the old to new
        position for a wall and sets the position of the
        first wall
        This is used for the N and E movements
        '''
        car_pos = self.car_pos[0] if not is_y else self.car_pos[1]
        sliced = lookahead[car_pos:new_pos+1]
        state_idx = utilities.find_crash(sliced)
        # If that state_idx is None then there was
        # no wall encountered
        if state_idx is None:
            return new_pos, 0
        # Otherwise return the position of the wall
        else:
            return (car_pos + state_idx), -1

    def __do_xy_positions(self, new_pos, is_y):
        '''
        This method performs the lookahead processing
        '''
        # Get the indices
        car_pos = self.car_pos[1] if is_y else self.car_pos[0]
        list_idx = self.car_pos[0] if is_y else self.car_pos[1]
        # Determine the difference
        diff = new_pos - car_pos
        lookahead = []
        # If this is the y coordinate then transpose the states
        # and get the row
        if is_y:
            lookahead = utilities.transpose(self.states)[list_idx]
        # Otherwise just get the row
        else:
            lookahead = self.states[list_idx]
        update = 0
        # Depending on the diff choose the right method
        if diff > 0:
            new_pos, update = self.__positive_lookahead(lookahead, new_pos, is_y)
        elif diff < 0:
            new_pos, update = self.__negative_lookahead(lookahead, new_pos, is_y)
        # Return the position and update values
        return new_pos, update

    def update_velocity(self, acceleration):
        '''
        This method updates the velocity based
        on the given acceleration
        '''
        # Correct the y component. N is actually -1
        # and S is +1
        acceleration = (acceleration[0], -acceleration[1])
        # Add the acceleration and velocity
        car_vel = list(tuple(map(sum, zip(self.car_velocity, acceleration))))
        # Make sure that the bew velocity is within bounds
        car_vel[0] = utilities.regulate_velocity(car_vel[0])
        car_vel[1] = utilities.regulate_velocity(car_vel[1])
        # Set the velocity
        self.car_velocity = tuple(car_vel)

    def inter_position_update(self):
        '''
        This method determines the next position
        '''
        # Add the velocity and position
        new_pos = list(map(lambda x, y: x+y, self.car_pos, self.car_velocity))
        # Make sure the new position is within the environment
        # limits
        new_pos[0] = self.__get_bounded_value(new_pos[0], self.x)
        new_pos[1] = self.__get_bounded_value(new_pos[1], self.y)
        # Determine the new x and y positions
        x, xup = self.__do_xy_positions(new_pos[0], False)
        y, yup = self.__do_xy_positions(new_pos[1], True)
        # Save the intermediate positions and updates
        self.inter_pos = (x, y)
        self.inter_updates = (xup, yup)

    def environment_update(self):
        '''
        This method updates the environment
        '''
        # Save off the car position
        self.prev_pos = self.car_pos
        # Get the state of the intermediate location
        state = self.states[self.inter_pos[1]][self.inter_pos[0]]
        # Check to see if the car crashed
        if state == "#":
            # If is start is true then the car will start
            # at the starting line.
            if self.is_start:
                self.car_pos = self.car_start
                self.car_state = "S"
            self.car_velocity = (0, 0)
        # Otherwise add the intermediate updates to the intermediate
        # position to get the closest track location.
        # If the car didnt crash then the intermediate position
        # will already have the next valid position and the
        # updates will be 0
        self.car_pos = list(map(lambda x, y: x+y, self.inter_pos, self.inter_updates))
        # Set the finished state if the car is at or passed the finish line
        if self.__is_finish():
            self.car_state = "F"
        else:
            self.car_state = self.states[self.car_pos[1]][self.car_pos[0]]
        # If the new position is still a wall then there is a good chance
        # that the wall is diagnol
        if self.car_state == "#":
            self.__determine_diagnol_crash()
        #print("Pos:",self.car_pos,"Vel",self.car_velocity, "")

    def get_state(self):
        '''
        This method gets the cars current state
        '''
        return self.car_state

    def get_current_key(self):
        '''
        This method returns the key for the current
        position
        '''
        return tuple(list(self.car_pos)[::-1]) + self.car_velocity

    def get_track_dimensions(self):
        '''
        This method returns the dimension of the track
        '''
        return [self.y, self.x]

    def get_intermediate_key(self):
        '''
        This method returns the intermediate key
        which is also the next position
        '''
        return tuple(list(self.inter_pos)[::-1]) + self.car_velocity

    def get_intemeriate_reward(self):
        '''
        This method returns the reward for the
        next state
        '''
        # If the finish line is crossed for the next
        # state then return 0
        if self.__is_finish():
            return 0
        # Otherwise obtain the state
        else:
            return(self.rewards[self.inter_pos[1]][self.inter_pos[0]])

    def episode_reset(self):
        self.car_pos = random.sample(self.start, 1)[0]
        self.car_start = self.car_pos
        self.car_velocity = (0, 0)
