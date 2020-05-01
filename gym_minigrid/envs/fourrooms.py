#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class FourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, max_steps=100, grid_size=19):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=max_steps)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info

register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)


class FourRoomsEnvMax200(FourRoomsEnv):
    def __init__(self):
        super().__init__(max_steps=200)


register(
    id='MiniGrid-FourRooms-max200-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnvMax200'
)


class SmallFourRoomsEnv(FourRoomsEnv):
    def __init__(self):
        super().__init__(grid_size=11)


register(
    id='MiniGrid-SmallFourRooms-v0',
    entry_point='gym_minigrid.envs:SmallFourRoomsEnv'
)


class FourRoomsEnvMax200Rp5(FourRoomsEnv):
    def __init__(self):
        super().__init__(max_steps=200)

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.5 * (self.step_count / self.max_steps)


register(
    id='MiniGrid-FourRooms-max200-rp5-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnvMax200'
)


class FourRoomsDiffRoomEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, room_str="", max_steps=100):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.room_str = room_str
        super().__init__(grid_size=19, max_steps=max_steps)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)

        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj_diff_room(Goal())

        self.mission = 'Reach the goal'

    def place_obj_diff_room(self,
                            obj,
                            top=None,
                            size=None,
                            reject_fn=None,
                            max_tries=math.inf
                            ):
        """
        Place an object at an empty position in the grid
        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Don't place the object where the agent is
            if np.array_equal(pos, self.agent_pos):
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            # Check if the agent and the goal are in the same room
            if self.room_str == "same":
                if self.get_room(self.agent_pos) != self.get_room(pos):
                    continue

            if self.room_str == "adj":
                if not self.check_adj_room(self.get_room(self.agent_pos), self.get_room(pos)):
                    continue

            if self.room_str == "cross":
                if not self.check_cross_room(self.get_room(self.agent_pos), self.get_room(pos)):
                    continue

            if self.room_str == "not_same":
                if self.get_room(self.agent_pos) == self.get_room(pos):
                    continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def check_adj_room(self, r1, r2):
        if r1 == "room1" and (r2 == "room4" or r2 == "room2"):
            return True
        elif r1 == "room2" and (r2 == "room1" or r2 == "room3"):
            return True
        elif r1 == "room3" and (r2 == "room2" or r2 == "room4"):
            return True
        elif r1 == "room4" and (r2 == "room3" or r2 == "room1"):
            return True
        # Following condition need to be removed, this has been added because there is no way to change agents location
        # and when agent initiates in door there is trouble
        if r1 == "door":
            self.place_agent()
            return False
        return False


    def check_cross_room(self, r1, r2):
        if r1 == "room1" and r2 == "room3":
            return True
        elif r1 == "room2" and r2 == "room4":
            return True
        elif r1 == "room3" and r2 == "room1":
            return True
        elif r1 == "room4" and r2 == "room2":
            return True
        # Following condition need to be removed, this has been added because there is no way to change agents location
        # and when agent initiates in door there is trouble
        if r1 == "door":
            self.place_agent()
            return False
        return False

    @staticmethod
    def get_room(pos):
        x, y = pos
        if x < 9 and y < 9:
            return "room1"
        elif x < 9 and y > 9:
            return "room4"
        elif x > 9 and y < 9:
            return "room2"
        elif x > 9 and y > 9:
            return "room3"
        else:
            return "door"

    @staticmethod
    def get_distance(p1, p2):
        return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class FourRoomSameRoomEnv(FourRoomsDiffRoomEnv):
    def __init__(self):
        super().__init__(room_str="same")


class FourRoomAdjRoomEnv(FourRoomsDiffRoomEnv):
    def __init__(self):
        super().__init__(room_str="adj")


class FourRoomCrossRoomEnv(FourRoomsDiffRoomEnv):
    def __init__(self):
        super().__init__(room_str="cross")


class FourRoomNotSameRoomEnv(FourRoomsDiffRoomEnv):
    def __init__(self):
        super().__init__(room_str="not_same")


register(
    id='MiniGrid-FourRooms-same-v0',
    entry_point='gym_minigrid.envs:FourRoomSameRoomEnv'
)

register(
    id='MiniGrid-FourRooms-adj-v0',
    entry_point='gym_minigrid.envs:FourRoomAdjRoomEnv'
)

register(
    id='MiniGrid-FourRooms-cross-v0',
    entry_point='gym_minigrid.envs:FourRoomCrossRoomEnv'
)

register(
    id='MiniGrid-FourRooms-not_same-v0',
    entry_point='gym_minigrid.envs:FourRoomNotSameRoomEnv'
)
