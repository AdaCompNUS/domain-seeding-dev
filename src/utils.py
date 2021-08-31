import math
from enum import Enum
from typing import List


class PType(Enum):
    BOX = 1
    BALL = 2
    CYLINDER = 3
    ELLIPSOID = 4
    FRUSTUM = 5  # cup like shape


class GType(Enum):
    BEYOND = 1  # manipulate object to stay beyond a certain line
    BEYOND_WITHIN = 2  # manipulate object to stay beyond a certain line but within a further line
    WITHIN_CIRCLE = 3  # manipulate object to stay within a circle
    WITHIN_BOX = 4  # manipulate object to stay within a box


class ObjState:
    def __init__(self, args: List[float]):
        self.pos = args[0:3]
        self.quaternion = args[3:7]


class TaskGoal:
    def __init__(self, t_type: GType, args: List[float]):
        self.type = t_type
        self.args = args

    def eval(self, state: ObjState):
        if self.type == GType.BEYOND:
            return state.pos[0] * self.args[0] + state.pos[1] * self.args[1] + self.args[2] > 0
        elif self.type == GType.BEYOND_WITHIN:
            return (state.pos[0] * self.args[0] + state.pos[1] * self.args[1] + self.args[2] > 0) + \
                   (state.pos[0] * self.args[3] + state.pos[1] * self.args[4] + self.args[5] < 0)
        elif self.type == GType.WITHIN_CIRCLE:
            dist_x = state.pos[0] - self.args[0]
            dist_y = state.pos[1] - self.args[1]
            return dist_x * dist_x + dist_y * dist_y < self.args[2] * self.args[2]
        elif self.type == GType.WITHIN_BOX:
            dist_x = math.fabs(state.pos[0] - self.args[0])
            dist_y = math.fabs(state.pos[1] - self.args[1])
            return dist_x < self.args[2] and dist_y < self.args[3]
