import math
from enum import Enum
from typing import List
import pybullet as p
from pyquaternion import Quaternion

''' Util classes '''


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


class PrimitiveState:
    def __init__(self, id: int, type: PType, scale: List[float], pos: List[float], ori: List[float],
                 com: List[float], mass: float, friction: float):
        self.id = id
        self.type = type
        self.scale = scale
        self.pos = pos
        self.quaternion = ori
        self.com = com
        self.mass = mass
        self.friction = friction


class ObjState:
    DIM_PREDICTIONS = 7

    def __init__(self, p_states: List[PrimitiveState]):
        self.primitives = p_states
        self.scale = p_states[0].scale
        self.pos = p_states[0].pos
        self.quaternion = p_states[0].quaternion
        self.com = p_states[0].com
        self.mass = sum([prm.mass for prm in p_states])
        self.friction = sum([prm.friction for prm in p_states]) / len(p_states)

    def serialize(self):
        res = self.scale + self.com + [self.mass, self.friction]
        assert(len(res) == self.DIM_PREDICTIONS)
        return res


class TaskGoal:
    def __init__(self, t_type: GType, args: List[float]):
        self.type = t_type
        self.args = args

    def eval(self, state: ObjState):
        if self.type == GType.BEYOND:
            return state.pos[0] * self.args[0] + state.pos[1] * self.args[1] - self.args[2]  # > 0
        elif self.type == GType.BEYOND_WITHIN:
            reward = 0.0
            # if (state.pos[0] * self.args[0] + state.pos[1] * self.args[1] - self.args[2] > 0):  # beyond
            reward += min(0.0, state.pos[0] * self.args[0] + state.pos[1] * self.args[1] - self.args[2])
            reward += min(0.0, -(state.pos[0] * self.args[3] + state.pos[1] * self.args[4] - self.args[5]))
            # if (state.pos[0] * self.args[3] + state.pos[1] * self.args[4] - self.args[5] < 0):
            return reward
        elif self.type == GType.WITHIN_CIRCLE:
            dist_x = state.pos[0] - self.args[0]
            dist_y = state.pos[1] - self.args[1]
            return self.args[2] * self.args[2] - (dist_x * dist_x + dist_y * dist_y)
        elif self.type == GType.WITHIN_BOX:
            dist_x = math.fabs(state.pos[0] - self.args[0])
            dist_y = math.fabs(state.pos[1] - self.args[1])
            return dist_x < self.args[2] and dist_y < self.args[3]


class ParameterizedPolicy():
    dim = 8
    rad_scale = 1.0
    time_scale = 1.0

    def __init__(self, x):
        # Fix some components
        assert(len(x) == self.dim)
        start_euler = self.rad_scale * (x[3] - math.radians(90)) + math.radians(90)
        end_eular = self.rad_scale * (x[7] - math.radians(90)) + math.radians(90)
        # duration = self.time_scale * (x[8] - 1.0) + 1.0
        duration = 1.0

        self.start_pos = x[0:3]
        self.start_pos[2] = max(self.start_pos[2], 0.6)
        self.start_euler = [0.0, 0.0, start_euler]  # x[3:6]
        self.start_ori = p.getQuaternionFromEuler(self.start_euler)
        self.end_pos = x[4:7]  # x[6:9]
        self.end_pos[2] = max(self.end_pos[2], 0.6)
        self.end_euler = [0.0, 0.0, end_eular]  # x[9:12]
        self.end_ori = p.getQuaternionFromEuler(self.end_euler)
        self.duration = duration  # x[12]

    def text(self):
        return 'parameterized policy:\n  start {}\n  ori {}\n  end {}\n  ori {}\n  duration {}'.format(
            self.start_pos, self.start_euler, self.end_pos, self.end_euler, self.duration)

    def refract(self):
        return self.start_pos, self.start_ori, self.end_pos, self.end_ori, self.duration

    def serialize(self):
        start_euler = 1.0 / self.rad_scale * (self.start_euler[2] - math.radians(90)) + math.radians(90)
        end_euler = 1.0 / self.rad_scale * (self.end_euler[2] - math.radians(90)) + math.radians(90)
        duration = 1.0 / self.time_scale * (self.duration - 1.0) + 1.0

        return self.start_pos + [start_euler] + self.end_pos + [end_euler]  # + [duration]

    def trans_dist(self):
        return math.sqrt((self.start_pos[0] - self.end_pos[0]) ** 2 +
                         (self.start_pos[1] - self.end_pos[1]) ** 2 +
                         (self.start_pos[2] - self.end_pos[2]) ** 2)

    def rot_dist(self):
        return Quaternion.absolute_distance(
            Quaternion(self.start_ori), Quaternion(self.end_ori))


class AverageMeter(object):
    """
    A utility class to compute statisitcs of losses and accuracies
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
