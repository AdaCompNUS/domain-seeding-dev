import math
from enum import Enum
from typing import List
import pybullet as p
from pyquaternion import Quaternion
from numpy import random as random

''' Util classes '''
from utils.functions import error_handler


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


class PScale:
    """A class for managing different representations of shape scale
    h_shift
    """
    LB = {PType.BOX: [0.05, 0.05, 0.05],
          PType.BALL: [0.05],
          PType.CYLINDER: [0.05, 0.1],  # [radius, length]
          PType.ELLIPSOID: [0.05, 0.05, 0.05],
          PType.FRUSTUM: [0.05, 0.05]}
    UB = {PType.BOX: [0.1, 0.1, 0.1],
          PType.BALL: [0.1],
          PType.CYLINDER: [0.1, 0.2],  # [radius, length]
          PType.ELLIPSOID: [0.1, 0.1, 0.1],
          PType.FRUSTUM: [0.1, 0.1]}

    def __init__(self, values, p_type: PType):
        """x have to be obstained from another PScale"""
        self.p_type = p_type
        """Input from a simulator randomizer or a planning sim constructor"""
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for PScale at [classes.py]")

        self.values = values

    @staticmethod
    def default(p_type: PType):
        """mean of the bounds"""
        return [sum(x) / 2.0 for x in zip(PScale.LB[p_type], PScale.UB[p_type])]

    @staticmethod
    def uniform(p_type: PType):
        return PScale(random.uniform(low=PScale.LB[p_type], high=PScale.UB[p_type]), p_type)

    def to_prm_params(self):
        """To construct the primitive"""
        return self.values

    def to_cemas(self):
        """To be used as part of init_x in CMA-ES"""
        return self.values

    def refract(self):
        if self.p_type == PType.CYLINDER:
            radius = self.values[0]
            length = self.values[1]
            return radius, length
        else:
            raise Exception("Unsupported primitive type for PScale.serialize at [classes.py]")


class PPos:
    """A class for managing different representations of shape scale
    """
    LB = [-0.2, 0.5, 0.5]
    UB = [0.2, 0.8, 0.5]

    def __init__(self, values: List[float], p_type: PType):
        """x have to be obstained from another PScale"""
        self.p_type = p_type
        """Input from a simulator randomizer or a planning sim constructor"""
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for PPos at [classes.py]")

        self.values = values

    @staticmethod
    def uniform(p_type: PType, radius: float, table_height=0.5):
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for PPos at [classes.py]")
        pos_args = list(random.uniform(low=PPos.LB, high=PPos.UB))
        height = pos_args[2]
        pos_args[2] = max(height, table_height + radius + 0.001)  # shift above the surface of the table
        return PPos(pos_args, p_type)

    def to_prm_params(self):
        """To construct the primitive"""
        return self.values

    def to_cemas(self):
        """To be used as part of init_x in CMA-ES"""
        return self.values

    def refract(self):
        if self.p_type == PType.CYLINDER:
            x = self.values[0]
            y = self.values[1]
            z = self.values[2]
            return z, y, z
        else:
            raise Exception("Unsupported primitive type for PPos.serialize at [classes.py]")


class POrientation:
    """A class for managing different representations of orientations
    """

    def __init__(self, values: List[float], p_type: PType):
        """x have to be obstained from another PScale"""
        self.p_type = p_type
        """Input from a simulator randomizer or a planning sim constructor"""
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for POrientation at [classes.py]")

        self.values = values

    @staticmethod
    def uniform(p_type: PType):
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for POrientation at [classes.py]")

        rad_z = math.radians(random.uniform(-180.0, 180.0))
        return POrientation([rad_z], p_type)

    @staticmethod
    def from_quaternion(quaternion: List[float], p_type: PType):
        rot_x, rot_y, rot_z = p.getEulerFromQuaternion(quaternion)
        return POrientation([rot_z], p_type)

    def to_prm_params(self):
        """To construct the primitive"""
        rad_z = self.values[0]
        return list(p.getQuaternionFromEuler([0, math.radians(90), rad_z]))

    def to_cemas(self):
        """To be used as part of init_x in CMA-ES"""
        return self.values

    def refract(self):
        if self.p_type == PType.CYLINDER:
            rad_z = self.values[0]
            return rad_z
        else:
            raise Exception("Unsupported primitive type for POrientation.serialize at [classes.py]")


class PCom:
    """A class for managing different representations of center of mass
    h_shift
    """
    LB = [-PScale.UB[PType.CYLINDER][1] / 2.0, 0.0]
    UB = [PScale.UB[PType.CYLINDER][1] / 2.0, PScale.UB[PType.CYLINDER][0]]

    def __init__(self, values: List[float], p_type: PType):
        self.p_type = p_type
        """Input from a simulator randomizer or a planning sim constructor"""
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for COM at [classes.py]")

        self.values = values

    @staticmethod
    def default(p_type: PType):
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for COM at [classes.py]")

        return [0.0, 0.0]

    @staticmethod
    def uniform(scale: PScale, p_type: PType):
        """scale = [radius, length]"""
        radius = scale.values[0]
        length = scale.values[1]
        if p_type == PType.CYLINDER:
            v_shift = random.uniform(-length / 2.0, length / 2.0)
            h_shift = random.uniform(0, radius)
            return PCom([h_shift, v_shift], p_type)
        else:
            raise Exception("Unsupported primitive type for COM at [classes.py]")

    @staticmethod
    def shift_to_prm_params(h_shift: float, v_shift: float):
        """To construct the primitive"""
        return [v_shift, 0.0, h_shift]

    def to_prm_params(self):
        """To construct the primitive"""
        h_shift = self.values[0]
        v_shift = self.values[1]
        return [v_shift, 0.0, h_shift]

    def to_cemas(self):
        h_shift = self.values[0]
        v_shift = self.values[1]
        """To be used as part of init_x in CMA-ES"""
        return [h_shift, v_shift]

    def refract(self):
        h_shift = self.values[0]
        v_shift = self.values[1]
        return h_shift, v_shift


class PPhysics:
    """A class for managing different representations of shape scale
    """
    LB = [0.0, 0.0]
    UB = [10.0, 1.0]

    def __init__(self, values: List[float], p_type: PType):
        """x have to be obstained from another PScale"""
        self.p_type = p_type
        """Input from a simulator randomizer or a planning sim constructor"""
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for PPos at [classes.py]")

        self.values = values

    @staticmethod
    def default(p_type: PType):
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for COM at [classes.py]")
        mass = (PPhysics.LB[0] + PPhysics.UB[0]) / 2.0
        friction = (PPhysics.LB[1] + PPhysics.UB[1]) / 2.0
        return [mass, friction]

    @staticmethod
    def uniform(p_type: PType):
        if p_type != PType.CYLINDER:
            raise Exception("Unsupported primitive type for PPos at [classes.py]")
        args = [random.uniform(low=PPhysics.LB[0], high=PPhysics.UB[0]),  # mass
                random.uniform(low=PPhysics.LB[1], high=PPhysics.UB[1])  # friction
                ]
        return PPhysics(args, p_type)

    def to_prm_params(self):
        """To construct the primitive"""
        return self.values

    def to_cemas(self):
        """To be used as part of init_x in CMA-ES"""
        return self.values

    def refract(self):
        if self.p_type == PType.CYLINDER:
            mass = self.values[0]
            friction = self.values[1]
            return mass, friction
        else:
            raise Exception("Unsupported primitive type for PPos.serialize at [classes.py]")


class PrimitiveState:
    def __init__(self, id: int, type: PType, scale: PScale, pos: List[float], ori: List[float],
                 com: PCom, physics: PPhysics):
        self.id = id
        self.type = type
        self.scale = scale
        self.pos = pos
        self.quaternion = ori
        self.com = com
        self.mass, self.friction = physics.refract()


class ObjState:
    """scale (2), center of mass (2), mass (1), friction (1)"""
    DIM_PREDICTIONS = 6

    def __init__(self, p_states: List[PrimitiveState]):
        self.primitives = p_states
        self.scale = p_states[0].scale
        self.pos = p_states[0].pos
        self.quaternion = p_states[0].quaternion
        self.com = p_states[0].com
        self.mass = sum([prm.mass for prm in p_states])
        self.friction = sum([prm.friction for prm in p_states]) / len(p_states)
        # This assumes the object only has 1 primitive
        self.type = p_states[0].type

    def to_cmaes(self):
        try:
            res = self.scale.to_cemas() + self.com.to_cemas() + [self.mass, self.friction]
            print(f'[class.py] self.scale.to_cemas():{self.scale.to_cemas()}, '
                  f'self.com.to_cemas():{self.com.to_cemas()}, res:{res}')
            assert (len(res) == self.DIM_PREDICTIONS)
            return res
        except Exception as e:
            error_handler(e)

    @staticmethod
    def to_cmaes_external(scale: List[float], com: List[float], mass: float, friction: float):
        return scale + com + [mass, friction]

    @staticmethod
    def distance(ref_param: List[float], param: List[float]):
        diff = [x0 - x1 for (x0, x1) in zip(ref_param, param)]
        return ObjState.norm(diff)

    @staticmethod
    def norm(x):
        return sum([abs(e) for e in x])


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

    def __init__(self, x: List[float]):
        # Fix some components
        assert (len(x) == self.dim)
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
