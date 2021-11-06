from numpy import random as random
from typing import List
import math
from utils.classes import PType, ObjState, PCom, PScale, POrientation, PPos, PPhysics
from pyquaternion import Quaternion
import pybullet as p


class PrimitiveRandomizer:
    ''' Generate samples from a bounded uniform distribution '''
    REQUIRE_QUAT = {PType.BOX: True, PType.BALL: False, PType.CYLINDER: True,
                    PType.ELLIPSOID: True, PType.FRUSTUM: True}

    SEARCH_LB = PScale.LB[PType.CYLINDER] + PCom.LB + PPhysics.LB
    SEARCH_UB = PScale.UB[PType.CYLINDER] + PCom.UB + PPhysics.UB

    def __init__(self):
        assert (len(self.SEARCH_LB) == ObjState.DIM_PREDICTIONS)
        assert (len(self.SEARCH_UB) == ObjState.DIM_PREDICTIONS)

    @staticmethod
    def mean_cemas_params(p_type: PType):
        """Only considering scale (2), center of mass (3), mass (1), friction (1)"""
        scale = PScale.default(p_type)
        com = PCom.default(p_type)
        mass, friction = PPhysics.default(p_type)

        return ObjState.to_cmaes_external(scale, com, mass, friction)

    @staticmethod
    def refract_cemas_params(x: List[float], p_type: PType):
        """Only considering scale (2), center of mass (3), mass (1), friction (1)"""
        scale = PScale(x[0:2], p_type)
        com = PCom(x[2:4], p_type)
        physics = PPhysics(x[4:6], p_type)

        return scale, com, physics

    def sample(self):
        # random.seed(30)
        p_args = {
            'shape': None,
            'pos': None,
            'rot': None,
            'com': None,
            'phy': None
        }
        p_type = self._sample_type()
        shape_obj = PScale.uniform(p_type)
        radius, length = shape_obj.refract()
        p_args['shape'] = shape_obj
        p_args['pos'] = PPos.uniform(p_type, radius)
        p_args['rot'] = POrientation.uniform(p_type)
        p_args['com'] = PCom.uniform(shape_obj, p_type)
        p_args['phy'] = PPhysics.uniform(p_type)

        return p_type, p_args

    def _sample_type(self):
        return PType.CYLINDER


class ObjectRandomizer:
    def __init__(self):
        self.randomizer = PrimitiveRandomizer()

    def sample(self, num_objects):
        prm_argss = []
        prm_types = []
        for i in range(num_objects):
            p_type, p_args = self.randomizer.sample()
            prm_types.append(p_type)
            prm_argss.append(p_args)
        # print('[dr] randomize object configuration:\n  types {}'.format(prm_types))
        # print('[dr] args\n  radius {}\n  length {}\n  pos {}\n  quat {}'.format(
        #     prm_argss[0][0], prm_argss[0][1],
        #     prm_argss[0][2:5], prm_argss[0][5:9]
        # ))
        return prm_types, prm_argss
