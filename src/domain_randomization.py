from numpy import random as random
from typing import List
from utils import PType
from pyquaternion import Quaternion


class PrimitiveRandomizer:
    ''' Generate samples from a bounded uniform distribution '''
    REQUIRE_QUAT = {PType.BOX: True, PType.BALL: False, PType.CYLINDER: True,
                    PType.ELLIPSOID: True, PType.FRUSTUM: True}
    SHAPE_LB = {PType.BOX: [0.05, 0.05, 0.05],
                PType.BALL: [0.05],
                PType.CYLINDER: [0.05, 0.05],
                PType.ELLIPSOID: [0.05, 0.05, 0.05],
                PType.FRUSTUM: [0.05, 0.05]}
    SHAPE_UB = {PType.BOX: [0.1, 0.1, 0.1],
                PType.BALL: [0.1],
                PType.CYLINDER: [0.1, 0.1],
                PType.ELLIPSOID: [0.1, 0.1, 0.1],
                PType.FRUSTUM: [0.1, 0.1]}
    POS_LB = [-0.5, 0.3, 0.5]
    POS_UB = [0.5, 1.3, 0.5]  # table height is 0.5

    def __init__(self):
        pass

    def sample(self):
        p_type = self._sample_type()
        shape_args = list(random.uniform(low=self.SHAPE_LB[p_type], high=self.SHAPE_UB[p_type]))
        pos_args = list(random.uniform(low=self.POS_LB, high=self.POS_UB))
        pos_args[2] += max(shape_args)  # shift above the surface of the table
        p_args = shape_args + pos_args
        if self.REQUIRE_QUAT[p_type]:
            p_args = p_args + list(Quaternion.random())
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
        return prm_types, prm_argss

