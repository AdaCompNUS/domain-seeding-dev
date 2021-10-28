from numpy import random as random
from typing import List
import math
from utils.classes import PType, ObjState
from pyquaternion import Quaternion
import pybullet as p


class PrimitiveRandomizer:
    ''' Generate samples from a bounded uniform distribution '''
    REQUIRE_QUAT = {PType.BOX: True, PType.BALL: False, PType.CYLINDER: True,
                    PType.ELLIPSOID: True, PType.FRUSTUM: True}
    SHAPE_LB = {PType.BOX: [0.05, 0.05, 0.05],
                PType.BALL: [0.05],
                PType.CYLINDER: [0.05, 0.1],
                PType.ELLIPSOID: [0.05, 0.05, 0.05],
                PType.FRUSTUM: [0.05, 0.05]}
    SHAPE_UB = {PType.BOX: [0.1, 0.1, 0.1],
                PType.BALL: [0.1],
                PType.CYLINDER: [0.1, 0.2],
                PType.ELLIPSOID: [0.1, 0.1, 0.1],
                PType.FRUSTUM: [0.1, 0.1]}
    # TODO to finetune these values
    # valid x: -0.1~0.1, valid y :0.5~0.8, valid z is 0.5
    POS_LB = [-0.2, 0.5, 0.5]
    POS_UB = [0.2, 0.8, 0.5]
    COM_LB = [-SHAPE_UB[PType.CYLINDER][0], -SHAPE_UB[PType.CYLINDER][0], -SHAPE_UB[PType.CYLINDER][1]/2.0]
    COM_UB = [SHAPE_UB[PType.CYLINDER][0], SHAPE_UB[PType.CYLINDER][0], SHAPE_UB[PType.CYLINDER][1]/2.0]
    PHY_LB = [0.0, 0.0]
    PHY_UB = [10.0, 1.0]

    SEARCH_LB = SHAPE_LB[PType.CYLINDER] + COM_LB + PHY_LB
    SEARCH_UB = SHAPE_UB[PType.CYLINDER] + COM_UB + PHY_UB

    def __init__(self):
        assert (len(self.SEARCH_LB) == ObjState.DIM_PREDICTIONS)
        assert (len(self.SEARCH_UB) == ObjState.DIM_PREDICTIONS)

    @staticmethod
    def mean_params(p_type):
        """Only considering scale (2), center of mass (3), mass (1), friction (1)"""
        scale = [sum(x)/2.0 for x in zip(PrimitiveRandomizer.SHAPE_LB[p_type], PrimitiveRandomizer.SHAPE_UB[p_type])]
        com = [0.0, 0.0, 0.0]
        mass = (PrimitiveRandomizer.PHY_LB[0] + PrimitiveRandomizer.PHY_UB[0]) / 2.0
        friction = (PrimitiveRandomizer.PHY_LB[1] + PrimitiveRandomizer.PHY_UB[1]) / 2.0

        return ObjState.serialize_external(scale, com, mass, friction)

    def sample(self):
        # random.seed(30)
        p_type = self._sample_type()
        shape_args = list(random.uniform(low=self.SHAPE_LB[p_type], high=self.SHAPE_UB[p_type]))
        pos_args = list(random.uniform(low=self.POS_LB, high=self.POS_UB))
        # pos_args[2] += max(shape_args)  # shift above the surface of the table
        pos_args[2] += shape_args[0] + 0.001  # shift above the surface of the table
        physics_args = [random.uniform(low=-math.pi, high=math.pi),  # com angle
                        random.uniform(low=0, high=shape_args[0]/2.0),  # com_dist
                        random.uniform(low=-shape_args[1]/4.0, high=shape_args[1]/4.0),  # com_height
                        random.uniform(low=self.PHY_LB[0], high=self.PHY_UB[0]),  # mass
                        random.uniform(low=self.PHY_LB[1], high=self.PHY_UB[1])  # friction
                        ]

        p_args = shape_args + pos_args
        if self.REQUIRE_QUAT[p_type]:
            # p_args = p_args + list(Quaternion.random())
            p_args = p_args + list(p.getQuaternionFromEuler([0, math.radians(90), math.radians(random.uniform(-180.0, 180.0))]))
        p_args = p_args + physics_args

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


