import math

from utils.classes import *
from utils.functions import error_handler
import pybullet as p


class Primitive:
    def __init__(self, type: PType, args: List[float]):
        self.id = None
        self.scale = None
        self.ini_ori, self.ini_pos = None, None
        self.com, self.mass, self.friction = None, None, None
        if type == PType.BOX:
            size_x = args[0]
            size_y = args[1]
            size_z = args[2]
            pos_x = args[3]
            pos_y = args[4]
            pos_z = args[5]
            quaternion = args[6:10]
            '''
            spawn a box
            '''
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size_x / 2.0, size_y / 2.0, size_z / 2.0])
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId,
                                        basePosition=[pos_x, pos_y, pos_z], baseOrientation=quaternion)
            self.ini_pos = [pos_x, pos_y, pos_z]
            self.ini_ori = quaternion
            self.scale = args[0:3]

        elif type == PType.BALL:
            radius = args[0]
            pos_x = args[1]
            pos_y = args[2]
            pos_z = args[3]
            '''
            spawn a ball
            '''
            colBoxId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId,
                                        basePosition=[pos_x, pos_y, pos_z])
            self.ini_pos = [pos_x, pos_y, pos_z]
            self.scale = args[0:1]

        elif type == PType.CYLINDER:
            radius = args[0]
            length = args[1]
            pos_x = args[2]
            pos_y = args[3]
            pos_z = args[4]
            quaternion = args[5:9]
            com_angle = args[9]
            com_dist = args[10]
            com_height = args[11]
            self.com = [com_dist * math.cos(com_angle), com_dist * math.sin(com_angle), com_height]
            self.mass = args[12]
            self.friction = args[13]
            self.ini_pos = [pos_x, pos_y, pos_z]
            self.ini_ori = quaternion
            self.scale = [radius, length]
            # linkMasses = [1.0]
            # linkCollisionShapeIndices = [-1]
            # linkVisualShapeIndices = [-1]
            # linkPositions = [[0, 0, 0]]
            # linkOrientations = [[0, 0, 0]]
            # linkInertialFramePositions = [[0, 0, 0]]
            # linkInertialFrameOrientations = [[0, 0, 0]]
            # linkParentIndices = [-1]
            # linkJointTypes = [p.JOINT_FIXED]
            # linkJointAxis = [[0, 0, 0]]
            '''
            spawn a cylinder
            '''
            print(f'[pb_objects.py] Object info: \n'
                  f'  Type: cylinder \n'
                  f'  Shape: [{radius}, {length}]\n'
                  f'  Pose: {self.ini_pos}, {self.ini_ori}\n'
                  f'  Physics: {self.com}, {self.mass}, {self.friction}')
            colBoxId = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length,
                                              collisionFramePosition=[0, 0, 0])
            self.id = p.createMultiBody(baseMass=self.mass, baseCollisionShapeIndex=colBoxId,
                                        basePosition=self.ini_pos, baseOrientation=self.ini_ori,
                                        baseInertialFramePosition=self.com)
            p.changeDynamics(self.id, -1, lateralFriction=self.friction)

        elif type == PType.ELLIPSOID:
            radius_x = args[0]
            radius_y = args[1]
            radius_z = args[2]
            pos_x = args[3]
            pos_y = args[4]
            pos_z = args[5]
            quaternion = args[6:10]
            '''
            spawn a ellipsoid TODO
            '''
            # self.shape = ...
            self.scale = args[0:3]
            raise Exception('Unsupported primitive type')

        elif type == PType.FRUSTUM:
            height = args[0]
            radius_top = args[1]
            radius_base = args[2]
            pos_x = args[3]
            pos_y = args[4]
            pos_z = args[5]
            quaternion = args[6:10]
            '''
            spawn a frustum TODO
            '''
            # self.shape = ...
            self.scale = args[0:3]
            raise Exception('Unsupported primitive type')

        else:
            raise Exception('Unsupported primitive type')

        self.type = type

    def reset(self):
        try:
            # print(f'[pb_object.py] Resetting primitive {self.id} to {self.ini_pos} {self.ini_ori}')
            p.resetBasePositionAndOrientation(self.id, self.ini_pos, self.ini_ori)
            p.resetBaseVelocity(self.id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        except Exception as e:
            error_handler(e)

    def __del__(self):
        '''
        clean up the primitive
        '''


class Object:
    def __init__(self, types: List[PType], argss: List[List[float]]):
        self.primitives = []
        if types is not None:
            for i in range(len(types)):
                type = types[i]
                args = argss[i]
                self.primitives.append(Primitive(type, args))
        '''
        Do something to collate the primitives together as one object in PyBullet.
        '''
        pass

    def __del__(self):
        for prim in self.primitives:
            p.removeBody(prim.id)
            del prim

    def reset(self):
        try:
            for primitive in self.primitives:
                primitive.reset()
        except Exception as e:
            error_handler(e)

    def get_states(self):
        p_states = []
        for primitive in self.primitives:
            pos, orientation = p.getBasePositionAndOrientation(primitive.id)
            p_states.append(PrimitiveState(primitive.id, primitive.type, primitive.scale, pos, orientation,
                                           primitive.com, primitive.mass, primitive.friction))
            # states[primitive.id] = {
            #     "type": primitive.type,
            #     "pos": pos,
            #     "orientation": orientation
            # }

        o_state = ObjState(p_states)
        return o_state
