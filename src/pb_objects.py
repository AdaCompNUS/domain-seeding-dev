from utils import *
import pybullet as p


class Primitive:
    def __init__(self, type: PType, args: List[float]):
        self.id = None
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

        elif type == PType.CYLINDER:
            radius = args[0]
            length = args[1]
            pos_x = args[2]
            pos_y = args[3]
            pos_z = args[4]
            quaternion = args[5:9]
            '''
            spawn a cylinder
            '''
            colBoxId = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId,
                                        basePosition=[pos_x, pos_y, pos_z], baseOrientation=quaternion)

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
            raise Exception('Unsupported primitive type')

        elif type == PType.FRUSTUM:
            pos_x = args[0]
            pos_y = args[1]
            pos_z = args[2]
            radius_top = args[3]
            radius_base = args[4]
            quaternion = args[5:9]
            '''
            spawn a frustum TODO
            '''
            # self.shape = ...
            raise Exception('Unsupported primitive type')

        else:
            raise Exception('Unsupported primitive type')

        self.type = type

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
        for p in self.primitives:
            del p

    def get_states(self):
        p_states = []
        for primitive in self.primitives:
            pos, orientation = p.getBasePositionAndOrientation(primitive.id)
            p_states.append(PrimitiveState(primitive.id, primitive.type, pos, orientation))
            # states[primitive.id] = {
            #     "type": primitive.type,
            #     "pos": pos,
            #     "orientation": orientation
            # }

        o_state = ObjState(p_states)
        return o_state
