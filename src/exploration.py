from utils import ObjState
import math
import pybullet as p


class ExplorationPolicy:
    def __init__(self):
        pass

    def next_action(self, state: ObjState):
        start_pos = [0, 0.6, 0.6]
        # start_ori = p.getQuaternionFromEuler([0, 0, math.radians(90)])
        start_ori = p.getQuaternionFromEuler([math.radians(90), 0, 0])
        end_pos = [0, 0.8, 0.6]
        # end_ori = p.getQuaternionFromEuler([0, 0, math.radians(90)])
        end_ori = p.getQuaternionFromEuler([math.radians(90), 0, 0])
        return (start_pos, start_ori, end_pos, end_ori)