from utils import ObjState
import math
import pybullet as p


class ExplorationPolicy:
    def __init__(self):
        pass

    def next_action(self, state: ObjState):
        obj_scale = max(state.scale) + 0.05
        start_pos = [state.pos[0]-0.1, state.pos[1] - obj_scale, 0.6]
        # start_ori = p.getQuaternionFromEuler([0, 0, math.radians(90)])
        start_ori = p.getQuaternionFromEuler([0, math.radians(180), 0])
        end_pos = [state.pos[0]-0.1, min(state.pos[1] + 0.2, 1.3), 0.6]
        # end_ori = p.getQuaternionFromEuler([0, 0, math.radians(90)])
        end_ori = p.getQuaternionFromEuler([0, math.radians(180), 0])
        return (start_pos, start_ori, end_pos, end_ori)