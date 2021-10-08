from utils.classes import ObjState, ParameterizedPolicy
import math
import pybullet as p


class ExplorationPolicy:
    def __init__(self):
        pass

    def next_action(self, state: ObjState):
        obj_scale = max(state.scale) + 0.05
        start_pos = [state.pos[0], state.pos[1] - obj_scale, 0.62]
        start_euler = [0, 0, math.radians(90)]
        end_pos = [state.pos[0], min(state.pos[1] + 0.2, 1.3), 0.62]
        end_euler = [0, 0, math.radians(90)]
        duration = 2.0

        ret = ParameterizedPolicy(x=start_pos + start_euler + end_pos + end_euler + [duration])
        print('[pi_e] Chosen {}'.format(ret.text()))
        return ret
