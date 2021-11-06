from utils.classes import ObjState, ParameterizedPolicy
import math
import pybullet as p


class ExplorationPolicy:
    DURATION = 1.0

    def __init__(self):
        pass

    def next_action(self, state: ObjState):
        if state:
            obj_scale = max(state.scale.values) + 0.05
            posx = state.pos[0]
            posy = state.pos[1]
        else:
            obj_scale = 0.5
            posx = 0.0
            posy = 0.7
        # start_pos = [posx, posy - obj_scale, 0.62]
        start_pos = [posx, posy - obj_scale, 0.65]
        # start_euler = [0, 0, math.radians(90)]
        start_euler = [math.radians(90)]
        end_pos = [posx, min(posy + 0.3, 1.3), 0.65]
        # end_pos = [posx, 0.8, 0.72]
        # end_euler = [0, 0, math.radians(90)]
        end_euler = [math.radians(90)]

        ret = ParameterizedPolicy(x=start_pos + start_euler + end_pos + end_euler)  #  + [self.DURATION])
        print('[pi_e] Chosen {}'.format(ret.text()))
        return ret
