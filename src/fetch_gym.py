import gym
import numpy as np
from gym import spaces
import cv2
from enum import Enum
import random
import math
import pybullet as p
import pybullet_data
import os
from typing import List

from utils import *
from pb_fetch import FetchRobot
from pb_objects import Object
from domain_randomization import ObjectRandomizer
from exploration import ExplorationPolicy


class FetchPushEnv(gym.Env):
    ACTION_PENALTY_FACTOR = -0.1
    GOAL_REWARD = 1.0

    def __init__(self, gui=False):
        super(FetchPushEnv, self).__init__()

        # connect to pybullet engine
        if gui:
            p.connect(p.GUI) # no GUI
        else:
            p.connect(p.DIRECT) # no GUI
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load floor to prevent objects from falling through floor
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        # internal attributes
        ws_range = 10 # T.B.D
        obs_res = 128 # T.B.D
        max_depth = 1.0 # T.B.D
        self.action_space = spaces.Box(low=-ws_range, high=ws_range, shape=(4,)) # A box in R^4. Each coordinate is bounded.
        self.observation_space = spaces.Box(low=0.0, high=max_depth, shape=(1, obs_res, obs_res), dtype=np.float32) # depth image

        # robot
        self.robot = FetchRobot()

        # table
        self.table = self._add_table()

        # objects
        self.object = None # no object at start
        self.goal = None # no task goal at start

    def reset(self, reset_env=True, reset_robot=True, reset_goal = True, prm_types: List[PType] = None, prm_argss: List[List[float]] = None, goal: TaskGoal = None):
        if reset_env: # re-generate the scene
            if self.object:
                del self.object
            '''
            re-locate the table, allow for slight noise in position and orientation
            '''
            self.table = self._add_table()

            # re-spawn object on the table, with some user-given types and parameters
            self.object = Object(prm_types, prm_argss)

        if reset_goal: # reset manipulation goal
            self.goal = goal

        if reset_robot:
            '''
            reset robot to default pose
            '''
            self.robot.reset()

        info = {}
        info['obj_state'] = self.object.get_states() # get true object state from simulator

        '''
        get the depth image as observation
        '''
        obs = self._get_depth()
        return obs, info # I need these two types of infos for different purposes: the former for learning, the latter for planning

    def step(self, action, mode='normal'):
        '''
        Execute the action using IK control and simulate object movement
        '''

        start_ee_pos, start_ee_ori, end_ee_pos, end_ee_ori = action
        if mode == 'normal':
            '''
            Run simulation with real-time speed, e.g., for visualizing an episode
            '''

            self.robot.plan_to_ee_pose(start_ee_pos, start_ee_ori)
            self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=5) #
            print(self.robot.get_ee_pose())

        elif mode == 'super':
            '''
            Both planning and RL require to run simulation FASTER than real time.
            '''

            self.robot.set_ee_pose(start_ee_pos, start_ee_ori) # this is faster
            self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=1)
        else:
            raise Exception('Unsupported simulation mode {}'.format(mode))

        info = {}
        info['obj_state'] = self.object.get_states() # get true object state from simulator
        obs = self._get_depth() # get depth image as obs
        # reward, done = self.reward(info['obj_state'], action)
        reward = 0
        done = True

        return obs, reward, done, info # include true state of the manipulated object (x,y,z, quat) in info.

    def reward(self, obj_state, action):
        done = self.goal.eval(obj_state)
        reward = self.ACTION_PENALTY_FACTOR * math.sqrt((action[0]-action[2])**2 + (action[1]-action[3])**2)
        if done:
            reward += self.GOAL_REWARD
        return reward, done

    def render(self):
        '''
        render the camera image to the window using OpenCV ONLY when this function is called.
        other than this function, the system should run in headless mode.
        '''


    def _add_table(self, pos=[0, 0.8, 0], orientation=[0,0,0,1]):
        '''
        Add table to scene
        pos, the center position of table. Default is 0.8m in front of robot
        '''
        table = p.createCollisionShape(p.GEOM_MESH, fileName="table/table.obj")
        tableId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table, basePosition=pos, baseOrientation=orientation)

        self.robot.fetch_mp.add_obstacle_body(tableId)

        return tableId

    def _get_depth(self):
        # TODO
        print("Dummy get depth, to be implemented!!!")
        return None

if __name__ == '__main__':
    env = FetchPushEnv(gui=True)

    randomizer = ObjectRandomizer()
    exp_policy = ExplorationPolicy()
    prm_types, prm_argss = randomizer.sample(num_objects=1)
    # prm_types = [PType.BOX]
    # prm_argss = [[0, 0.75, 0.55, 0.1, 0.1, 0.1, 0, 0, 0, 1]]
    obs, info = env.reset(prm_types=prm_types, prm_argss=prm_argss)
    obs, reward, done, info = env.step(exp_policy.next_action(info['obj_state']))
    input()

