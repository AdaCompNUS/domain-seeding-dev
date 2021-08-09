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

import gibson2
from gibson2.robots.fetch_robot import Fetch
from gibson2.utils.utils import parse_config
import gibson2.external.pybullet_tools.utils as pb_utils

from fetch_pb_motion_planning import FetchMotionPlanningPyBullet
from iGibson.gibson2.simulator import Simulator


# -------- Constants ---------
SIMULATION_FREQ = 240


class PType(Enum):
    BOX = 1
    BALL = 2
    CYLINDER = 3
    ELLIPSOID = 4
    FRUSTUM = 5 # cup like shape


class GType(Enum):
    BEYOND = 1 # manipulate object to stay beyond a certain line
    BEYOND_WITHIN = 2 # manipulate object to stay beyond a certain line but within a further line
    WITHIN_CIRCLE = 3 # manipulate object to stay within a circle
    WITHIN_BOX = 4 # manipulate object to stay within a box


class ObjState():
    def __init__(self, args: List[float]):
        self.pos = args[0:3]
        self.quaternion = args[3:7]


class TaskGoal():
    def __init__(self, type:GType, args:List[float]):
        self.type = type
        self.args = args

    def eval(self, state:ObjState):
        if self.type == GType.BEYOND:
            return (state.pos[0] * self.args[0] + state.pos[1] * self.args[1] + self.args[2] > 0)
        elif self.type == GType.BEYOND_WITHIN:
            return (state.pos[0] * self.args[0] + state.pos[1] * self.args[1] + self.args[2] > 0) + \
                    (state.pos[0] * self.args[3] + state.pos[1] * self.args[4] + self.args[5] < 0)
        elif self.type == GType.WITHIN_CIRCLE:
            dist_x = state.pos[0] - self.args[0]
            dist_y = state.pos[1] - self.args[1]
            return dist_x * dist_x + dist_y * dist_y < self.args[2] * self.args[2]
        elif self.type == GType.WITHIN_BOX:
            dist_x = math.fabs(state.pos[0] - self.args[0])
            dist_y = math.fabs(state.pos[1] - self.args[1])
            return dist_x < self.args[2] and dist_y < self.args[3]


class Primitive():
    def __init__(self, type:PType, args:List[float]):
        self.id = None
        if type == PType.BOX:
            pos_x = args[0]
            pos_y = args[1]
            pos_z = args[2]
            size_x = args[3]
            size_y = args[4]
            size_z = args[5]
            quaternion = args[6:10]
            '''
            spawn a box
            '''
            colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size_x / 2.0, size_y / 2.0, size_z / 2.0])
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, basePosition=[pos_x, pos_y, pos_z], baseOrientation=quaternion)

        elif type == PType.BALL:
            radius = args[0]
            pos_x = args[1]
            pos_y = args[2]
            pos_z = args[3]
            '''
            spawn a ball
            '''
            colBoxId = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, basePosition=[pos_x, pos_y, pos_z])

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
            colBoxId = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height = length)
            self.id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, basePosition=[pos_x, pos_y, pos_z], baseOrientation=quaternion)

        elif type == PType.ELLIPSOID:
            pos_x = args[0]
            pos_y = args[1]
            pos_z = args[2]
            radius_x = args[3]
            radius_y = args[4]
            radius_z = args[5]
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


class Object():
    def __init__(self, types:List[PType], argss:List[List[float]]):
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
        states = {}
        for primitive in self.primitives:
            pos, orientation = p.getBasePositionAndOrientation(primitive.id)
            states[primitive.id] = {
                "type" : primitive.type,
                "pos" : pos,
                "orientation" : orientation
            }

        return states


class FetchRobot():
    '''
    Fetch Robot, internally use iGibson's Fetch robot class
    '''
    def __init__(self) -> None:
        config = parse_config(os.path.join(gibson2.example_config_path, 'fetch_reaching.yaml'))
        self.fetch = Fetch(config)
        self.fetch.load()
        self.fetchId = self.fetch.robot_ids[0]
        print(self.fetchId)

        # get arm joints ids
        self.fetch_non_fixed_joints = []
        self.fetch_non_fixed_joint_names = []
        fetch_num_joints = p.getNumJoints(self.fetchId)
        ee_idx = pb_utils.link_from_name(self.fetchId, "gripper_link")
        print(ee_idx)
        for i in range(fetch_num_joints):
            joint_info = pb_utils.get_joint_info(self.fetchId, i)
            if joint_info.jointType!= p.JOINT_FIXED:
                self.fetch_non_fixed_joints.append(i)
                self.fetch_non_fixed_joint_names.append(joint_info.jointName)

        # end effector
        self.fetch_ee_idx = 19

        # motion planning
        self.fetch_mp = FetchMotionPlanningPyBullet(robot=self.fetch)

    def reset(self):
        self.fetch.robot_specific_reset() # reset arm joint positions
        self.set_base_pose([0,0,0], p.getQuaternionFromEuler([0, 0, math.radians(90)])) # reset base positions

    def set_base_pose(self, pos, orientation):
        self.fetch.set_position(pos)
        self.fetch.set_orientation(orientation)

    def set_arm_joint_positions(self, joint_positions):
        '''
        set arm to joint positions. This ignores simulation.
        '''
        self.fetch.set_joint_positions(joint_positions)

    def set_ee_pose(self, tgt_pos, tgt_ori):
        '''
        set ee to target pose. This ignores simulation.
        '''
        joint_positions = self.fetch_mp.get_arm_joint_positions(tgt_pos, tgt_ori) # IK
        if joint_positions is not None:
            self.set_arm_joint_positions(joint_positions)
        else:
            print("[FetchRobot] ERROR: IK failed")

    def get_ee_pose(self):
        pos = self.fetch.parts['gripper_link'].get_position()
        ori = self.fetch.parts['gripper_link'].get_orientation()
        return pos, ori

    def plan_to_joint_positions(self, joint_positions):
        '''
        Plan arm to joint positions. Invokes motion planning
        '''
        print("[FetchRobot]: plan_to_joint_positions")

        arm_path = self.fetch_mp.plan_to_joint_goal(joint_positions)
        if arm_path is not None:
            self.fetch_mp.execute(arm_path)
        else:
            print("[FetchRobot] ERROR: planning failed")

    def plan_to_ee_pose(self, tgt_pos, tgt_ori):
        '''
        plan ee to pose. Invoke motion planning
        '''
        print("[FetchRobot]: plan_to_ee_pose")

        arm_path = self.fetch_mp.plan_to_pose_goal(tgt_pos, tgt_ori)
        if arm_path is not None:
            self.fetch_mp.execute(arm_path)
        else:
            print("[FetchRobot] ERROR: planning failed")

    def ctrl_to_ee_pose(self, tgt_pos, tgt_ori, duration = 3.0):
        '''
        move ee to pose in a straight line using joint position control, without collision checking
        '''
        print("[FetchRobot]: ctrl_to_ee_pose")

        ctrl_duration = duration * 0.8 # the last 20% of time is to wait for controller to settle
        settle_duration = duration * 0.2
        ctrl_steps = int(ctrl_duration * SIMULATION_FREQ)
        settle_steps = int(settle_duration * SIMULATION_FREQ)

        # attempt to guide the ee in straight line motion
        cur_pos, cur_ori = self.get_ee_pose()
        dist_pos = tgt_pos - cur_pos
        dist_ori = tgt_ori - cur_ori
        dist_pos_per_step = dist_pos / ctrl_steps
        dist_ori_per_step = dist_ori / ctrl_steps
        # control
        for _ in range(ctrl_steps):
            cur_pos += dist_pos_per_step
            cur_ori += dist_ori_per_step

            jointPoses = p.calculateInverseKinematics(self.fetchId,
                                                    self.fetch_ee_idx,
                                                    cur_pos,
                                                    cur_ori,
                                                    maxNumIterations=100,
                                                    residualThreshold=.01)
            # print(jointPoses)
            # print(start_y)

            for i in range(len(self.fetch_non_fixed_joints)):
                p.setJointMotorControl2(bodyIndex=self.fetchId,
                                        jointIndex=self.fetch_non_fixed_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
            p.stepSimulation()

        # wait for settle
        for _ in range(settle_steps):
            for i in range(len(self.fetch_non_fixed_joints)):
                p.setJointMotorControl2(bodyIndex=self.fetchId,
                                        jointIndex=self.fetch_non_fixed_joints[i],
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[i],
                                        targetVelocity=0,
                                        force=500,
                                        positionGain=0.03,
                                        velocityGain=1)
            p.stepSimulation()

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

        '''
        get the depth image as observation
        '''
        obs = self._get_depth()
        return obs # I need these two types of infos for different purposes: the former for learning, the latter for planning

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
        render the camera image to the window ONLY when this function is called.
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

    prm_types = [PType.BOX]
    prm_argss = [[0, 0.75, 0.55, 0.1, 0.1, 0.1, 0, 0, 0, 1]]
    obs = env.reset(prm_types=prm_types, prm_argss=prm_argss)
    obs, reward, done, info = env.step(([0, 0.6, 0.6], p.getQuaternionFromEuler([0, 0, math.radians(90)]), [0, 0.8, 0.6], p.getQuaternionFromEuler([0, 0, math.radians(90)])))

