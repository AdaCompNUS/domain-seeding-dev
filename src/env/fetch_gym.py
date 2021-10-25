import gym
import numpy as np
from gym import spaces
import cv2
import pybullet_data
import os
import sys
import time
from pathlib import Path

ws_root = Path(os.path.realpath(__file__)).parent.parent
print(f'workspace root: {ws_root}')
sys.stdout.flush()
sys.path.append(str(ws_root))

from utils.classes import *
from utils.variables import LOGGING_MIN, LOGGING_INFO, LOGGING_DEBUG
from utils.functions import error_handler
from pb_modules.pb_fetch import FetchRobot
from pb_modules.pb_objects import Object
from pb_modules.debug_draw import FrameDrawManager
from random_sim.domain_randomization import ObjectRandomizer, PrimitiveRandomizer
from models.exploration import ExplorationPolicy

# ------- Settings ----------
SIMULATION_FREQ = 60
CAMERA_FREQ = 5  # Due to the speed of getCameraImage, this has to be slower than 20Hz depending on the machine


class FetchPushEnv(gym.Env):
    MOVE_PENALTY_FACTOR = -0.0
    ROTATE_PENALTY_FACTOR = -0.0
    GOAL_REWARD = 5.0
    NUM_OBS_FRAMES = int(CAMERA_FREQ * (ExplorationPolicy.DURATION + FetchRobot.SETTLE_DURATION))
    NUM_DEPTH_SEGMENTS = 2
    WS_RANGE = 10  # T.B.D
    OBS_RES = 640  # T.B.D
    MAX_DEPTH = 1.0  # T.B.D

    def __init__(self, gui=False, logging_level=LOGGING_MIN):
        super(FetchPushEnv, self).__init__()

        # connect to pybullet engine
        if gui:
            p.connect(p.GUI)  # no GUI
        else:
            p.connect(p.DIRECT)  # no GUI
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / SIMULATION_FREQ)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load floor to prevent objects from falling through floor
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        # internal attributes

        self.action_space = spaces.Box(low=-self.WS_RANGE, high=self.WS_RANGE,
                                       shape=(4,))  # A box in R^4. Each coordinate is bounded.
        # self.param_space = spaces.Box(low=np.atleast_1d(PrimitiveRandomizer.SEARCH_LB),
        #                               high=np.atleast_1d(PrimitiveRandomizer.SEARCH_UB),
        #                               shape=(ObjState.DIM_PREDICTIONS,))  # A box in R^4. Each coordinate is bounded.
        self.param_space = spaces.Box(low=-10.0,  # dummy value
                                      high=10.0,  # dummy value
                                      shape=(ObjState.DIM_PREDICTIONS,))  # A box in R^4. Each coordinate is bounded.
        self.observation_space = spaces.Box(low=0.0, high=self.MAX_DEPTH,
                                            shape=(self.NUM_DEPTH_SEGMENTS * self.NUM_OBS_FRAMES, self.OBS_RES, self.OBS_RES),
                                            dtype=np.float32)  # depth image

        # robot
        self.robot = FetchRobot(SIMULATION_FREQ, logging_level)

        # table
        self.table = self._add_table()

        # objects
        self.object = None  # no object at start
        self.goal = None  # no task goal at start

        # camera
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        print('numpy enabled:', p.isNumpyEnabled())
        self.pixelWidth = 640
        self.pixelHeight = 640
        camTargetPos, camDistance = [0, 0.8, 0], 1.02
        pitch, roll = -90, 0
        upAxisIndex = 2
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, 0, pitch, roll, upAxisIndex)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=90, aspect=float(self.pixelWidth) / self.pixelHeight,
                                                        nearVal=0.1, farVal=0.6)
        self.rgb_seq = []
        self.depth_seq = []
        self.mask_seq = []
        self.mode = None

        self.prm_types = None
        self.prm_argss = None
        self.logging_level = logging_level
        self.frame_visualizer = FrameDrawManager()

    def print_with_level(self, msg, level=LOGGING_DEBUG):
        if self.logging_level >= level:
            print(msg)

    def reset(self, mode='normal', reset_env=True, reset_object=False, reset_robot=True, reset_goal=True,
              prm_types: List[PType] = None, prm_argss: List[List[float]] = None, goal: TaskGoal = None,
              need_return=True):
        try:
            self.mode = mode
            # if mode == 'super' or mode == 'quick':
            #     p.setRealTimeSimulation(0)
            # else:
            #     p.setRealTimeSimulation(1)

            self.print_with_level('[fetch_gym.py] Reset gym env', LOGGING_INFO)
            if reset_env:  # re-generate the scene
                if self.object:
                    del self.object
                if self.mode == 'normal':
                    self.frame_visualizer = FrameDrawManager()
                '''
                re-locate the table, allow for slight noise in position and orientation
                '''
                self._remove_table()
                self.table = self._add_table()

                # re-spawn object on the table, with some user-given types and parameters
                if prm_types and prm_argss:
                    self.object = Object(prm_types, prm_argss)
                    self.prm_types = prm_types
                    self.prm_argss = prm_argss
                else:  # reuse existing parameters
                    self.object = Object(self.prm_types, self.prm_argss)

            self.print_with_level(f'[fetch_gym.py] Reset object {self.object}', LOGGING_INFO)
            if reset_object:
                self.object.reset()

            self.print_with_level('[fetch_gym.py] Reset goal', LOGGING_INFO)
            if reset_goal:  # reset manipulation goal
                self.goal = goal

            self.print_with_level('[fetch_gym.py] Reset robot', LOGGING_INFO)
            if reset_robot:
                '''
                reset robot to default pose
                '''
                self.robot.reset()

            if mode == 'normal':
                self.frame_visualizer.add_inertial_frame(self.object.primitives[0].id, -1)
                self.frame_visualizer.add_collision_frame(self.object.primitives[0].id, -1)

            if need_return:
                info = {}
                info['obj_state'] = self.object.get_states()  # get true object state from simulator

                '''
                get the depth image as observation
                '''
                obs = self._get_depth()
                # print('output observation sizes {}'.format(obs.shape))
                # sys.stdout.flush()
                return obs, info  # I need these two types of infos for different purposes: the former for learning, the latter for planning
            else:
                return None, None

        except Exception as e:
            error_handler(e)

    def step(self, action: ParameterizedPolicy, generate_obs=True):
        '''
        Execute the action using IK control and simulate object movement

        Args:
            action: [start_ee_pose, start_ee_ori, end_ee_pose, end_ee_ori, duration],
                    duration controls the speed of the push action and therefore the force.
        '''
        try:
            self.print_with_level('[fetch_gym.py] Step', LOGGING_INFO)
            start_time = time.time()

            self.rgb_seq.clear()
            self.depth_seq.clear()
            self.mask_seq.clear()

            start_ee_pos, start_ee_ori, end_ee_pos, end_ee_ori, duration = action.refract()
            self.print_with_level(f'End pos {end_ee_pos}', LOGGING_INFO)
            self.print_with_level(f'Start pos {start_ee_pos}', LOGGING_INFO)
            self.print_with_level(f'End ori {end_ee_ori}', LOGGING_INFO)
            self.print_with_level(f'Start ori {start_ee_ori}', LOGGING_INFO)
            self.print_with_level(f'Duration {duration}', LOGGING_INFO)

            if not self.robot.set_ee_pose(end_ee_pos, end_ee_ori):
                self.print_with_level('End ee pose not feasible', LOGGING_INFO)
                return None, -1000, False, {}

            if not self.robot.set_ee_pose(start_ee_pos, start_ee_ori):
                self.print_with_level('Start ee pose not feasible', LOGGING_INFO)
                return None, -1000, False, {}

            if self.mode == 'normal':
                '''
                Run simulation with real-time speed, e.g., for visualizing an episode
                '''
                # self.robot.ctrl_to_ee_pose(start_ee_pos, start_ee_ori, duration=0.5)
                if generate_obs:
                    self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration,
                                               step_cb=self.save_render, camera_freq=CAMERA_FREQ,
                                               frame_draw_manager=self.frame_visualizer)
                else:
                    self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration,
                                               frame_draw_manager=self.frame_visualizer)
                # print(self.robot.get_ee_pose())

            elif self.mode == 'quick':
                '''
                Both planning and RL require to run simulation FASTER than real time.
                '''
                # self.robot.ctrl_to_ee_pose(start_ee_pos, start_ee_ori, duration=2, quick_mode=True)
                if generate_obs:
                    self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration, quick_mode=True,
                                               step_cb=self.save_render, camera_freq=CAMERA_FREQ,
                                               frame_draw_manager=None)
                else:
                    self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration,
                                               frame_draw_manager=None)
            elif self.mode == 'super':
                '''
                Both planning and RL require to run simulation FASTER than real time.
                '''
                # self.robot.ctrl_to_ee_pose(start_ee_pos, start_ee_ori, duration=1, quick_mode=True)
                self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration, quick_mode=True)
            else:
                raise Exception('Unsupported simulation mode {}'.format(self.mode))
            self.print_with_level('Simulation mode {}'.format(self.mode), LOGGING_INFO)
            self.print_with_level('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`move duration {}'.format(time.time() - start_time),
                                  LOGGING_DEBUG)
            sys.stdout.flush()

            info = {}
            info['obj_state'] = self.object.get_states()  # get true object state from simulator
            obs = None
            if generate_obs:
                obs = self._process_img_seq(self.depth_seq, self.mask_seq)  # get depth images as obs

            if self.goal:
                reward, succeed = self.reward_function(info['obj_state'], action)
            elif self.mode != 'super' and generate_obs and obs is None:
                self.print_with_level('[fetch_gym] No obs generated while requesting obs data', LOGGING_MIN)
                reward, succeed = -1000, False
            else:
                reward, succeed = 0.0, True

            return obs, reward, succeed, info  # include true state of the manipulated object (x,y,z, quat) in info.
        except Exception as e:
            error_handler(e)

    def reward_function(self, obj_state: ObjState, action: ParameterizedPolicy):
        progress = self.goal.eval(obj_state)
        self.print_with_level(f'Object end pos {obj_state.pos}', LOGGING_INFO)
        self.print_with_level(f'Task status {progress}', LOGGING_INFO)
        reward = self.MOVE_PENALTY_FACTOR * action.trans_dist() + self.ROTATE_PENALTY_FACTOR * action.rot_dist()
        # if succeed:
        reward += progress * self.GOAL_REWARD
        return reward, bool(progress >= 0)

    def render(self):
        '''
        render the camera image to the window using OpenCV ONLY when this function is called.
        other than this function, the system should run in headless mode.
        '''
        p.getCameraImage(self.pixelWidth,
                         self.pixelHeight,
                         viewMatrix=self.viewMatrix,
                         projectionMatrix=self.proj_matrix,
                         shadow=1,
                         lightDirection=[1, 1, 1],
                         renderer=p.ER_BULLET_HARDWARE_OPENGL)

    def save_render(self):
        '''
        Save the camera image during the action. Note this happens at CAMERA_FREQ which is slower than SIMULATION_FREQ
        '''
        # start_time = time.time()
        w, h, rgb, depth, mask = p.getCameraImage(self.pixelWidth,
                                                  self.pixelHeight,
                                                  viewMatrix=self.viewMatrix,
                                                  projectionMatrix=self.proj_matrix,
                                                  shadow=1,
                                                  lightDirection=[1, 1, 1],
                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # end_time = time.time()
        # print(f'[fetch_gym.py ]Depth image query time: {end_time - start_time}')
        '''The following three lines blocked just for testing !!!'''
        self.rgb_seq.append(rgb)
        self.depth_seq.append(depth)
        self.mask_seq.append(mask)
        cv2.imwrite(f'snapshot{len(self.rgb_seq)}.jpg', rgb)

    def _remove_table(self):
        '''
        Remove table from scene
        '''
        self.robot.fetch_mp.remove_obstacle_body(self.table)
        p.removeBody(self.table)

    def _add_table(self, pos=[0, 0.8, 0.25], orientation=[0, 0, 0, 1]):
        '''
        Add table to scene
        pos, the center position of table. Default is 0.8m in front of robot
        '''
        # table = p.createCollisionShape(p.GEOM_MESH, fileName="table/table.obj")
        table = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.25])
        tableId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table, basePosition=pos,
                                    baseOrientation=orientation)

        self.robot.fetch_mp.add_obstacle_body(tableId)

        return tableId

    def _process_img_seq(self, depth_seq, mask_seq):
        # print(len(self.rgb_seq))
        # print(len(self.depth_seq))
        # print(len(self.mask_seq))
        try:
            depth_seq = np.array(depth_seq)
            mask_seq = np.array(mask_seq)
            # print(mask_seq.max())
            # print(mask_seq.min())
            # print(depth_seq.max())
            # print(depth_seq.min())

            robot_idx = (mask_seq == 1)
            non_robot_idx = (mask_seq != 1)

            robot_depth_seq = np.copy(depth_seq)
            robot_depth_seq[non_robot_idx] = 1.0
            # robot_depth_seq = depth_seq[robot_idx]
            table_depth_seq = np.copy(depth_seq)
            table_depth_seq[robot_idx] = 1.0
            # print(table_depth_seq.max())
            # print(table_depth_seq.min())

            # use depth to segment table and object
            table_mask = table_depth_seq > 0.95
            obj_mask = table_depth_seq <= 0.95
            obj_depth_seq = np.copy(table_depth_seq)
            obj_depth_seq[table_mask] = 1.0
            table_depth_seq[obj_mask] = 1.0

            # for i in range(0, 100, 10):
            #     sample = obj_depth_seq[i]
            #     cv2.imshow("tmp", sample)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # for i in range(0, 100, 10):
            #     sample = table_depth_seq[i]
            #     cv2.imshow("tmp", sample)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()
            # for i in range(0, 100, 10):
            #     sample = robot_depth_seq[i]
            #     cv2.imshow("tmp", sample)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

            if len(depth_seq.shape) > 2:
                assert(depth_seq.shape[0] == self.NUM_OBS_FRAMES)
                ret = np.zeros((self.NUM_DEPTH_SEGMENTS * depth_seq.shape[0],
                                depth_seq.shape[1], depth_seq.shape[2]), depth_seq.dtype)
                for i in range(depth_seq.shape[0]):
                    ret[2 * i:2 * (i + 1)] = np.array([robot_depth_seq[i], obj_depth_seq[i]])
            else:
                ret = np.array([robot_depth_seq, obj_depth_seq])
            # return robot_depth_seq, table_depth_seq, obj_depth_seq
            return ret
        except Exception as e:
            error_handler(e)

    def _get_depth(self):
        w, h, rgb, depth, mask = p.getCameraImage(self.pixelWidth,
                                                  self.pixelHeight,
                                                  viewMatrix=self.viewMatrix,
                                                  projectionMatrix=self.proj_matrix,
                                                  shadow=0,
                                                  lightDirection=[1, 1, 1],
                                                  renderer=p.ER_BULLET_HARDWARE_OPENGL)
        ret = self._process_img_seq(depth, mask)
        # return robot_d, table_d, obj_d
        return ret


if __name__ == '__main__':
    env = FetchPushEnv(gui=False, logging_level=LOGGING_MIN)
    # env.render()

    logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "timings")

    # p.setPhysicsEngineParameter(numSolverIterations=1)
    # val = p.getPhysicsEngineParameters()
    # print(f'Physics engine set up: \n{val}')

    randomizer = ObjectRandomizer()
    exp_policy = ExplorationPolicy()
    prm_types, prm_argss = randomizer.sample(num_objects=1)

    obs, info = env.reset(prm_types=prm_types, prm_argss=prm_argss, goal=TaskGoal(GType.BEYOND, [0, 1.0, 0.8]),
                          reset_object=False, mode='quick')
    # env.render()
    # # env.render()
    # start_time = time.time()
    # obs, reward, done, info1 = env.step(exp_policy.next_action(info['obj_state']), generate_obs=False)
    # end_time = time.time()
    # print(end_time - start_time)

    while True:
        env.reset(reset_env=False, reset_object=True, reset_robot=True, reset_goal=False, need_return=False)
        # env.render()
        start_time = time.time()
        _, reward, done, info2 = env.step(exp_policy.next_action(info['obj_state']), generate_obs=False)
        end_time = time.time()
        print(end_time - start_time)

        # print(info1['obj_state'].pos, info1['obj_state'].quaternion)
        # print(info2['obj_state'].pos, info2['obj_state'].quaternion)
        # env.render()
        input()

    p.stopStateLogging(logId)
