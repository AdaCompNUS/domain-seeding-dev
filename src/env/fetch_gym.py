import gym
import numpy as np
from gym import spaces
import pybullet as p
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
from utils.variables import NUM_PARAMS_TO_LEARN, PARAM_SEARCH_BOUNDS
from pb_modules.pb_fetch import FetchRobot
from pb_modules.pb_objects import Object
from random_sim.domain_randomization import ObjectRandomizer
from models.exploration import ExplorationPolicy

# ------- Settings ----------
SIMULATION_FREQ = 240
CAMERA_FREQ = 5  # Due to the speed of getCameraImage, this has to be slower than 20Hz depending on the machine


class FetchPushEnv(gym.Env):
    MOVE_PENALTY_FACTOR = -0.1
    ROTATE_PENALTY_FACTOR = -0.1

    GOAL_REWARD = 1.0

    def __init__(self, gui=False):
        super(FetchPushEnv, self).__init__()

        # connect to pybullet engine
        if gui:
            p.connect(p.GUI)  # no GUI
        else:
            p.connect(p.DIRECT)  # no GUI
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / SIMULATION_FREQ)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load floor to prevent objects from falling through floor
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        # internal attributes
        ws_range = 10  # T.B.D
        obs_res = 640  # T.B.D
        max_depth = 1.0  # T.B.D
        self.action_space = spaces.Box(low=-ws_range, high=ws_range,
                                       shape=(4,))  # A box in R^4. Each coordinate is bounded.
        self.param_space = spaces.Box(low=np.atleast_1d([b[0] for b in PARAM_SEARCH_BOUNDS]),
                                      high=np.atleast_1d([b[1] for b in PARAM_SEARCH_BOUNDS]),
                                      shape=(NUM_PARAMS_TO_LEARN,))  # A box in R^4. Each coordinate is bounded.
        self.observation_space = spaces.Box(low=0.0, high=max_depth, shape=(int(CAMERA_FREQ * 3.2), obs_res, obs_res),
                                            dtype=np.float32)  # depth image

        # robot
        self.robot = FetchRobot(SIMULATION_FREQ)

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

        self.prm_types = None
        self.prm_argss = None

    def reset(self, reset_env=True, reset_robot=True, reset_goal=True, prm_types: List[PType] = None,
              prm_argss: List[List[float]] = None, goal: TaskGoal = None):
        print('[actor.py] Reset gym env')
        if reset_env:  # re-generate the scene
            if self.object:
                del self.object
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

        if reset_goal:  # reset manipulation goal
            self.goal = goal

        if reset_robot:
            '''
            reset robot to default pose
            '''
            self.robot.reset()

        info = {}
        info['obj_state'] = self.object.get_states()  # get true object state from simulator

        '''
        get the depth image as observation
        '''
        obs = self._get_depth()
        # print('output observation sizes {}'.format(obs.shape))
        # sys.stdout.flush()
        return obs, info  # I need these two types of infos for different purposes: the former for learning, the latter for planning

    def step(self, action:ParameterizedPolicy, mode='normal'):
        '''
        Execute the action using IK control and simulate object movement

        Args:
            action: [start_ee_pose, start_ee_ori, end_ee_pose, end_ee_ori, duration],
                    duration controls the speed of the push action and therefore the force.
        '''
        print('[actor.py] Step')
        self.rgb_seq.clear()
        self.depth_seq.clear()
        self.mask_seq.clear()

        start_ee_pos, start_ee_ori, end_ee_pos, end_ee_ori, duration = action.serialize()
        if mode == 'normal':
            '''
            Run simulation with real-time speed, e.g., for visualizing an episode
            '''
            # self.robot.set_ee_pose(start_ee_pos, start_ee_ori)
            self.robot.ctrl_to_ee_pose(start_ee_pos, start_ee_ori, duration=2)
            self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration, step_cb=self.save_render,
                                       camera_freq=CAMERA_FREQ)
            # print(self.robot.get_ee_pose())

        elif mode == 'super':
            '''
            Both planning and RL require to run simulation FASTER than real time.
            '''

            # self.robot.set_ee_pose(start_ee_pos, start_ee_ori)
            self.robot.ctrl_to_ee_pose(start_ee_pos, start_ee_ori, duration=2, quick_mode=True)
            self.robot.ctrl_to_ee_pose(end_ee_pos, end_ee_ori, duration=duration, quick_mode=True,
                                       step_cb=self.save_render, camera_freq=CAMERA_FREQ)
        else:
            raise Exception('Unsupported simulation mode {}'.format(mode))

        info = {}
        info['obj_state'] = self.object.get_states()  # get true object state from simulator
        obs = self._process_img_seq(self.depth_seq, self.mask_seq)  # get depth images as obs
        # print('output observation sizes {}'.format(obs.shape))
        # sys.stdout.flush()
        reward, succeed = self.reward_function(info['obj_state'], action)

        return obs, reward, succeed, info  # include true state of the manipulated object (x,y,z, quat) in info.

    def reward_function(self, obj_state:ObjState, action:ParameterizedPolicy):
        succeed = self.goal.eval(obj_state)
        reward = self.MOVE_PENALTY_FACTOR * action.trans_dist() + self.ROTATE_PENALTY_FACTOR * action.rot_dist()
        if succeed:
            reward += self.GOAL_REWARD
        return reward, succeed

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
        # print(end_time - start_time)
        self.rgb_seq.append(rgb)
        self.depth_seq.append(depth)
        self.mask_seq.append(mask)
        # print("here")

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
            ret = np.zeros((2 * depth_seq.shape[0], depth_seq.shape[1], depth_seq.shape[2]), depth_seq.dtype)
            for i in range(depth_seq.shape[0]):
                ret[2 * i:2 * (i + 1)] = np.array([robot_depth_seq[i], obj_depth_seq[i]])
        else:
            ret = np.array([robot_depth_seq, obj_depth_seq])
        # return robot_depth_seq, table_depth_seq, obj_depth_seq
        return ret

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
    env = FetchPushEnv(gui=True)
    env.render()

    randomizer = ObjectRandomizer()
    exp_policy = ExplorationPolicy()
    prm_types, prm_argss = randomizer.sample(num_objects=1)
    # prm_types = [PType.BOX]
    # prm_argss = [[0, 0.75, 0.55, 0.1, 0.1, 0.1, 0, 0, 0, 1]]
    obs, info = env.reset(prm_types=prm_types, prm_argss=prm_argss, goal=TaskGoal(GType.BEYOND, [0, 1.0, 0.8]))
    # env.render()
    start_time = time.time()
    obs, reward, done, info1 = env.step(exp_policy.next_action(info['obj_state']))
    end_time = time.time()
    print(end_time - start_time)

    obs, info = env.reset(prm_types=prm_types, prm_argss=prm_argss, goal=TaskGoal(GType.BEYOND, [0, 1.0, 0.8]))
    # env.render()
    start_time = time.time()
    obs, reward, done, info2 = env.step(exp_policy.next_action(info['obj_state']), mode='super')
    end_time = time.time()
    print(end_time - start_time)

    print(info1['obj_state'].pos, info1['obj_state'].quaternion)
    print(info2['obj_state'].pos, info2['obj_state'].quaternion)
    # env.render()
    input()
