import igibson
from igibson.envs.igibson_env import iGibsonEnv
from time import time, sleep
import os
from igibson.utils.assets_utils import download_assets, download_demo_data
import numpy as np
from igibson.external.pybullet_tools.utils import control_joints
from igibson.external.pybullet_tools.utils import get_joint_positions
from igibson.external.pybullet_tools.utils import get_joint_velocities
from igibson.external.pybullet_tools.utils import get_max_limits
from igibson.external.pybullet_tools.utils import get_min_limits
from igibson.external.pybullet_tools.utils import plan_joint_motion
from igibson.external.pybullet_tools.utils import link_from_name
from igibson.external.pybullet_tools.utils import joints_from_names
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.external.pybullet_tools.utils import get_sample_fn
from igibson.external.pybullet_tools.utils import set_base_values_with_z
from igibson.external.pybullet_tools.utils import get_base_values
from igibson.external.pybullet_tools.utils import plan_base_motion_2d
from igibson.external.pybullet_tools.utils import get_moving_links
from igibson.external.pybullet_tools.utils import is_collision_free

from igibson.utils.utils import rotate_vector_2d, rotate_vector_3d
from igibson.utils.utils import l2_distance, quatToXYZW
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.objects.visual_marker import VisualMarker
from transforms3d import euler

import pybullet as p

class FetchMotionPlanningPyBullet(object):
    """
    Motion planner wrapper that supports both base and arm motion
    """

    def __init__(self,
                 robot=None,
                 arm_mp_algo='birrt',
                 optimize_iter=0,
                 fine_motion_plan=True):
        """
        Get planning related parameters.
        """
        # self.env = env
        # assert 'occupancy_grid' in self.env.output
        # get planning related parameters from env
        self.robot = robot
        self.robot_id = robot.robot_ids[0]
        # self.mesh_id = self.scene.mesh_body_id
        # mesh id should not be used
        self.ee_link_idx = 19
        self.arm_mp_algo = arm_mp_algo
        self.fine_motion_plan = fine_motion_plan

        # if self.env.simulator.viewer is not None:
        #     self.env.simulator.viewer.setup_motion_planner(self)

        self.setup_arm_mp()
        self.arm_interaction_length = 0.2
        self.marker = None
        self.marker_direction = None

        # if self.mode in ['gui', 'iggui']:
        #     self.marker = VisualMarker(
        #         radius=0.04, rgba_color=[0, 0, 1, 1])
        #     self.marker_direction = VisualMarker(visual_shape=p.GEOM_CAPSULE, radius=0.01, length=0.2,
        #                                          initial_offset=[0, 0, -0.1], rgba_color=[0, 0, 1, 1])
        #     self.env.simulator.import_object(
        #         self.marker, use_pbr=False)
        #     self.env.simulator.import_object(
        #         self.marker_direction, use_pbr=False)

    def setup_arm_mp(self):
        """
        Set up arm motion planner
        """
        self.arm_default_joint_positions = (0.10322468280792236,
                                            -1.414019864768982,
                                            1.5178184935241699,
                                            0.8189625336474915,
                                            2.200358942909668,
                                            2.9631312579803466,
                                            -1.2862852996643066,
                                            0.0008453550418615341)
        self.arm_joint_ids = joints_from_names(self.robot_id,
                                                [
                                                    'torso_lift_joint',
                                                    'shoulder_pan_joint',
                                                    'shoulder_lift_joint',
                                                    'upperarm_roll_joint',
                                                    'elbow_flex_joint',
                                                    'forearm_roll_joint',
                                                    'wrist_flex_joint',
                                                    'wrist_roll_joint'
                                                ])

        self.arm_joint_ids_all = get_moving_links(self.robot_id, self.arm_joint_ids)
        self.arm_joint_ids_all = [item for item in self.arm_joint_ids_all if item != self.robot.end_effector_part_index()]
        # print(self.arm_joint_ids_all)
        self.arm_ik_threshold = 0.05

        self.mp_obstacles = []
        # if type(self.env.scene) == StaticIndoorScene:
        #     if self.env.scene.mesh_body_id is not None:
        #         self.mp_obstacles.append(self.env.scene.mesh_body_id)
        # elif type(self.env.scene) == InteractiveIndoorScene:
        #     self.mp_obstacles.extend(self.env.scene.get_body_ids())

    def add_obstacle_body(self, body_id):
        self.mp_obstacles.append(body_id)

    def remove_obstacle_body(self, body_id):
        self.mp_obstacles.remove(body_id)

    def plan_to_joint_goal(self, arm_joint_positions):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :return: arm trajectory or None if no plan can be found
        """
        disabled_collisions = {
            (link_from_name(self.robot_id, 'torso_lift_link'),
                link_from_name(self.robot_id, 'torso_fixed_link')),
            (link_from_name(self.robot_id, 'torso_lift_link'),
                link_from_name(self.robot_id, 'shoulder_lift_link')),
            (link_from_name(self.robot_id, 'torso_lift_link'),
                link_from_name(self.robot_id, 'upperarm_roll_link')),
            (link_from_name(self.robot_id, 'torso_lift_link'),
                link_from_name(self.robot_id, 'forearm_roll_link')),
            (link_from_name(self.robot_id, 'torso_lift_link'),
                link_from_name(self.robot_id, 'elbow_flex_link'))}

        if self.fine_motion_plan:
            self_collisions = True
            mp_obstacles = self.mp_obstacles
        else:
            self_collisions = False
            mp_obstacles = []

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()

        # allow_collision_links = []
        allow_collision_links = [19]

        arm_path = plan_joint_motion(
            self.robot_id,
            self.arm_joint_ids,
            arm_joint_positions,
            disabled_collisions=disabled_collisions,
            self_collisions=self_collisions,
            obstacles=mp_obstacles,
            algorithm=self.arm_mp_algo,
            allow_collision_links=allow_collision_links,
        )
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        return arm_path

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        max_limits = [0., 0.] + \
            get_max_limits(self.robot_id, self.arm_joint_ids)
        min_limits = [0., 0.] + \
            get_min_limits(self.robot_id, self.arm_joint_ids)
        # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
        min_limits[2] += 0.02
        rest_position = [0., 0.] + \
            list(get_joint_positions(self.robot_id, self.arm_joint_ids))
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.1 for _ in joint_range]

        return (
            max_limits, min_limits, rest_position,
            joint_range, joint_damping
        )

    def get_arm_joint_positions(self, arm_ik_goal, arm_ik_ori):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None

        :param arm_ik_goal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = \
            self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 3
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)
        state_id = p.saveState()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        # find collision-free IK solution for arm_subgoal
        while n_attempt < max_attempt:
            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robot.end_effector_part_index(),
                targetPosition=arm_ik_goal,
                targetOrientation=arm_ik_ori,
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_DLS,
                maxNumIterations=500)

            arm_joint_positions = arm_joint_positions[2:10]

            set_joint_positions(
                self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(
                self.robot.get_end_effector_position(), arm_ik_goal)
            # print('dist', dist)
            if dist > self.arm_ik_threshold:
                # print('[FetchPbMP/IK] WARN: dist too large')
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            # self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            # set_base_values_with_z(
            #     self.robot_id, base_pose, z=self.initial_height)
            # self.reset_object_states()
            # TODO: have a princpled way for stashing and resetting object states

            # arm should not have any collision
            collision_free = is_collision_free(
                body_a=self.robot_id,
                link_a_list=self.arm_joint_ids)

            if not collision_free:
                n_attempt += 1
                # print('[FetchPbMP/IK] WARN: arm has collision')
                continue

            # gripper should not have any self-collision
            collision_free = is_collision_free(
                body_a=self.robot_id,
                link_a_list=[
                    self.robot.end_effector_part_index()],
                body_b=self.robot_id)
            if not collision_free:
                n_attempt += 1
                # print('[FetchPbMP/IK] WARN: gripper has collision')
                continue

            #self.episode_metrics['arm_ik_time'] += time() - ik_start
            #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            p.restoreState(state_id)
            p.removeState(state_id)
            return arm_joint_positions

        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        #self.episode_metrics['arm_ik_time'] += time() - ik_start
        return None

    def plan_to_pose_goal(self, ee_pos, ee_ori, num_of_attempts=5):
        """
        Attempt to reach arm arm_joint_positions and return arm trajectory
        If failed, reset the arm to its original pose and return None

        :param arm_joint_positions: final arm joint position to reach
        :return: arm trajectory or None if no plan can be found
        """

        joint_positions = self.get_arm_joint_positions(ee_pos, ee_ori)
        if joint_positions is None:
            return None
        arm_path = None
        for _ in range(num_of_attempts):
            arm_path = self.plan_to_joint_goal(joint_positions)
            if arm_path is not None:
                return arm_path

        return None

    def execute(self, path):
        # print(len(path))
        for q in path:
            set_joint_positions(self.robot_id, self.arm_joint_ids, q)
            # #wait_if_gui('Continue?')
            sleep(0.01)
            # for i in range(len(self.arm_joint_ids_all)):
            #     p.setJointMotorControl2(bodyIndex=self.robot_id,
            #                     jointIndex=self.arm_joint_ids[i],
            #                     controlMode=p.POSITION_CONTROL,
            #                     targetPosition=q[i],
            #                     targetVelocity=0,
            #                     force=500,
            #                     positionGain=0.03,
            #                     velocityGain=1)
            # p.stepSimulation()
            # sleep(1/240.0)