import os
import math
import igibson
from igibson.robots.fetch_robot import Fetch
from igibson.utils.utils import parse_config
import igibson.external.pybullet_tools.utils as pb_utils

from fetch_pb_motion_planning import FetchMotionPlanningPyBullet
from igibson.simulator import Simulator
import pybullet as p


# -------- Constants ---------
SIMULATION_FREQ = 240


class FetchRobot:
    '''
    Fetch Robot, internally use iGibson's Fetch robot class
    '''
    def __init__(self) -> None:
        config = parse_config(os.path.join(igibson.example_config_path, 'fetch_reaching.yaml'))
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
            return True
        else:
            print("[FetchRobot] ERROR: planning failed")
            return False

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
