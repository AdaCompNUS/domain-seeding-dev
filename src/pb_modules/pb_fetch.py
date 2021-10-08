import os
import math
import igibson
from igibson.robots.fetch_robot import Fetch
from igibson.utils.utils import parse_config
import igibson.external.pybullet_tools.utils as pb_utils
import time
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from pb_modules.fetch_pb_motion_planning import FetchMotionPlanningPyBullet
import pybullet as p

class FetchRobot:
    '''
    Fetch Robot, internally use iGibson's Fetch robot class
    '''
    def __init__(self, simulation_freq) -> None:
        config = parse_config(os.path.join(igibson.example_config_path, 'fetch_reaching.yaml'))
        self.fetch = Fetch(config)
        self.fetch.load()
        self.fetchId = self.fetch.robot_ids[0]
        self.simulation_freq = simulation_freq
        print("[FetchRobot]: id :".format(self.fetchId))

        # get arm joints ids
        self.fetch_non_fixed_joints = []
        self.fetch_non_fixed_joint_names = []
        fetch_num_joints = p.getNumJoints(self.fetchId)
        ee_idx = pb_utils.link_from_name(self.fetchId, "gripper_link")
        print("[FetchRobot]: end effector id :".format(ee_idx))
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
        # self.fetch.robot_specific_reset() # reset arm joint positions
        home_arm_positions = (0.3812102776278893, -0.5277115733278102, 1.1416475129889028, 3.1555377149919033, -1.1435719512665534, 2.4627678231156582, -2.1644855078770386, -0.42874708820913476)
        self.set_arm_joint_positions(home_arm_positions)
        self.set_base_pose([0,0,0], p.getQuaternionFromEuler([0, 0, math.radians(90)])) # reset base positions

    def set_base_pose(self, pos, orientation):
        self.fetch.set_position(pos)
        self.fetch.set_orientation(orientation)

    def set_arm_joint_positions(self, joint_positions):
        '''
        set arm to joint positions. This ignores simulation.
        '''
        arm_joints = joints_from_names(self.fetchId,
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
        set_joint_positions(self.fetchId, arm_joints, joint_positions)

    def set_ee_pose(self, tgt_pos, tgt_ori):
        '''
        set ee to target pose. This ignores simulation.
        '''
        joint_positions = self.fetch_mp.get_arm_joint_positions(tgt_pos, tgt_ori) # IK
        print(joint_positions)
        # joint_positions = (0.38116413675270866, 0.34336355518104916, 0.9988880515069087, -2.9291926623373805, -1.1392974724751825, 0.19869520116838874, 2.161313899113838, 3.1359014553879114)
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

    def ctrl_to_ee_pose(self, tgt_pos, tgt_ori, duration = 2.0, quick_mode=False, step_cb = None, camera_freq=10):
        '''
        move ee to pose in a straight line using joint position control, without collision checking
        '''
        print("[FetchRobot]: ctrl_to_ee_pose")

        ctrl_duration = duration * 0.8 # the last 20% of time is to wait for controller to settle
        settle_duration = duration * 0.2
        ctrl_steps = int(ctrl_duration * self.simulation_freq)
        settle_steps = int(settle_duration * self.simulation_freq)
        camera_steps = int(1.0 / camera_freq * self.simulation_freq)

        # attempt to guide the ee in straight line motion
        cur_pos, cur_ori = self.get_ee_pose()
        dist_pos = tgt_pos - cur_pos
        dist_ori = tgt_ori - cur_ori
        dist_pos_per_step = dist_pos / ctrl_steps
        dist_ori_per_step = dist_ori / ctrl_steps
        # control
        # print(ctrl_steps)
        # print(settle_steps)
        for step in range(ctrl_steps):
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


            if step_cb and step % camera_steps == 0:
                step_cb()
            p.stepSimulation()
            if not quick_mode:
                time.sleep(1.0 / self.simulation_freq)

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
            if step_cb and step % camera_steps == 0:
                step_cb()
            p.stepSimulation()
            if not quick_mode:
                time.sleep(1.0 / self.simulation_freq)
