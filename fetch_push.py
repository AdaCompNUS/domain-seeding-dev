from gibson2.robots.fetch_robot import Fetch
from gibson2.utils.utils import parse_config
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions, link_from_name
import gibson2.external.pybullet_tools.utils as pb_utils
import os
import time
import numpy as np
import pybullet as p
import pybullet_data
import gibson2
import math

from fetch_pb_motion_planning import FetchMotionPlanningPyBullet

FETCH_EE_IDX = 19

def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    # Add fetch
    config = parse_config(os.path.join(gibson2.example_config_path, 'fetch_reaching.yaml'))
    fetch = Fetch(config)
    fetch.load()
    fetchId = fetch.robot_ids[0]
    fetch.robot_specific_reset()
    fetch.set_orientation(p.getQuaternionFromEuler([0, 0, math.radians(90)]))

    # print fetch information
    fetch_non_fixed_joints = []
    fetch_non_fixed_joint_names = []
    print(fetchId)
    fetch_num_joints = p.getNumJoints(fetchId)
    ee_idx = link_from_name(fetchId, "gripper_link")
    print(ee_idx)
    for i in range(fetch_num_joints):
        joint_info = pb_utils.get_joint_info(fetchId, i)
        if joint_info.jointType!= p.JOINT_FIXED:
            fetch_non_fixed_joints.append(i)
            fetch_non_fixed_joint_names.append(joint_info.jointName)

    # Add table
    table = p.createCollisionShape(p.GEOM_MESH, fileName="table/table.obj")
    tableId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table, basePosition=[0, 0.8, 0])

    # Add objects
    sphereRadius = 0.05
    colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sphereRadius, sphereRadius, sphereRadius])

    ## single box
    box1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=colBoxId, basePosition=[0, 0.75, 0.55])

    ## multi box
    mass = 1
    visualShapeId = -1
    link_Masses = [1, 1]
    linkCollisionShapeIndices = [colBoxId, colBoxId]
    linkVisualShapeIndices = [-1, -1]
    linkPositions = [[0.1, 0.05, 0], [0.1, -0.05, 0]]
    linkOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
    linkInertialFramePositions = [[0, 0, 0], [0, 0, 0]]
    linkInertialFrameOrientations = [[0, 0, 0, 1], [0, 0, 0, 1]]
    indices = [0, 0]
    jointTypes = [p.JOINT_FIXED, p.JOINT_FIXED]
    axis = [[0, 0, 1], [0, 0, 1]]
    multibox = p.createMultiBody(mass,
                                colBoxId,
                                visualShapeId,
                                [0, 0.9, 0.55],
                                [0, 0, 0, 1],
                                linkMasses=link_Masses,
                                linkCollisionShapeIndices=linkCollisionShapeIndices,
                                linkVisualShapeIndices=linkVisualShapeIndices,
                                linkPositions=linkPositions,
                                linkOrientations=linkOrientations,
                                linkInertialFramePositions=linkInertialFramePositions,
                                linkInertialFrameOrientations=linkInertialFrameOrientations,
                                linkParentIndices=indices,
                                linkJointTypes=jointTypes,
                                linkJointAxis=axis)
    p.changeDynamics(multibox, -1, lateralFriction=0.01)

    # set robot to initial position
    # start_y = 0.6
    # jointPoses = p.calculateInverseKinematics(fetchId,
    #                                         FETCH_EE_IDX,
    #                                         [0, start_y, 0.6],
    #                                         p.getQuaternionFromEuler([0, 0, math.radians(90)]),
    #                                         maxNumIterations=100,
    #                                         residualThreshold=.01)
    # print(jointPoses)
    # fetch.set_joint_positions(jointPoses[2:]) # ignore first 2 wheel joints

    print("---------------------Motion planning")
    fetch_mp = FetchMotionPlanningPyBullet(robot=fetch)
    fetch_mp.add_obstacle_body(tableId)
    arm_path = fetch_mp.plan_to_pose_goal([0, 0.6, 0.6], p.getQuaternionFromEuler([0, 0, math.radians(90)]))
    if arm_path:
        print("motion plan found with len {}".format(len(arm_path)))
        # print(arm_path)
        fetch_mp.execute(arm_path)
    time.sleep(1)

    # command robot to push a box
    state_id = p.saveState()
    start_y = 0.6
    print("---------------------Test1")
    for _ in range(2400): # 10 sec
        jointPoses = p.calculateInverseKinematics(fetchId,
                                                FETCH_EE_IDX,
                                                [0, start_y, 0.6],
                                                p.getQuaternionFromEuler([0, 0, math.radians(90)]),
                                                maxNumIterations=100,
                                                residualThreshold=.01)
        # print(jointPoses)
        # print(start_y)

        for i in range(len(fetch_non_fixed_joints)):
            p.setJointMotorControl2(bodyIndex=fetchId,
                                    jointIndex=fetch_non_fixed_joints[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        p.stepSimulation()
        time.sleep(1./ 240)
        if start_y <= 1.0:
            start_y += 0.0005

    # reset, try again with different physics
    print("---------------------Test2")
    p.restoreState(stateId = state_id)
    p.changeDynamics(multibox, -1, lateralFriction=2)
    start_y = 0.6
    for _ in range(4800):  # move with small random actions for 10 seconds
        jointPoses = p.calculateInverseKinematics(fetchId,
                                                FETCH_EE_IDX,
                                                [0, start_y, 0.6],
                                                p.getQuaternionFromEuler([0, 0, math.radians(90)]),
                                                maxNumIterations=100,
                                                residualThreshold=.01)
        # print(jointPoses)
        # print(start_y)

        for i in range(len(fetch_non_fixed_joints)):
            p.setJointMotorControl2(bodyIndex=fetchId,
                                    jointIndex=fetch_non_fixed_joints[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        p.stepSimulation()
        time.sleep(1./ 240)
        if start_y <= 1.0:
            start_y += 0.0005

    p.disconnect()

if __name__ == '__main__':
    main()