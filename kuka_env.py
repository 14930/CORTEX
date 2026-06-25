import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import math

class KukaCableGymEnv:
    def __init__(self, use_gui=False, target_pos=[0.5, 0.2, 0.01], num_segments=12):
        """
        Custom Environment tailored for Member 3 (RL) and Member 4 (Vision).
        """
        self.target_pos = target_pos
        self.num_segments = num_segments
        self.radius = 0.02
        self.stiffness = 80
        self.end_effector_link = 6
        
        # Connect to PyBullet
        mode = p.GUI if use_gui else p.DIRECT
        p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load assets
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
        
        self._build_cable()
        self._build_target()
        self._setup_camera_matrices()

    def _build_cable(self):
        ee_state = p.getLinkState(self.robot, self.end_effector_link)
        ee_pos = ee_state[0]
        segment_gap = self.radius * 2.5 
        
        self.cable_segments = []
        for i in range(self.num_segments):
            shape = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=[1, 0.4, 0, 1])
            body = p.createMultiBody(
                baseMass=0.008, baseCollisionShapeIndex=shape, baseVisualShapeIndex=visual,
                basePosition=[ee_pos[0], ee_pos[1], ee_pos[2] - i * segment_gap]
            )
            p.changeDynamics(body, -1, linearDamping=2.0, angularDamping=2.0)
            self.cable_segments.append(body)

        for i in range(len(self.cable_segments) - 1):
            c = p.createConstraint(
                parentBodyUniqueId=self.cable_segments[i], parentLinkIndex=-1,
                childBodyUniqueId=self.cable_segments[i + 1], childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, -self.radius], childFramePosition=[0, 0, self.radius]
            )
            p.changeConstraint(c, maxForce=self.stiffness)

        attach = p.createConstraint(
            parentBodyUniqueId=self.robot, parentLinkIndex=self.end_effector_link,
            childBodyUniqueId=self.cable_segments[0], childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0], childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(attach, maxForce=300)

    def _build_target(self):
        t_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.10, length=0.02, rgbaColor=[1, 0, 0, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=t_visual, basePosition=self.target_pos)

    def _setup_camera_matrices(self):
        self.view_matrix = p.computeViewMatrix(cameraEyePosition=[1.4, 1.2, 0.9], cameraTargetPosition=[0.2, 0.0, 0.5], cameraUpVector=[0, 0, 1])
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=65, aspect=640/480, nearVal=0.1, farVal=10)

    # ─── ADDITION FOR MEMBER 4 (VISION) ───
    def get_full_camera_data(self):
        """
        Returns full synthetic camera data including RGB, Depth, and Segmentation.
        Essential for Member 4 to simulate Intel RealSense point clouds.
        """
        width, height, rgb, depth, seg = p.getCameraImage(
            640, 480, self.view_matrix, self.proj_matrix
        )
        # Convert to numpy arrays for easier OpenCV/Open3D processing
        rgb_array = np.array(rgb)[:, :, :3].astype(np.uint8)
        depth_array = np.array(depth)
        seg_array = np.array(seg)
        return rgb_array, depth_array, seg_array

    # ─── ADDITION FOR MEMBER 3 (RL) ───
    def get_observation(self):
        """
        Constructs the observation vector for RL training.
        Includes joint states, cable positions, and target location.
        """
        # 1. Get joint positions of the 7 robot joints
        joint_states = p.getJointStates(self.robot, range(7))
        joint_positions = [state[0] for state in joint_states]
        
        # 2. Get 3D positions of all cable segments
        cable_positions = []
        for seg in self.cable_segments:
            pos, _ = p.getBasePositionAndOrientation(seg)
            cable_positions.extend(pos) # Flattens into [x1,y1,z1, x2,y2,z2...]
            
        # Combine everything into a single observation array
        obs = np.array(joint_positions + cable_positions + self.target_pos, dtype=np.float32)
        return obs

    def step(self, action_joint_angles):
        """
        Applies action from RL agent, steps physics, and returns (obs, reward, done).
        """
        # Apply actions to robot joints
        for j in range(7):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, 
                                    targetPosition=action_joint_angles[j], force=500)
        p.stepSimulation()
        
        # Calculate Reward (Distance between cable tip and target)
        cable_tip_pos, _ = p.getBasePositionAndOrientation(self.cable_segments[-1])
        distance = np.linalg.norm(np.array(cable_tip_pos) - np.array(self.target_pos))
        
        # Basic reward structure (Member 3 will refine this with penalties)
        reward = -distance 
        
        # Define termination condition
        done = True if distance < 0.05 else False
        
        return self.get_observation(), reward, done, {}

    def reset(self):
        """Resets the environment for a new RL episode."""
        # Reset joints to home position
        for j in range(7):
            p.resetJointState(self.robot, j, targetValue=0.0)
        p.stepSimulation()
        return self.get_observation()

    def close(self):
        p.disconnect()