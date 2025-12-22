import mujoco
import numpy as np
import os
import time

from operator_utils.four_percentage_matching_utils import SigmoidRegression
from operator_utils.thumb_percentage_matching_utils import PolynomialRegression2D
from operator_utils.four_point_mathcing_utils import PolynomialCurveFitting, find_nearest_point
from operator_utils.thumb_point_matching_utils import QuadraticSurfaceFitting, find_nearest_point_on_surface
from hand_sdk import RH56RobotHand
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class InspireHandOperator:
    def __init__(
        self, 
        port="/dev/tty.usbserial-110", 
        hand_id=1,
        forces = [40] * 6,
        speeds = [100] * 6,
    ):
        self.four_fingers = ['index', 'middle', 'ring', 'pinky']
        self.finger_joint_id = {
            'thumb': [0, 1, 2, 3],
            'index': [4, 5],
            'middle': [6, 7],
            'ring': [8, 9],
            'pinky': [10, 11],
        }
        self.finger_joint_range = {
            'thumb': [1.31, 0.523, 0.607, 0.431],
            'index': [1.34, 1.6],
            'middle': [1.34, 1.6],
            'ring': [1.34, 1.6],
            'pinky': [1.34, 1.6],
        }
        self.index_fingertip_space = PolynomialCurveFitting.load_coeffs(finger_name="index", filepath=os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))
        self.middle_fingertip_space = PolynomialCurveFitting.load_coeffs(finger_name="middle", filepath=os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))
        self.ring_fingertip_space = PolynomialCurveFitting.load_coeffs(finger_name="ring", filepath=os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))
        self.pinky_fingertip_space = PolynomialCurveFitting.load_coeffs(finger_name="pinky", filepath=os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))
        self.thumb_fingertip_space = QuadraticSurfaceFitting.load_coeffs(finger_name="thumb", filepath=os.path.join(PROJECT_ROOT, "operator_utils/thumb_surface_coeffs.json"))

        self.index_fingertip_operator = SigmoidRegression.load_coeffs(finger_name="index", filepath=os.path.join(PROJECT_ROOT, "operator_utils/sigmoid_coeffs.json"))
        self.middle_fingertip_operator = SigmoidRegression.load_coeffs(finger_name="middle", filepath=os.path.join(PROJECT_ROOT, "operator_utils/sigmoid_coeffs.json"))
        self.ring_fingertip_operator = SigmoidRegression.load_coeffs(finger_name="ring", filepath=os.path.join(PROJECT_ROOT, "operator_utils/sigmoid_coeffs.json"))
        self.pinky_fingertip_operator = SigmoidRegression.load_coeffs(finger_name="pinky", filepath=os.path.join(PROJECT_ROOT, "operator_utils/sigmoid_coeffs.json"))
        self.thumb_fingertip_operator = PolynomialRegression2D.load_coeffs(finger_name="thumb", filepath=os.path.join(PROJECT_ROOT, "operator_utils/thumb_polynomial_coeffs.json"))

        self.hand_sdk = RH56RobotHand(port=port, baudrate=115200, hand_id=hand_id)
        self.hand_sdk.connect()
        self.hand_sdk.set_angles([100] * 6)
        self.hand_sdk.set_speeds(speeds)
        self.hand_sdk.set_forces(forces)

    def reset(self):
        self.hand_sdk.set_angles([100] * 6)


    def step(self, action: np.ndarray):
        # The Input should be (5, 3) fingertip target positions
        # The sequence of fingers should be : index -> middle -> ring -> pinky -> thumb
        index_target_ft, middle_target_ft, ring_target_ft, pinky_target_ft, thumb_target_ft = action
        
        index_ft, _, _ = find_nearest_point(index_target_ft, self.index_fingertip_space.coeffs)
        middle_ft, _, _ = find_nearest_point(middle_target_ft, self.middle_fingertip_space.coeffs)
        ring_ft, _, _ = find_nearest_point(ring_target_ft, self.ring_fingertip_space.coeffs)
        pinky_ft, _, _ = find_nearest_point(pinky_target_ft, self.pinky_fingertip_space.coeffs)
        thumb_ft, _, _, _ = find_nearest_point_on_surface(thumb_target_ft, self.thumb_fingertip_space.coeffs)

        index_dof1_ratio = self.index_fingertip_operator.predict(index_ft[1])
        middle_dof1_ratio = self.middle_fingertip_operator.predict(middle_ft[1])
        ring_dof1_ratio = self.ring_fingertip_operator.predict(ring_ft[1])
        pinky_dof1_ratio = self.pinky_fingertip_operator.predict(pinky_ft[1])
        thumb_dof1_ratio, thumb_dof2_ratio = self.thumb_fingertip_operator.xyz_to_ratios(thumb_ft[0], thumb_ft[1], thumb_ft[2])

        motor_action = np.array([thumb_dof1_ratio, thumb_dof2_ratio, index_dof1_ratio, middle_dof1_ratio, ring_dof1_ratio, pinky_dof1_ratio])
        step_action = np.array([pinky_dof1_ratio, ring_dof1_ratio, middle_dof1_ratio, index_dof1_ratio, thumb_dof2_ratio, thumb_dof1_ratio])
        step_action = 100 - (step_action * 100) 
        self.hand_sdk.set_angles(step_action)

        return motor_action

    def motor_to_joint(self, motor_action: np.ndarray):
        thumb_cmc = motor_action[0] * self.finger_joint_range['thumb'][0]
        thumb_mcp = motor_action[1] * self.finger_joint_range['thumb'][1]
        thumb_dip = motor_action[1] * self.finger_joint_range['thumb'][2]
        thumb_ip = motor_action[1] * self.finger_joint_range['thumb'][3]
        index_mcp = motor_action[2] * self.finger_joint_range['index'][0]
        index_dip = motor_action[2] * self.finger_joint_range['index'][1]
        middle_mcp = motor_action[3] * self.finger_joint_range['middle'][0]
        middle_dip = motor_action[3] * self.finger_joint_range['middle'][1]
        ring_mcp = motor_action[4] * self.finger_joint_range['ring'][0]
        ring_dip = motor_action[4] * self.finger_joint_range['ring'][1]
        pinky_mcp = motor_action[5] * self.finger_joint_range['pinky'][0]
        pinky_dip = motor_action[5] * self.finger_joint_range['pinky'][1]
        
        joint_action = np.array([thumb_cmc, thumb_mcp, thumb_dip, thumb_ip, index_mcp, index_dip, middle_mcp, middle_dip, ring_mcp, ring_dip, pinky_mcp, pinky_dip])

        return joint_action

   

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    index_target = np.load(f"{PROJECT_ROOT}/fingertip_data/two_dof_fingertips_index.npy")
    middle_target = np.load(f"{PROJECT_ROOT}/fingertip_data/two_dof_fingertips_middle.npy")
    ring_target = np.load(f"{PROJECT_ROOT}/fingertip_data/two_dof_fingertips_ring.npy")
    pinky_target = np.load(f"{PROJECT_ROOT}/fingertip_data/two_dof_fingertips_pinky.npy")
    thumb_target = np.load(f"{PROJECT_ROOT}/fingertip_data/thumb_tip_random.npy")
    hand_operator = InspireHandOperator()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        frame_count = 0
        
        for i in range(1000):
            frame_start = time.time()
            
            # Add Gaussian noise to target positions (simulating sensor noise)
            noise_std = 0.005  # Standard deviation of noise (5mm)
            index_target_ft = index_target[i] + np.random.normal(0, noise_std, 3)
            middle_target_ft = middle_target[i] + np.random.normal(0, noise_std, 3)
            ring_target_ft = ring_target[i] + np.random.normal(0, noise_std, 3)
            pinky_target_ft = pinky_target[i] + np.random.normal(0, noise_std, 3)
            thumb_target_ft = thumb_target[i] + np.random.normal(0, noise_std, 3)
            motor_action = hand_operator.step(np.array([index_target_ft, middle_target_ft, ring_target_ft, pinky_target_ft, thumb_target_ft]))
            print(f"Motor action: {motor_action}")
            model.site("index_tip_target").pos = index_target_ft
            model.site("middle_tip_target").pos = middle_target_ft
            model.site("ring_tip_target").pos = ring_target_ft
            model.site("pinky_tip_target").pos = pinky_target_ft
            model.site("thumb_tip_target").pos = thumb_target_ft

            data.qpos[hand_operator.finger_joint_id['thumb'][0]] = motor_action[0] * hand_operator.finger_joint_range['thumb'][0]
            data.qpos[hand_operator.finger_joint_id['thumb'][1]] = motor_action[1] * hand_operator.finger_joint_range['thumb'][1]
            data.qpos[hand_operator.finger_joint_id['thumb'][2]] = motor_action[1] * hand_operator.finger_joint_range['thumb'][2]
            data.qpos[hand_operator.finger_joint_id['thumb'][3]] = motor_action[1] * hand_operator.finger_joint_range['thumb'][3]
            data.qpos[hand_operator.finger_joint_id['index'][0]] = motor_action[2] * hand_operator.finger_joint_range['index'][0]
            data.qpos[hand_operator.finger_joint_id['index'][1]] = motor_action[2] * hand_operator.finger_joint_range['index'][1]
            data.qpos[hand_operator.finger_joint_id['middle'][0]] = motor_action[3] * hand_operator.finger_joint_range['middle'][0]
            data.qpos[hand_operator.finger_joint_id['middle'][1]] = motor_action[3] * hand_operator.finger_joint_range['middle'][1]
            data.qpos[hand_operator.finger_joint_id['ring'][0]] = motor_action[4] * hand_operator.finger_joint_range['ring'][0]
            data.qpos[hand_operator.finger_joint_id['ring'][1]] = motor_action[4] * hand_operator.finger_joint_range['ring'][1]
            data.qpos[hand_operator.finger_joint_id['pinky'][0]] = motor_action[5] * hand_operator.finger_joint_range['pinky'][0]
            data.qpos[hand_operator.finger_joint_id['pinky'][1]] = motor_action[5] * hand_operator.finger_joint_range['pinky'][1]
            mujoco.mj_forward(model, data)
            viewer.sync()
            
            frame_count += 1
            frame_time = time.time() - frame_start
            
            # Calculate and print FPS every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                avg_frame_time = elapsed_time / frame_count
                print(f"\n[FPS Stats] Frame {frame_count}: FPS = {fps:.2f}, Avg frame time = {avg_frame_time*1000:.2f}ms\n")
            
            time.sleep(5)
        
        # Final FPS calculation
        total_time = time.time() - start_time
        final_fps = frame_count / total_time
        avg_frame_time = total_time / frame_count
        print(f"\n[Final FPS Stats] Total frames: {frame_count}, Total time: {total_time:.2f}s, FPS: {final_fps:.2f}, Avg frame time: {avg_frame_time*1000:.2f}ms\n")