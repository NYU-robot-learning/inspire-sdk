import argparse
import pickle
import time
import numpy as np
import zmq
import mujoco
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation
from hand_operator import InspireHandOperator

FINGERTIP_INDICES = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinky': 4}
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="100.94.225.27")
    parser.add_argument("--port", type=int, default=10012)
    parser.add_argument("--xml-path", default=None)
    parser.add_argument("--enable-hand-control", action="store_true", default=False)
    parser.add_argument("--hand-port", default=None)
    return parser.parse_args()


class AriaMuJoCoVisualizer:
    def __init__(self, host, port, xml_path=None, hand_port=None, enable_hand_control=False):
        self.enable_hand_control = enable_hand_control
        xml_path = xml_path or os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.fingertip_site_ids = {k: self.model.site(f"{k}_tip_target").id for k in ['thumb', 'index', 'middle', 'ring', 'pinky']}
        self.hand_operator = None
        if self.enable_hand_control:
            try:
                self.hand_operator = InspireHandOperator(port=hand_port or "/dev/tty.usbserial-110")
            except:
                self.enable_hand_control = False
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"hand_landmarks ")
        self.socket.connect(f"tcp://{host}:{port}")
        self.landmarks_count = 0
        self.last_stats_time = time.time()

    def quaternion_to_rotation_matrix(self, quaternion):
        quat = np.array(quaternion).flatten()
        if quat.size != 4:
            raise ValueError(f"Invalid quaternion length: {quat.size}")
        try:
            return Rotation.from_quat(quat).as_matrix()
        except:
            return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()

    def wrist_axes_to_rotation_matrix(self, wrist_axes):
        return np.column_stack([np.array(wrist_axes[k]) for k in ['x_axis', 'y_axis', 'z_axis']])

    def transform_to_wrist_frame(self, point_world, wrist_translation_world, wrist_axes=None, wrist_quaternion=None):
        point_world = np.array(point_world)
        wrist_translation_world = np.array(wrist_translation_world)
        R_world_wrist = self.wrist_axes_to_rotation_matrix(wrist_axes) if wrist_axes else self.quaternion_to_rotation_matrix(wrist_quaternion)
        return R_world_wrist.T @ (point_world - wrist_translation_world)

    def extract_fingertips(self, landmarks_world, wrist_translation_world=None, wrist_axes=None, wrist_quaternion=None):
        fingertips = {}
        for finger_name, landmark_idx in FINGERTIP_INDICES.items():
            if landmark_idx < len(landmarks_world):
                landmark = landmarks_world[landmark_idx]
                if isinstance(landmark, (list, np.ndarray)) and len(landmark) >= 3:
                    fingertip_world = np.array(landmark[:3])
                    if wrist_translation_world is not None and (wrist_axes is not None or wrist_quaternion is not None):
                        try:
                            fingertips[finger_name] = self.transform_to_wrist_frame(fingertip_world, wrist_translation_world, wrist_axes, wrist_quaternion)
                        except:
                            fingertips[finger_name] = fingertip_world
                    else:
                        fingertips[finger_name] = fingertip_world
                else:
                    fingertips[finger_name] = None
            else:
                fingertips[finger_name] = None
        return fingertips

    def rotate_around_z_axis(self, point, angle_degrees=-90):
        angle_rad = np.deg2rad(angle_degrees)
        R_z = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                        [0, 0, 1]])
        return R_z @ np.array(point)

    def update_mujoco_targets(self, fingertips, y_offset=0.0):
        site_name_map = {k: f"{k}_tip_target" for k in ['thumb', 'index', 'middle', 'ring', 'pinky']}
        for finger_name, position in fingertips.items():
            if position is not None and finger_name in site_name_map:
                rotated = self.rotate_around_z_axis(position, -90)
                rotated[1] -= y_offset
                self.model.site(site_name_map[finger_name]).pos[:] = rotated

    def process_landmarks_message(self, data):
        self.landmarks_count += 1
        if not data.get("right_hand") or 'landmarks_device' not in data["right_hand"]:
            return False
        right_hand = data["right_hand"]
        landmarks_world = right_hand['landmarks_device']
        wrist_translation_world = right_hand.get('wrist_translation')
        wrist_axes = right_hand.get('wrist_axes')
        wrist_quaternion = right_hand.get('wrist_quaternion')
        fingertips = self.extract_fingertips(landmarks_world, wrist_translation_world, wrist_axes, wrist_quaternion)
        self.update_mujoco_targets(fingertips, y_offset=0.02)
        if self.enable_hand_control and self.hand_operator is not None:
            try:
                fingertip_targets = []
                for finger_name in ['index', 'middle', 'ring', 'pinky', 'thumb']:
                    if fingertips.get(finger_name) is not None:
                        rotated = self.rotate_around_z_axis(fingertips[finger_name], -90)
                        rotated[1] -= 0.02
                        fingertip_targets.append(rotated)
                    else:
                        fingertip_targets.append(np.array([0.0, 0.0, 0.0]))
                self.hand_operator.step(np.array(fingertip_targets))
            except:
                pass
        return True

    def print_stats(self):
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        if elapsed >= 5.0:
            landmarks_fps = self.landmarks_count / elapsed if elapsed > 0 else 0
            print(f"\n[Stats] Landmarks processed: {self.landmarks_count} ({landmarks_fps:.2f} fps)")
            self.landmarks_count = 0
            self.last_stats_time = current_time

    def run(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                while True:
                    if self.socket.poll(timeout=10):
                        try:
                            raw_message = self.socket.recv(zmq.NOBLOCK)
                            if raw_message.startswith(b"hand_landmarks "):
                                data = pickle.loads(raw_message[len(b"hand_landmarks "):])
                                if self.process_landmarks_message(data):
                                    mujoco.mj_forward(self.model, self.data)
                                    viewer.sync()
                                self.print_stats()
                        except (zmq.Again, Exception):
                            continue
                    else:
                        viewer.sync()
                        time.sleep(0.001)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        self.socket.close()
        self.context.term()


def main():
    args = parse_args()
    visualizer = AriaMuJoCoVisualizer(args.host, args.port, args.xml_path, args.hand_port, args.enable_hand_control)
    visualizer.run()


if __name__ == "__main__":
    main()
