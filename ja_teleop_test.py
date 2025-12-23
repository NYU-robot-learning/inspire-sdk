import argparse
import pickle
import time
import numpy as np
import zmq
from scipy.spatial.transform import Rotation
from hand_sdk import RH56RobotHand

JOINT_LIMITS = {
    'thumb': {'CMC': 1.31, 'MCP': 0.523, 'DIP': 0.607, 'IP': 0.431},
    'index': {'MCP': 1.34, 'DIP': 1.6}, 'middle': {'MCP': 1.34, 'DIP': 1.6},
    'ring': {'MCP': 1.34, 'DIP': 1.6}, 'pinky': {'MCP': 1.34, 'DIP': 1.6},
}
LANDMARK_INDICES = {
    'wrist': 5, 'thumb': {'tip': 0, 'intermediate': 6, 'distal': 7},
    'index': {'tip': 1, 'proximal': 8, 'intermediate': 9, 'distal': 10},
    'middle': {'tip': 2, 'proximal': 11, 'intermediate': 12, 'distal': 13},
    'ring': {'tip': 3, 'proximal': 14, 'intermediate': 15, 'distal': 16},
    'pinky': {'tip': 4, 'proximal': 17, 'intermediate': 18, 'distal': 19},
}


def calculate_joint_angle(v1, v2):
    v1_norm, v2_norm = v1 / (np.linalg.norm(v1) + 1e-8), v2 / (np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))


def calculate_finger_joint_angles(landmarks, finger_name, wrist_pos):
    angles = {}
    if len(landmarks) < 21:
        return angles
    wrist_idx = LANDMARK_INDICES['wrist']
    wrist_pos = np.array(landmarks[wrist_idx][:3]) if wrist_idx < len(landmarks) else np.array([0, 0, 0])
    finger_landmarks = LANDMARK_INDICES.get(finger_name)
    if finger_landmarks is None:
        return angles
    
    if finger_name == 'thumb':
        if all(k in finger_landmarks for k in ['intermediate', 'distal', 'tip']):
            i, d, t = [np.array(landmarks[finger_landmarks[k]][:3]) for k in ['intermediate', 'distal', 'tip']]
            angles['CMC'] = np.clip(np.pi - calculate_joint_angle(wrist_pos - i, d - i), 0, JOINT_LIMITS['thumb']['CMC'])
            angles['MCP'] = np.clip(np.pi - calculate_joint_angle(i - d, t - d), 0, JOINT_LIMITS['thumb']['MCP'])
            angles['IP'] = angles['DIP'] = angles['MCP']
    else:
        if all(k in finger_landmarks for k in ['proximal', 'intermediate', 'distal', 'tip']):
            p, i, d, t = [np.array(landmarks[finger_landmarks[k]][:3]) for k in ['proximal', 'intermediate', 'distal', 'tip']]
            angles['MCP'] = np.clip(np.pi - calculate_joint_angle(p - i, d - i), 0, JOINT_LIMITS[finger_name]['MCP'])
            angles['DIP'] = np.clip(np.pi - calculate_joint_angle(i - d, t - d), 0, JOINT_LIMITS[finger_name]['DIP'])
    return angles


def transform_to_wrist_frame(point, wrist_trans, wrist_axes=None, wrist_quat=None):
    point = np.array(point)
    if wrist_axes is not None:
        R = np.column_stack([np.array(wrist_axes[k]) for k in ['x_axis', 'y_axis', 'z_axis']])
    elif wrist_quat is not None:
        quat = np.array(wrist_quat).flatten()
        try:
            R = Rotation.from_quat(quat).as_matrix()
        except:
            R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    else:
        return point
    return R.T @ (point - np.array(wrist_trans))


def get_joint_angles(data):
    if not data.get("right_hand") or 'landmarks_device' not in data["right_hand"]:
        return None
    rh = data["right_hand"]
    landmarks_world = rh['landmarks_device']
    wrist_trans = rh.get('wrist_translation')
    wrist_axes = rh.get('wrist_axes')
    wrist_quat = rh.get('wrist_quaternion')
    
    landmarks_wrist = []
    for lm in landmarks_world:
        if isinstance(lm, (list, np.ndarray)) and len(lm) >= 3:
            lm_world = np.array(lm[:3])
            if wrist_trans is not None:
                try:
                    landmarks_wrist.append(transform_to_wrist_frame(lm_world, wrist_trans, wrist_axes, wrist_quat))
                except:
                    landmarks_wrist.append(lm_world)
            else:
                landmarks_wrist.append(lm_world)
        else:
            landmarks_wrist.append(np.array([0, 0, 0]))
    
    if len(landmarks_wrist) < 21:
        return None
    
    return {name: calculate_finger_joint_angles(landmarks_wrist, name, None) 
            for name in ['thumb', 'index', 'middle', 'ring', 'pinky']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="100.94.225.27")
    parser.add_argument("--port", type=int, default=10012)
    args = parser.parse_args()
    hand = RH56RobotHand(port='/dev/tty.usbserial-110', baudrate=115200, hand_id=1)
    hand.connect()
    hand.set_speeds([100] * 6)
    hand.set_forces([50] * 6)
    hand.set_angles([100] * 6)

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"hand_landmarks ")
    socket.connect(f"tcp://{args.host}:{args.port}")
    print(f"Connected to tcp://{args.host}:{args.port}")
    
    landmarks_count = 0
    last_stats_time = time.time()
    
    try:
        while True:
            if socket.poll(timeout=100):
                raw_msg = socket.recv(zmq.NOBLOCK)
                if raw_msg.startswith(b"hand_landmarks "):
                    data = pickle.loads(raw_msg[len(b"hand_landmarks "):])
                    finger_angles = get_joint_angles(data)

                    if finger_angles is not None:
                        landmarks_count += 1
                        
                        def get_ratio(finger, joints):
                            if finger not in finger_angles or not all(k in finger_angles[finger] for k in joints):
                                return 0.0
                            return np.mean([finger_angles[finger][j] / JOINT_LIMITS[finger][j] for j in joints])
                        
                        ratios = {
                            'index': get_ratio('index', ['MCP', 'DIP']),
                            'middle': get_ratio('middle', ['MCP', 'DIP']),
                            'ring': get_ratio('ring', ['MCP', 'DIP']),
                            'pinky': get_ratio('pinky', ['MCP', 'DIP']),
                            'thumb_dof1': finger_angles['thumb'].get('CMC', 0) / JOINT_LIMITS['thumb']['CMC'] if 'thumb' in finger_angles and 'CMC' in finger_angles['thumb'] else 0.0,
                            'thumb_dof2': get_ratio('thumb', ['MCP', 'DIP', 'IP']),
                        }
                        
                        ratios = {k: 100 - (v * 100) for k, v in ratios.items()}
                        hand.set_angles([ratios['pinky'], ratios['ring'], ratios['middle'], ratios['index'], ratios['thumb_dof2'], ratios['thumb_dof1']])
                        
                        current_time = time.time()
                        elapsed = current_time - last_stats_time
                        if elapsed >= 5.0:
                            print(f"\n[Stats] Landmarks processed: {landmarks_count} ({landmarks_count / elapsed:.2f} fps)")
                            landmarks_count = 0
                            last_stats_time = current_time
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
