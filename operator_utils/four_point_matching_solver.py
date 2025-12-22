from four_point_mathcing_utils import PolynomialCurveFitting
import numpy as np
import mujoco
import time
import yourdfpy
import viser
from viser.extras import ViserUrdf
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os

# Get the project root directory (parent of operator_utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def configuration_space_vis():
    server = viser.ViserServer()
    urdf = yourdfpy.URDF.load(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.urdf"))
    viser_urdf = ViserUrdf(
        server,  
        urdf,
        root_node_name="/robot_hand"
    )

    num_joints = len(urdf.actuated_joint_names)
    joint_angles = np.zeros(num_joints)
    viser_urdf.update_cfg(joint_angles)

    index_one_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/one_dof_fingertips_index.npy"))
    index_two_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/two_dof_fingertips_index.npy"))

    middle_one_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/one_dof_fingertips_middle.npy"))
    middle_two_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/two_dof_fingertips_middle.npy"))

    ring_one_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/one_dof_fingertips_ring.npy"))
    ring_two_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/two_dof_fingertips_ring.npy"))

    pinky_one_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/one_dof_fingertips_pinky.npy"))
    pinky_two_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/two_dof_fingertips_pinky.npy"))

    server.scene.add_point_cloud(
        name="/index_one_dof_fingertips",
        points=index_one_dof_fingertips,
        colors=(0.0, 0.0, 1.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/index_two_dof_fingertips",
        points=index_two_dof_fingertips,
        colors=(0.0, 1.0, 0.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/middle_one_dof_fingertips",
        points=middle_one_dof_fingertips,
        colors=(0.0, 0.0, 1.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/middle_two_dof_fingertips",
        points=middle_two_dof_fingertips,
        colors=(0.0, 1.0, 0.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/ring_one_dof_fingertips",
        points=ring_one_dof_fingertips,
        colors=(0.0, 0.0, 1.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/ring_two_dof_fingertips",
        points=ring_two_dof_fingertips,
        colors=(0.0, 1.0, 0.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/pinky_one_dof_fingertips",
        points=pinky_one_dof_fingertips,
        colors=(0.0, 0.0, 1.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    server.scene.add_point_cloud(
        name="/pinky_two_dof_fingertips",
        points=pinky_two_dof_fingertips,
        colors=(0.0, 1.0, 0.0), 
        point_size=0.001,
        point_shape='circle'
    )
    
    while True:
        time.sleep(0.01)


def visualize_all_curves(coeff_file: str = None):
    if coeff_file is None:
        coeff_file = os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json")
    server = viser.ViserServer()
    urdf = yourdfpy.URDF.load(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.urdf"))
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_hand")
    
    num_joints = len(urdf.actuated_joint_names)
    joint_angles = np.zeros(num_joints)
    viser_urdf.update_cfg(joint_angles)
    
    colors = {
        'index': (255, 0, 0),
        'middle': (0, 0, 255),
        'ring': (0, 255, 0),
        'pinky': (255, 165, 0)
    }
    
    u_curve = np.linspace(0, 1, 1000)
    
    with open(coeff_file, 'r') as f:
        all_data = json.load(f)
    
    for finger_name, color in colors.items():
        if finger_name in all_data:
            curve = PolynomialCurveFitting.load_coeffs(finger_name, coeff_file)
            curve_points = curve.curve_func(u_curve)
            
            segments = np.stack([curve_points[:-1], curve_points[1:]], axis=1)
            
            server.scene.add_line_segments(
                name=f"/curve_{finger_name}",
                points=segments,
                colors=color,
                line_width=3.0
            )
    
    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    four_fingertips = ['index', 'middle', 'ring', 'pinky']
    finger_joint_id = {
        'index': [4, 5],
        'middle': [6, 7],
        'ring': [8, 9],
        'pinky': [10, 11],
    }
    load_finger_data = False
    fit_finger_data = True

    if not load_finger_data:
        model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        for finger in four_fingertips:
            one_dof_fingertips = []
            two_dof_fingertips = []

            # Generate the one and two dof fingertips for {finger}
            for i in tqdm(range(10000)):
                finger_mcp = np.random.uniform(0, 1.34)
                finger_joint_ratio = finger_mcp / 1.34
                finger_dip = 1.6 * finger_joint_ratio
                data.qpos[finger_joint_id[finger][0]] = finger_mcp
                data.qpos[finger_joint_id[finger][1]] = finger_dip
                mujoco.mj_forward(model, data)
                site_id = model.site(f"right_{finger}_tip_sphere").id
                one_dof_fingertips.append(data.site_xpos[site_id].copy())
            
            for i in tqdm(range(10000)):
                finger_mcp = np.random.uniform(0, 1.34)
                finger_dip = np.random.uniform(0, 1.6)
                data.qpos[finger_joint_id[finger][0]] = finger_mcp
                data.qpos[finger_joint_id[finger][1]] = finger_dip
                mujoco.mj_forward(model, data)
                site_id = model.site(f"right_{finger}_tip_sphere").id
                two_dof_fingertips.append(data.site_xpos[site_id].copy())

            one_dof_fingertips = np.array(one_dof_fingertips)
            two_dof_fingertips = np.array(two_dof_fingertips)
            np.save(os.path.join(PROJECT_ROOT, f"fingertip_data/one_dof_fingertips_{finger}.npy"), one_dof_fingertips)
            np.save(os.path.join(PROJECT_ROOT, f"fingertip_data/two_dof_fingertips_{finger}.npy"), two_dof_fingertips)

        configuration_space_vis()

    # Find the fingertip curve for each finger
    if fit_finger_data:
        for finger in four_fingertips:
            one_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, f"fingertip_data/one_dof_fingertips_{finger}.npy"))
            two_dof_fingertips = np.load(os.path.join(PROJECT_ROOT, f"fingertip_data/two_dof_fingertips_{finger}.npy"))

            one_dof_curve_fitting = PolynomialCurveFitting(one_dof_fingertips, finger_name=finger)
            one_dof_curve_fitting.fit()
            one_dof_curve_fitting.save_coeffs(os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))

        visualize_all_curves(os.path.join(PROJECT_ROOT, "operator_utils/finger_coeffs.json"))




        


    

            
