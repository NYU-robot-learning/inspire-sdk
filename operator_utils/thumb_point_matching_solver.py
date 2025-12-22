from thumb_point_matching_utils import QuadraticSurfaceFitting, find_nearest_point_on_surface
import numpy as np
import os
import mujoco
import viser
from viser.extras import ViserUrdf
import yourdfpy
import time
from scipy.spatial.distance import cdist
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def visualize_result(surface_fitting: QuadraticSurfaceFitting = None, points: np.ndarray = None, 
                     coeff_file: str = None, finger_name: str = "thumb"):
    if surface_fitting is None:
        if coeff_file is None:
            coeff_file = os.path.join(PROJECT_ROOT, "operator_utils/thumb_surface_coeffs.json")
        surface_fitting = QuadraticSurfaceFitting.load_coeffs(finger_name, coeff_file)
        print(f"Loaded coefficients from: {coeff_file}")
    
    if points is None:
        points_path = os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip.npy")
        if os.path.exists(points_path):
            points = np.load(points_path)
            print(f"Loaded points from: {points_path}")
        else:
            raise ValueError(f"Points file not found: {points_path}. Please provide points array.")
    
    server = viser.ViserServer()
    urdf = yourdfpy.URDF.load(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.urdf"))
    viser_urdf = ViserUrdf(server, urdf, root_node_name="/robot_hand")
    
    num_joints = len(urdf.actuated_joint_names)
    joint_angles = np.zeros(num_joints)
    viser_urdf.update_cfg(joint_angles)
    
    server.scene.add_point_cloud(
        name="/points_thumb",
        points=points,
        colors=(255, 0, 0),
        point_size=0.003,
        point_shape='circle'
    )
    
    u_min, u_max = surface_fitting.u_bounds
    v_min, v_max = surface_fitting.v_bounds
    u_grid = np.linspace(u_min, u_max, 50)
    v_grid = np.linspace(v_min, v_max, 50)
    surface_points = surface_fitting.surface_func(u_grid, v_grid)
    
    data_min = points.min(axis=0)
    data_max = points.max(axis=0)
    margin = 0.005
    surface_flat = surface_points.reshape(-1, 3)
    
    within_bounds = np.all(
        (surface_flat >= data_min - margin) & (surface_flat <= data_max + margin),
        axis=1
    )
    
    distances = cdist(surface_flat, points)
    min_distances = distances.min(axis=1)
    original_distances = cdist(points, points)
    original_distances = original_distances[original_distances > 0]
    distance_threshold = np.percentile(original_distances, 90) * 2
    
    valid_mask = within_bounds & (min_distances <= distance_threshold)
    surface_filtered = surface_flat[valid_mask]
    
    server.scene.add_point_cloud(
        name="/surface_thumb",
        points=surface_filtered,
        colors=(0, 255, 255),
        point_size=0.002,
        point_shape='circle'
    )
    
    test_points = np.array([
        points.mean(axis=0),
        points.mean(axis=0) + np.array([0.02, 0.0, 0.0]),
        points.mean(axis=0) + np.array([0.0, 0.02, 0.0]),
        points.mean(axis=0) + np.array([0.0, 0.0, 0.02]),
    ])
    
    query_colors = [(255, 255, 0), (255, 165, 0), (255, 192, 203), (128, 0, 128)]
    line_segments = []
    
    for i, test_point in enumerate(test_points):
        nearest, u, v, dist = find_nearest_point_on_surface(
            test_point, surface_fitting.coeffs, surface_fitting.u_bounds, 
            surface_fitting.v_bounds, data_points=points
        )
        
        server.scene.add_point_cloud(
            name=f"/query_point_{i}",
            points=test_point.reshape(1, -1),
            colors=query_colors[i],
            point_size=0.005,
            point_shape='circle'
        )
        
        server.scene.add_point_cloud(
            name=f"/nearest_point_{i}",
            points=nearest.reshape(1, -1),
            colors=query_colors[i],
            point_size=0.005,
            point_shape='circle'
        )
        
        line_segments.append([test_point, nearest])
    
    if line_segments:
        segments_array = np.array(line_segments)
        server.scene.add_line_segments(
            name="/query_lines",
            points=segments_array,
            colors=(255, 255, 0),
            line_width=2.0
        )
    
    print("Visualization started. Press Ctrl+C to exit.")
    while True:
        time.sleep(0.01)

if __name__ == "__main__":
    load = False
    model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    thumb_joint_id = [0, 1, 2, 3]

    if not load:
        thumb_tip = []
        thumb_tip_random = []

        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in tqdm(range(10000)):
                thumb_cmc = np.random.uniform(0, 1.31)
                thumb_mcp = np.random.uniform(0, 0.523)
                ratio = thumb_mcp / 0.523
                thumb_dip = 0.607 * ratio
                thumb_ip = 0.431 * ratio

                data.qpos[thumb_joint_id[0]] = thumb_cmc
                data.qpos[thumb_joint_id[1]] = thumb_mcp
                data.qpos[thumb_joint_id[2]] = thumb_dip
                data.qpos[thumb_joint_id[3]] = thumb_ip
                mujoco.mj_forward(model, data)
                site_id = model.site("right_thumb_tip_sphere").id
                thumb_tip.append(data.site_xpos[site_id].copy())
                viewer.sync()
            
            for i in tqdm(range(10000)):
                thumb_cmc = np.random.uniform(0, 1.31)
                thumb_mcp = np.random.uniform(0, 0.523)
                thumb_dip = np.random.uniform(0, 0.607)
                thumb_ip = np.random.uniform(0, 0.431)
                data.qpos[thumb_joint_id[0]] = thumb_cmc
                data.qpos[thumb_joint_id[1]] = thumb_mcp
                data.qpos[thumb_joint_id[2]] = thumb_dip
                data.qpos[thumb_joint_id[3]] = thumb_ip
                mujoco.mj_forward(model, data)
                site_id = model.site("right_thumb_tip_sphere").id
                thumb_tip_random.append(data.site_xpos[site_id].copy())
                viewer.sync()


        thumb_tip = np.array(thumb_tip)
        thumb_tip_random = np.array(thumb_tip_random)
        np.save(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip.npy"), thumb_tip)
        np.save(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip_random.npy"), thumb_tip_random)
    
    else:
        thumb_tip = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip.npy"))
        thumb_tip_random = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip_random.npy"))

    surface_fitting = QuadraticSurfaceFitting(thumb_tip, num_points_u=50, num_points_v=50, finger_name="thumb")
    surface_fitting.fit()
    surface_fitting.save_coeffs(os.path.join(PROJECT_ROOT, "operator_utils/thumb_surface_coeffs.json"))
    
    visualize_result(surface_fitting, thumb_tip)
