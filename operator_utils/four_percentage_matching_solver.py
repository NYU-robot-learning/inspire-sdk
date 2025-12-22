from four_percentage_matching_utils import SigmoidRegression
import mujoco
import numpy as np
from tqdm import tqdm
import os

# Get the project root directory (parent of operator_utils)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    four_fingertips = ['index', 'middle', 'ring', 'pinky']
    finger_joint_id = {
        'index': [4, 5],
        'middle': [6, 7],
        'ring': [8, 9],
        'pinky': [10, 11],
    }
    model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for finger in four_fingertips:
        # Collect percentage and y-value data for the finger
        percentage_data = []
        y_value = []

        for i in tqdm(np.arange(0, 100, 0.01)):
            finger_mcp = i / 100 * 1.34
            finger_dip = i / 100 * 1.6
            data.qpos[finger_joint_id[finger][0]] = finger_mcp
            data.qpos[finger_joint_id[finger][1]] = finger_dip
            mujoco.mj_forward(model, data)
            site_id = model.site(f"right_{finger}_tip_sphere").id
            y_value.append(data.site_xpos[site_id][1])
            percentage_data.append(i)
        
        points = np.column_stack([percentage_data, y_value])
        sigmoid_regression = SigmoidRegression(points, num_points=10000, finger_name=finger)
        sigmoid_regression.fit()
        sigmoid_regression.save_coeffs(os.path.join(PROJECT_ROOT, "operator_utils/sigmoid_coeffs.json"))
        sigmoid_regression.print_equation()
        sigmoid_regression.plot(save_path=os.path.join(PROJECT_ROOT, f"img/sigmoid_fit_{finger}.png"), show=False)

