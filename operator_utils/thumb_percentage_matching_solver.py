from thumb_percentage_matching_utils import PolynomialRegression2D
import mujoco
import numpy as np
from tqdm import tqdm
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    load = False
    model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    thumb_joint_id = [0, 1, 2, 3]

    if not load:
        thumb_tip = []
        dof1_ratios = []
        dof2_ratios = []

        with mujoco.viewer.launch_passive(model, data) as viewer:
            for i in tqdm(range(10000)):
                thumb_cmc = np.random.uniform(0, 1.31)
                thumb_mcp = np.random.uniform(0, 0.523)
                dof1_ratio = thumb_cmc / 1.31
                dof2_ratio = thumb_mcp / 0.523
                thumb_dip = 0.607 * dof2_ratio
                thumb_ip = 0.431 * dof2_ratio

                data.qpos[thumb_joint_id[0]] = thumb_cmc
                data.qpos[thumb_joint_id[1]] = thumb_mcp
                data.qpos[thumb_joint_id[2]] = thumb_dip
                data.qpos[thumb_joint_id[3]] = thumb_ip
                mujoco.mj_forward(model, data)
                viewer.sync()
                site_id = model.site("right_thumb_tip_sphere").id
                thumb_tip.append(data.site_xpos[site_id].copy())
                dof1_ratios.append(dof1_ratio)
                dof2_ratios.append(dof2_ratio)

        thumb_tip = np.array(thumb_tip)
        dof1_ratios = np.array(dof1_ratios)
        dof2_ratios = np.array(dof2_ratios)
        np.save(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip.npy"), thumb_tip)
        np.save(os.path.join(PROJECT_ROOT, "fingertip_data/dof1_ratios.npy"), dof1_ratios)
        np.save(os.path.join(PROJECT_ROOT, "fingertip_data/dof2_ratios.npy"), dof2_ratios)

    else:
        thumb_tip = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/thumb_tip.npy"))
        dof1_ratios = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/dof1_ratios.npy"))
        dof2_ratios = np.load(os.path.join(PROJECT_ROOT, "fingertip_data/dof2_ratios.npy"))

    best_degree = 3
    best_error = float('inf')
    best_regressor = None

    for degree in [2, 3, 4]:
        try:
            regressor = PolynomialRegression2D(dof1_ratios, dof2_ratios, thumb_tip, degree=degree, finger_name="thumb")
            regressor.fit()
            errors = regressor.evaluate_error()
            if errors['mean_error'] < best_error:
                best_error = errors['mean_error']
                best_degree = degree
                best_regressor = regressor
        except Exception as e:
            print(f"Degree {degree} failed: {e}")

    if best_regressor is None:
        raise RuntimeError("Failed to fit polynomial regression for any degree.")

    regressor = best_regressor
    regressor.save_coeffs(os.path.join(PROJECT_ROOT, "operator_utils/thumb_polynomial_coeffs.json"))
