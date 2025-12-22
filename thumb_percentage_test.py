from operator_utils.thumb_percentage_matching_utils import PolynomialRegression2D
import mujoco
import numpy as np
import os
import time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.join(PROJECT_ROOT, "inspire_sdk")


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(os.path.join(PROJECT_ROOT, "rh56_urdf/non-wrist-inspire-right.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    thumb_joint_id = [0, 1, 2, 3]
    sample_data = np.load("/Users/yiboyan/Desktop/inspire_sdk/fingertip_data/thumb_tip.npy")
    coeff_file = os.path.join(PROJECT_ROOT, "operator_utils/thumb_polynomial_coeffs.json")
    poly_model = PolynomialRegression2D.load_coeffs(finger_name="thumb", filepath=coeff_file)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for sample in sample_data:
            dof1, dof2 = poly_model.xyz_to_ratios(sample[0], sample[1], sample[2])
            model.site("target_point").pos = sample
            cmc_joint = dof1 * 1.31
            mcp_joint = dof2 * 0.523
            dip_joint = 0.607 * dof2
            ip_joint = dof2 * 0.431
            data.qpos[thumb_joint_id[0]] = cmc_joint
            data.qpos[thumb_joint_id[1]] = mcp_joint
            data.qpos[thumb_joint_id[2]] = dip_joint
            data.qpos[thumb_joint_id[3]] = ip_joint
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.5)
