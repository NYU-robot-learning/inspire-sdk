import viser
from viser.extras import ViserUrdf
import yourdfpy
import numpy as np
import time
import json
import os


class PolynomialCurveFitting:
    def __init__(self, points: np.ndarray, degree: int = 3, num_points: int = 5000, sort_by_pca: bool = True, finger_name: str = None):
        self.raw_points = np.asarray(points, dtype=float)
        self.degree = degree
        self.num_points = num_points
        self.finger_name = finger_name
        self.points = self._order_points_by_pca(self.raw_points) if sort_by_pca else self.raw_points
        self.coeffs = None
        self.curve_func = None
        self.fitted_points = None

    def _order_points_by_pca(self, points: np.ndarray) -> np.ndarray:
        center = points - np.mean(points, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(center, full_matrices=False)
        return points[np.argsort(center @ vt[0])]

    def _arc_length_parameter(self, points: np.ndarray) -> np.ndarray:
        n = len(points)
        if n <= 1:
            return np.array([0.0]) if n == 1 else np.array([])
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        u = np.zeros(n)
        u[1:] = np.cumsum(distances)
        return u / u[-1] if u[-1] > 0 else np.linspace(0.0, 1.0, n)

    def fit(self):
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3). Got {self.points.shape}.")
        if len(self.points) < (self.degree + 1):
            raise ValueError(f"Polynomial degree {self.degree} needs at least {self.degree + 1} points.")
        
        u = self._arc_length_parameter(self.points)
        self.coeffs = np.column_stack([np.polyfit(u, self.points[:, dim], self.degree) for dim in range(3)])
        
        self.curve_func = lambda u_vals: np.column_stack([np.polyval(self.coeffs[:, d], u_vals) for d in range(3)])
        self.fitted_points = self.curve_func(np.linspace(0, 1, self.num_points))
        
        return self.coeffs, self.curve_func, self.fitted_points
    
    def print_equation(self):
        if self.coeffs is None:
            print("No fit performed yet. Call fit() first.")
            return
        
        def format_poly(coeffs):
            terms = []
            for i, c in enumerate(coeffs):
                power = len(coeffs) - 1 - i
                if abs(c) < 1e-10:
                    continue
                sign = '+' if c >= 0 else '-'
                val = abs(c)
                if power == 0:
                    terms.append(f"{sign} {val:.6f}")
                elif power == 1:
                    terms.append(f"{sign} {val:.6f}*u")
                else:
                    terms.append(f"{sign} {val:.6f}*u^{power}")
            eq = ' '.join(terms) if terms else "0"
            return eq[2:] if eq.startswith('+') else eq
        
        print("Polynomial Curve Fit:")
        print(f"x(u) = {format_poly(self.coeffs[:, 0])}")
        print(f"y(u) = {format_poly(self.coeffs[:, 1])}")
        print(f"z(u) = {format_poly(self.coeffs[:, 2])}")
        print(f"where u ∈ [0, 1]")
    
    def save_coeffs(self, filepath: str = "finger_coeffs.json"):
        if self.coeffs is None:
            raise ValueError("No coefficients to save. Call fit() first.")
        if self.finger_name is None:
            raise ValueError("finger_name must be set to save coefficients.")
        
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath)) if os.path.exists(filepath) else {}
            all_data[self.finger_name] = {
                'degree': int(self.degree),
                'num_points': int(self.num_points),
                'coeffs': self.coeffs.tolist()
            }
            json.dump(all_data, open(filepath, 'w'), indent=2)
        elif filepath.endswith('.npy'):
            np.save(filepath.replace('.npy', f'_{self.finger_name}.npy'), self.coeffs)
        else:
            raise ValueError("Filepath must end with .json or .npy")
    
    @classmethod
    def load_coeffs(cls, finger_name: str, filepath: str = "finger_coeffs.json"):
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath))
            if finger_name not in all_data:
                raise ValueError(f"Finger '{finger_name}' not found. Available: {list(all_data.keys())}")
            data = all_data[finger_name]
            coeffs = np.array(data['coeffs'])
            degree, num_points = data['degree'], data.get('num_points', 5000)
        elif filepath.endswith('.npy'):
            npy_path = filepath.replace('.npy', f'_{finger_name}.npy')
            coeffs = np.load(npy_path)
            degree, num_points = len(coeffs) - 1, 5000
        else:
            raise ValueError("Filepath must end with .json or .npy")
        
        instance = cls(np.zeros((degree+1, 3)), degree, num_points, finger_name=finger_name)
        instance.coeffs = coeffs
        instance.curve_func = lambda u: np.column_stack([np.polyval(coeffs[:, d], u) for d in range(3)])
        instance.fitted_points = instance.curve_func(np.linspace(0, 1, num_points))
        return instance


def find_nearest_point(point: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, float, float]:
    from scipy.optimize import minimize_scalar
    point = np.asarray(point, dtype=float).copy()  # Make sure we have a copy
    def squared_distance(u):
        curve_point = np.array([np.polyval(coeffs[:, dim], u) for dim in range(3)])
        return np.sum((curve_point - point) ** 2)
    
    result = minimize_scalar(squared_distance, bounds=(0, 1), method='bounded')
    
    u_optimal = result.x
    min_distance = np.sqrt(result.fun)
    
    # If we're very close to a boundary, do a more thorough grid search to ensure we found the true minimum
    # This helps when targets are far from the curve and optimization might miss better solutions
    if u_optimal < 0.01 or u_optimal > 0.99:
        # Sample more densely near boundaries
        u_grid = np.concatenate([
            np.linspace(0, 0.1, 100),
            np.linspace(0.1, 0.9, 200),
            np.linspace(0.9, 1.0, 100)
        ])
        distances = np.array([squared_distance(u) for u in u_grid])
        u_grid_optimal = u_grid[np.argmin(distances)]
        grid_min_distance = np.sqrt(np.min(distances))
        
        # Use grid search result if it's better
        if grid_min_distance < min_distance:
            u_optimal = u_grid_optimal
            min_distance = grid_min_distance
    
    nearest_point = np.array([np.polyval(coeffs[:, dim], u_optimal) for dim in range(3)])
    
    return nearest_point, u_optimal, min_distance
    


if __name__ == "__main__":
    one_dof_fingertips_index = np.load("one_dof_fingertips_index.npy")
    two_dof_fingertips_index = np.load("two_dof_fingertips_index.npy")

    coeff_file = "finger_coeffs.json"
    finger_name = "index"
    
    if os.path.exists(coeff_file):
        try:
            one_dof_curve_fitting = PolynomialCurveFitting.load_coeffs(finger_name, coeff_file)
        except (ValueError, KeyError):
            one_dof_curve_fitting = PolynomialCurveFitting(one_dof_fingertips_index, finger_name=finger_name)
            one_dof_curve_fitting.fit()
            one_dof_curve_fitting.save_coeffs(coeff_file)
    else:
        one_dof_curve_fitting = PolynomialCurveFitting(one_dof_fingertips_index, finger_name=finger_name)
        one_dof_curve_fitting.fit()
        one_dof_curve_fitting.save_coeffs(coeff_file)
    
    one_dof_curve_fitting.print_equation()
    
    test_point = np.array([0.09, -0.02, 0.15]) 
    nearest, u_opt, distance = find_nearest_point(test_point, one_dof_curve_fitting.coeffs)
    
    print(f"\nNearest Point Search ({finger_name}):")
    print(f"Query: {test_point}")
    print(f"Nearest: {nearest}")
    print(f"u: {u_opt:.6f}, distance: {distance:.6f}")
    
    
    server = viser.ViserServer()
    
    urdf = yourdfpy.URDF.load("/Users/yiboyan/Desktop/inspire_sdk/rh56_urdf/non-wrist-inspire-right.urdf")
    
    viser_urdf = ViserUrdf(
        server,  
        urdf,
        root_node_name="/robot_hand"
    )
    
    num_joints = len(urdf.actuated_joint_names)
    joint_angles = np.zeros(num_joints)
    viser_urdf.update_cfg(joint_angles)
    print(f"Number of joints: {num_joints}")
    print(f"Joint names: {urdf.actuated_joint_names}")

    server.scene.add_point_cloud(
        name="/one_dof_fingertips_index",
        points=one_dof_fingertips_index,
        colors=(0.0, 0.0, 1.0), 
        point_size=0.001,
        point_shape='circle'
    )

    server.scene.add_point_cloud(
        name="/two_dof_fingertips_index",
        points=two_dof_fingertips_index,
        colors=(0.0, 1.0, 0.0), 
        point_size=0.001,
        point_shape='circle'
    )

    while True:
        time.sleep(0.01)
