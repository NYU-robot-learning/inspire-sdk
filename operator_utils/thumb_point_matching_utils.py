import numpy as np
import json
import os
from scipy.optimize import minimize


class QuadraticSurfaceFitting:
    def __init__(self, points: np.ndarray, num_points_u: int = 50, num_points_v: int = 50, 
                 sort_by_pca: bool = True, finger_name: str = None):
        self.raw_points = np.asarray(points, dtype=float)
        self.num_points_u = num_points_u
        self.num_points_v = num_points_v
        self.finger_name = finger_name
        self.points = self._order_points_by_pca(self.raw_points) if sort_by_pca else self.raw_points
        self.coeffs = None
        self.surface_func = None
        self.fitted_points = None
        self.uv_params = None
        self.u_bounds = None
        self.v_bounds = None
        
    def _order_points_by_pca(self, points: np.ndarray) -> np.ndarray:
        center = points - np.mean(points, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(center, full_matrices=False)
        proj = center @ vt[:2].T
        sort_idx = np.lexsort((proj[:, 1], proj[:, 0]))
        return points[sort_idx]
    
    def _parameterize_points(self, points: np.ndarray) -> np.ndarray:
        center = points - np.mean(points, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(center, full_matrices=False)
        proj = center @ vt[:2].T
        u = (proj[:, 0] - proj[:, 0].min()) / (proj[:, 0].max() - proj[:, 0].min() + 1e-10)
        v = (proj[:, 1] - proj[:, 1].min()) / (proj[:, 1].max() - proj[:, 1].min() + 1e-10)
        return np.column_stack([u, v])
    
    def _build_quadratic_basis(self, uv: np.ndarray) -> np.ndarray:
        u, v = uv[:, 0], uv[:, 1]
        return np.column_stack([
            np.ones_like(u),
            u,
            v,
            u**2,
            u * v,
            v**2
        ])
    
    def fit(self):
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3). Got {self.points.shape}.")
        if len(self.points) < 6:
            raise ValueError(f"Quadratic surface needs at least 6 points. Got {len(self.points)}.")
        
        self.uv_params = self._parameterize_points(self.points)
        self.u_bounds = (self.uv_params[:, 0].min(), self.uv_params[:, 0].max())
        self.v_bounds = (self.uv_params[:, 1].min(), self.uv_params[:, 1].max())
        
        A = self._build_quadratic_basis(self.uv_params)
        self.coeffs = np.linalg.lstsq(A, self.points, rcond=None)[0]
        
        def surface_func(u_vals, v_vals):
            u_vals = np.asarray(u_vals)
            v_vals = np.asarray(v_vals)
            
            if u_vals.ndim == 0 and v_vals.ndim == 0:
                uv = np.array([[u_vals, v_vals]])
            elif u_vals.ndim == 0:
                uv = np.column_stack([np.full_like(v_vals, u_vals), v_vals])
            elif v_vals.ndim == 0:
                uv = np.column_stack([u_vals, np.full_like(u_vals, v_vals)])
            else:
                if u_vals.ndim == 1 and v_vals.ndim == 1:
                    U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
                    uv = np.column_stack([U.ravel(), V.ravel()])
                else:
                    uv = np.column_stack([u_vals.ravel(), v_vals.ravel()])
            
            basis = self._build_quadratic_basis(uv)
            points = basis @ self.coeffs
            
            if u_vals.ndim == 1 and v_vals.ndim == 1:
                n_u, n_v = len(u_vals), len(v_vals)
                points = points.reshape(n_u, n_v, 3)
            
            return points
        
        self.surface_func = surface_func
        
        u_min, u_max = self.u_bounds
        v_min, v_max = self.v_bounds
        u_grid = np.linspace(u_min, u_max, self.num_points_u)
        v_grid = np.linspace(v_min, v_max, self.num_points_v)
        self.fitted_points = self.surface_func(u_grid, v_grid)
        
        return self.coeffs, self.surface_func, self.fitted_points
    
    def save_coeffs(self, filepath: str = "surface_coeffs.json"):
        if self.coeffs is None:
            raise ValueError("No coefficients to save. Call fit() first.")
        if self.finger_name is None:
            raise ValueError("finger_name must be set to save coefficients.")
        
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath)) if os.path.exists(filepath) else {}
            all_data[self.finger_name] = {
                'num_points_u': int(self.num_points_u),
                'num_points_v': int(self.num_points_v),
                'coeffs': self.coeffs.tolist(),
                'u_bounds': list(self.u_bounds),
                'v_bounds': list(self.v_bounds)
            }
            json.dump(all_data, open(filepath, 'w'), indent=2)
        elif filepath.endswith('.npy'):
            np.save(filepath.replace('.npy', f'_{self.finger_name}.npy'), self.coeffs)
        else:
            raise ValueError("Filepath must end with .json or .npy")
    
    @classmethod
    def load_coeffs(cls, finger_name: str, filepath: str = "surface_coeffs.json"):
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath))
            if finger_name not in all_data:
                raise ValueError(f"Finger '{finger_name}' not found. Available: {list(all_data.keys())}")
            data = all_data[finger_name]
            coeffs = np.array(data['coeffs'])
            num_points_u = data.get('num_points_u', 50)
            num_points_v = data.get('num_points_v', 50)
        elif filepath.endswith('.npy'):
            npy_path = filepath.replace('.npy', f'_{finger_name}.npy')
            coeffs = np.load(npy_path)
            num_points_u, num_points_v = 50, 50
            u_bounds = [0.0, 1.0]
            v_bounds = [0.0, 1.0]
        else:
            raise ValueError("Filepath must end with .json or .npy")
        
        instance = cls(np.zeros((6, 3)), num_points_u, num_points_v, finger_name=finger_name)
        instance.coeffs = coeffs
        
        if filepath.endswith('.json'):
            u_bounds = data.get('u_bounds', [0.0, 1.0])
            v_bounds = data.get('v_bounds', [0.0, 1.0])
        
        instance.u_bounds = tuple(u_bounds)
        instance.v_bounds = tuple(v_bounds)
        
        def surface_func(u_vals, v_vals):
            u_vals = np.asarray(u_vals)
            v_vals = np.asarray(v_vals)
            
            if u_vals.ndim == 0 and v_vals.ndim == 0:
                uv = np.array([[u_vals, v_vals]])
            elif u_vals.ndim == 0:
                uv = np.column_stack([np.full_like(v_vals, u_vals), v_vals])
            elif v_vals.ndim == 0:
                uv = np.column_stack([u_vals, np.full_like(u_vals, v_vals)])
            else:
                if u_vals.ndim == 1 and v_vals.ndim == 1:
                    U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
                    uv = np.column_stack([U.ravel(), V.ravel()])
                else:
                    uv = np.column_stack([u_vals.ravel(), v_vals.ravel()])
            
            basis = instance._build_quadratic_basis(uv)
            points = basis @ coeffs
            
            if u_vals.ndim == 1 and v_vals.ndim == 1:
                n_u, n_v = len(u_vals), len(v_vals)
                points = points.reshape(n_u, n_v, 3)
            
            return points
        
        instance.surface_func = surface_func
        
        u_grid = np.linspace(u_bounds[0], u_bounds[1], num_points_u)
        v_grid = np.linspace(v_bounds[0], v_bounds[1], num_points_v)
        instance.fitted_points = instance.surface_func(u_grid, v_grid)
        
        return instance

def find_nearest_point_on_surface(point: np.ndarray, coeffs: np.ndarray, 
                                   u_bounds: tuple[float, float] = (0.0, 1.0),
                                   v_bounds: tuple[float, float] = (0.0, 1.0),
                                   data_points: np.ndarray = None) -> tuple[np.ndarray, float, float, float]:
    point = np.asarray(point, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    
    if coeffs.shape != (6, 3):
        raise ValueError(f"coeffs must have shape (6, 3). Got {coeffs.shape}.")
    
    u_min, u_max = u_bounds
    v_min, v_max = v_bounds
    
    data_min = None
    data_max = None
    if data_points is not None:
        data_points = np.asarray(data_points, dtype=float)
        data_min = data_points.min(axis=0) - 0.001
        data_max = data_points.max(axis=0) + 0.001
    
    def squared_distance(uv):
        u_val, v_val = uv[0], uv[1]
        basis = np.array([1.0, u_val, v_val, u_val**2, u_val * v_val, v_val**2])
        surface_point = basis @ coeffs
        
        penalty = 0.0
        if data_min is not None and data_max is not None:
            outside = np.any(surface_point < data_min) or np.any(surface_point > data_max)
            if outside:
                penalty = 1e6 * np.sum(np.maximum(0, data_min - surface_point) + 
                                       np.maximum(0, surface_point - data_max))
        
        return np.sum((surface_point - point) ** 2) + penalty
    
    u_center = (u_min + u_max) / 2
    v_center = (v_min + v_max) / 2
    
    result = minimize(squared_distance, x0=[u_center, v_center],
                     bounds=[(u_min, u_max), (v_min, v_max)], method='L-BFGS-B')
    
    u_optimal = result.x[0]
    v_optimal = result.x[1]
    
    basis_optimal = np.array([1.0, u_optimal, v_optimal, u_optimal**2, u_optimal * v_optimal, v_optimal**2])
    nearest_point = basis_optimal @ coeffs
    
    if data_points is not None:
        within_bounds = np.all(nearest_point >= data_min) and np.all(nearest_point <= data_max)
        
        if not within_bounds:
            u_grid = np.linspace(u_min, u_max, 100)
            v_grid = np.linspace(v_min, v_max, 100)
            U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
            uv_candidates = np.column_stack([U.ravel(), V.ravel()])
            
            basis_candidates = np.column_stack([
                np.ones(len(uv_candidates)),
                uv_candidates[:, 0],
                uv_candidates[:, 1],
                uv_candidates[:, 0]**2,
                uv_candidates[:, 0] * uv_candidates[:, 1],
                uv_candidates[:, 1]**2
            ])
            surface_candidates = basis_candidates @ coeffs
            
            within_mask = np.all((surface_candidates >= data_min) & (surface_candidates <= data_max), axis=1)
            
            if np.any(within_mask):
                valid_candidates = surface_candidates[within_mask]
                valid_uv = uv_candidates[within_mask]
                distances = np.linalg.norm(valid_candidates - point, axis=1)
                nearest_idx = np.argmin(distances)
                nearest_point = valid_candidates[nearest_idx]
                u_optimal, v_optimal = valid_uv[nearest_idx]
                min_distance = distances[nearest_idx]
            else:
                distances_to_data = np.linalg.norm(data_points - point, axis=1)
                nearest_idx = np.argmin(distances_to_data)
                target_point = data_points[nearest_idx]
                
                def dist_to_target(uv):
                    u_val, v_val = uv[0], uv[1]
                    basis = np.array([1.0, u_val, v_val, u_val**2, u_val * v_val, v_val**2])
                    surface_pt = basis @ coeffs
                    return np.sum((surface_pt - target_point) ** 2)
                
                uv_result = minimize(dist_to_target, x0=[u_center, v_center],
                                   bounds=[(u_min, u_max), (v_min, v_max)], method='L-BFGS-B')
                u_optimal, v_optimal = uv_result.x[0], uv_result.x[1]
                basis_optimal = np.array([1.0, u_optimal, v_optimal, u_optimal**2, u_optimal * v_optimal, v_optimal**2])
                nearest_point = basis_optimal @ coeffs
                min_distance = np.linalg.norm(nearest_point - point)
        else:
            min_distance = np.sqrt(result.fun)
    else:
        min_distance = np.sqrt(result.fun)
    
    return nearest_point, u_optimal, v_optimal, min_distance


if __name__ == "__main__":
    import os
    
    json_file = os.path.join(os.path.dirname(__file__), "thumb_surface_coeffs.json")
    
    if os.path.exists(json_file):
        loaded_fitting = QuadraticSurfaceFitting.load_coeffs("thumb", json_file)
        
        query_point = np.array([0.05, 0.05, 0.1])
        nearest, u, v, dist = find_nearest_point_on_surface(
            query_point, loaded_fitting.coeffs, loaded_fitting.u_bounds, loaded_fitting.v_bounds
        )
        
        print(f"Loaded coefficients from: {json_file}")
        print(f"Query point: {query_point}")
        print(f"Nearest point on surface: {nearest}")
        print(f"Parameters: u={u:.4f}, v={v:.4f}")
        print(f"Distance: {dist:.4f}")
    else:
        points = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [0.05, 0.05, 0.05],
            [0.05, 0.0, 0.02],
            [0.0, 0.05, 0.02],
            [0.1, 0.05, 0.02],
            [0.05, 0.1, 0.02]
        ])
        
        fitting = QuadraticSurfaceFitting(points, finger_name="example")
        coeffs, surface_func, fitted_points = fitting.fit()
        fitting.save_coeffs("example_coeffs.json")
        
        query_point = np.array([0.05, 0.05, 0.1])
        nearest, u, v, dist = find_nearest_point_on_surface(
            query_point, coeffs, fitting.u_bounds, fitting.v_bounds, data_points=points
        )
        
        print(f"Query point: {query_point}")
        print(f"Nearest point on surface: {nearest}")
        print(f"Parameters: u={u:.4f}, v={v:.4f}")
        print(f"Distance: {dist:.4f}")
        
        loaded_fitting = QuadraticSurfaceFitting.load_coeffs("example", "example_coeffs.json")
        print(f"\nLoaded from JSON - Coeffs match: {np.allclose(fitting.coeffs, loaded_fitting.coeffs)}")
