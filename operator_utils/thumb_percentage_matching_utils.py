import numpy as np
import json
import os
from scipy.optimize import minimize
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class PolynomialRegression2D:
    """Polynomial regression mapping (dof1_ratio, dof2_ratio) -> (x, y, z) thumb_tip position."""
    
    def __init__(self, dof1_ratios: np.ndarray, dof2_ratios: np.ndarray, 
                 thumb_tips: np.ndarray, degree: int = 3, finger_name: str = None):
        self.dof1_ratios = np.asarray(dof1_ratios, dtype=float).flatten()
        self.dof2_ratios = np.asarray(dof2_ratios, dtype=float).flatten()
        self.thumb_tips = np.asarray(thumb_tips, dtype=float)
        
        if len(self.dof1_ratios) != len(self.dof2_ratios) or len(self.dof1_ratios) != len(self.thumb_tips):
            raise ValueError(f"Input arrays must have same length.")
        
        if self.thumb_tips.ndim != 2 or self.thumb_tips.shape[1] != 3:
            raise ValueError(f"thumb_tips must have shape (N, 3).")
        
        self.degree = degree
        self.finger_name = finger_name
        self.coeffs = None
        self.prediction_func = None
        self.n_features = sum(range(self.degree + 2))
        self.dof1_range = None
        self.dof2_range = None
    
    def _build_polynomial_features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Build polynomial features: u^i * v^j where i + j <= degree."""
        features = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                features.append((u ** i) * (v ** j))
        return np.column_stack(features)
    
    def fit(self):
        """Fit polynomial regression model."""
        min_points = self.n_features
        if len(self.dof1_ratios) < min_points:
            raise ValueError(f"Polynomial degree {self.degree} needs at least {min_points} points.")
        
        X = self._build_polynomial_features(self.dof1_ratios, self.dof2_ratios)
        self.coeffs = np.linalg.lstsq(X, self.thumb_tips, rcond=None)[0]
        
        self.dof1_range = [float(self.dof1_ratios.min()), float(self.dof1_ratios.max())]
        self.dof2_range = [float(self.dof2_ratios.min()), float(self.dof2_ratios.max())]
        
        def predict_func(dof1, dof2):
            dof1 = np.asarray(dof1)
            dof2 = np.asarray(dof2)
            
            if dof1.ndim == 0 and dof2.ndim == 0:
                X_pred = self._build_polynomial_features(np.array([dof1]), np.array([dof2]))
                return X_pred @ self.coeffs
            elif dof1.ndim == 0:
                X_pred = self._build_polynomial_features(np.full_like(dof2, dof1), dof2)
                return X_pred @ self.coeffs
            elif dof2.ndim == 0:
                X_pred = self._build_polynomial_features(dof1, np.full_like(dof1, dof2))
                return X_pred @ self.coeffs
            else:
                if dof1.shape != dof2.shape:
                    raise ValueError(f"dof1 and dof2 must have same shape.")
                X_pred = self._build_polynomial_features(dof1.flatten(), dof2.flatten())
                return (X_pred @ self.coeffs).reshape(*dof1.shape, 3)
        
        self.prediction_func = predict_func
        return self.coeffs, self.prediction_func
    
    def predict(self, dof1_ratios: np.ndarray, dof2_ratios: np.ndarray) -> np.ndarray:
        """Predict thumb tip positions for given DOF ratios."""
        if self.prediction_func is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.prediction_func(dof1_ratios, dof2_ratios)
    
    def xyz_to_ratios(self, x: float, y: float, z: float) -> tuple:
        """Inverse mapping: given (x, y, z), return (dof1_ratio, dof2_ratio)."""
        if self.prediction_func is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        target = np.array([x, y, z])
        
        def error_func(params):
            dof1, dof2 = params
            predicted = self.prediction_func(dof1, dof2)
            if predicted.ndim == 2:
                predicted = predicted[0]
            return np.linalg.norm(predicted - target)
        
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        if self.dof1_range:
            bounds[0] = (max(0.0, self.dof1_range[0]), min(1.0, self.dof1_range[1]))
        if self.dof2_range:
            bounds[1] = (max(0.0, self.dof2_range[0]), min(1.0, self.dof2_range[1]))
        
        initial_guess = [0.5, 0.5]
        result = minimize(error_func, initial_guess, method='L-BFGS-B', bounds=bounds)
        
        return float(result.x[0]), float(result.x[1])
    
    def evaluate_error(self, dof1_ratios: np.ndarray = None, 
                      dof2_ratios: np.ndarray = None, 
                      thumb_tips: np.ndarray = None) -> dict:
        """Evaluate prediction error."""
        if self.prediction_func is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if dof1_ratios is None:
            dof1_ratios = self.dof1_ratios
        if dof2_ratios is None:
            dof2_ratios = self.dof2_ratios
        if thumb_tips is None:
            thumb_tips = self.thumb_tips
        
        predicted = self.predict(dof1_ratios, dof2_ratios)
        
        if predicted.ndim == 2:
            errors = np.linalg.norm(predicted - thumb_tips, axis=1)
        else:
            errors = np.linalg.norm(predicted - thumb_tips)
        
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'median_error': np.median(errors),
            'rmse': np.sqrt(np.mean(errors ** 2))
        }
    
    def save_coeffs(self, filepath: str = "thumb_polynomial_coeffs.json"):
        """Save coefficients to JSON file."""
        if self.coeffs is None:
            raise ValueError("No coefficients to save. Call fit() first.")
        if self.finger_name is None:
            raise ValueError("finger_name must be set to save coefficients.")
        
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath)) if os.path.exists(filepath) else {}
            all_data[self.finger_name] = {
                'degree': int(self.degree),
                'coeffs': self.coeffs.tolist(),
                'dof1_range': self.dof1_range if self.dof1_range else [0.0, 1.0],
                'dof2_range': self.dof2_range if self.dof2_range else [0.0, 1.0]
            }
            json.dump(all_data, open(filepath, 'w'), indent=2)
        elif filepath.endswith('.npy'):
            np.save(filepath.replace('.npy', f'_{self.finger_name}.npy'), self.coeffs)
        else:
            raise ValueError("Filepath must end with .json or .npy")
    
    @classmethod
    def load_coeffs(cls, finger_name: str, filepath: str = "thumb_polynomial_coeffs.json"):
        """Load coefficients from JSON file."""
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath))
            if finger_name not in all_data:
                raise ValueError(f"Finger '{finger_name}' not found. Available: {list(all_data.keys())}")
            data = all_data[finger_name]
            coeffs = np.array(data['coeffs'])
            degree = data['degree']
            dof1_range = data.get('dof1_range', [0.0, 1.0])
            dof2_range = data.get('dof2_range', [0.0, 1.0])
        elif filepath.endswith('.npy'):
            npy_path = filepath.replace('.npy', f'_{finger_name}.npy')
            coeffs = np.load(npy_path)
            n_features = len(coeffs)
            degree = int(np.ceil(np.sqrt(2 * n_features) - 1.5))
            dof1_range = [0.0, 1.0]
            dof2_range = [0.0, 1.0]
        else:
            raise ValueError("Filepath must end with .json or .npy")
        
        dummy_dof1 = np.array([0.0, 1.0])
        dummy_dof2 = np.array([0.0, 1.0])
        dummy_tips = np.zeros((2, 3))
        instance = cls(dummy_dof1, dummy_dof2, dummy_tips, degree, finger_name=finger_name)
        instance.coeffs = coeffs
        instance.dof1_range = dof1_range
        instance.dof2_range = dof2_range
        
        def predict_func(dof1, dof2):
            dof1 = np.asarray(dof1)
            dof2 = np.asarray(dof2)
            
            if dof1.ndim == 0 and dof2.ndim == 0:
                X_pred = instance._build_polynomial_features(np.array([dof1]), np.array([dof2]))
                return X_pred @ coeffs
            elif dof1.ndim == 0:
                X_pred = instance._build_polynomial_features(np.full_like(dof2, dof1), dof2)
                return X_pred @ coeffs
            elif dof2.ndim == 0:
                X_pred = instance._build_polynomial_features(dof1, np.full_like(dof1, dof2))
                return X_pred @ coeffs
            else:
                if dof1.shape != dof2.shape:
                    raise ValueError(f"dof1 and dof2 must have same shape.")
                X_pred = instance._build_polynomial_features(dof1.flatten(), dof2.flatten())
                return (X_pred @ coeffs).reshape(*dof1.shape, 3)
        
        instance.prediction_func = predict_func
        return instance


if __name__ == "__main__":
    # Minimal example: load coefficients and convert xyz to percentages
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coeff_file = os.path.join(script_dir, "thumb_polynomial_coeffs.json")
    
    model = PolynomialRegression2D.load_coeffs(
        finger_name="thumb",
        filepath=coeff_file
    )
    
    # Example: given xyz coordinates, get dof1_ratio and dof2_ratio
    x, y, z = -0.05, 0.08, 0.05
    dof1_ratio, dof2_ratio = model.xyz_to_ratios(x, y, z)
    
    # Verify round-trip: predict xyz from the ratios
    predicted_xyz = model.predict(dof1_ratio, dof2_ratio)
    if predicted_xyz.ndim == 2:
        predicted_xyz = predicted_xyz[0]
    
    print(f"Input xyz: ({x:.3f}, {y:.3f}, {z:.3f})")
    print(f"Output ratios: dof1={dof1_ratio:.3f}, dof2={dof2_ratio:.3f}")
    print(f"Predicted xyz: ({predicted_xyz[0]:.3f}, {predicted_xyz[1]:.3f}, {predicted_xyz[2]:.3f})")
    print(f"Error: {np.linalg.norm(predicted_xyz - np.array([x, y, z])):.6f}")
