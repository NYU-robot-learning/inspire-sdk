import mujoco
import numpy as np
import json
import os
from scipy.optimize import curve_fit


class SigmoidRegression:
    def __init__(self, points: np.ndarray, num_points: int = 10000, 
                 sort_by_x: bool = True, finger_name: str = None):
        """2D sigmoid regression: z = sigmoid(percentage). Points shape (N, 2): [percentage, z]."""
        self.raw_points = np.asarray(points, dtype=float)
        if self.raw_points.ndim != 2 or self.raw_points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2). Got {self.raw_points.shape}.")
        
        self.num_points = num_points
        self.finger_name = finger_name
        
        if sort_by_x:
            sort_idx = np.argsort(self.raw_points[:, 0])
            self.points = self.raw_points[sort_idx]
        else:
            self.points = self.raw_points
        
        self.coeffs = None
        self.curve_func = None
        self.fitted_points = None
    
    @staticmethod
    def _sigmoid(x, a, b, k, x0):
        """Sigmoid: a + b / (1 + exp(-k(x - x0)))."""
        return a + b / (1 + np.exp(-k * (x - x0)))
    
    def fit(self):
        """Fit sigmoid curve. Returns (coeffs, curve_func, fitted_points)."""
        if len(self.points) < 4:
            raise ValueError(f"Sigmoid fitting needs at least 4 points.")
        
        x = self.points[:, 0]
        z = self.points[:, 1]
        
        z_min, z_max = z.min(), z.max()
        a0 = z_min
        b0 = z_max - z_min
        k0 = 0.1
        x0_init = x.mean()
        
        try:
            popt, _ = curve_fit(
                self._sigmoid, x, z,
                p0=[a0, b0, k0, x0_init],
                maxfev=10000
            )
            self.coeffs = popt
        except RuntimeError:
            print(f"Warning: Optimization failed, using initial guess.")
            self.coeffs = np.array([a0, b0, k0, x0_init])
        
        self.curve_func = lambda x_vals: self._sigmoid(x_vals, *self.coeffs)
        x_fitted = np.linspace(0, 100, self.num_points)
        z_fitted = self.curve_func(x_fitted)
        self.fitted_points = np.column_stack([x_fitted, z_fitted])
        
        return self.coeffs, self.curve_func, self.fitted_points
    
    def print_equation(self):
        """Print fitted sigmoid equation."""
        if self.coeffs is None:
            print("No fit performed yet. Call fit() first.")
            return
        
        a, b, k, p0 = self.coeffs
        print("Sigmoid Curve Fit:")
        print(f"\ny(percentage) = {a:.6f} + {b:.6f} / (1 + exp(-{k:.6f}(percentage - {p0:.6f})))")
        print(f"\nwhere percentage ∈ [0, 100]")
        print("\nParameter meanings:")
        print(f"  a  = {a:.6f} → lower asymptote (y-value at percentage=0)")
        print(f"  b  = {b:.6f} → range (upper - lower asymptote)")
        print(f"  k  = {k:.6f} → steepness/growth rate")
        print(f"  p₀ = {p0:.6f} → midpoint (inflection point) in percentage")
    
    def plot(self, save_path: str = None, show: bool = True):
        """Plot fitted curve and original points."""
        if self.coeffs is None:
            raise ValueError("No fit performed yet. Call fit() first.")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        
        plt.scatter(self.points[:, 0], self.points[:, 1], 
                   alpha=0.5, s=10, color='blue', label='Original Data Points')
        
        plt.plot(self.fitted_points[:, 0], self.fitted_points[:, 1], 
                'r-', linewidth=2.5, label='Fitted Sigmoid Curve')
        
        a, b, k, p0 = self.coeffs
        z_midpoint = self.curve_func(p0)
        plt.plot(p0, z_midpoint, 'go', markersize=10, label=f'Midpoint (p₀={p0:.2f})')
        
        plt.xlabel('Percentage (%)', fontsize=12)
        plt.ylabel('Y position', fontsize=12)
        plt.title('Sigmoid Curve Fit: Percentage vs Y Position', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def y_to_percentage(self, y_value: float) -> float:
        """Convert y-value to percentage [0-100] using instance coefficients."""
        if self.coeffs is None:
            raise ValueError("No coefficients loaded. Call fit() or load_coeffs() first.")
        
        a, b, k, p0 = self.coeffs
        
        # Handle degenerate case: if b is essentially zero, y is constant
        # Return midpoint percentage as default when y doesn't vary
        if abs(b) < 1e-10:
            return p0
        
        y_min = min(a, a + b)
        y_max = max(a, a + b)
        
        y_value_clamped = np.clip(y_value, y_min, y_max)
        if y_value != y_value_clamped:
            y_value = y_value_clamped
        
        numerator = a + b - y_value
        denominator = y_value - a
        
        if abs(denominator) < 1e-10:
            return 0.0 if y_value <= a else 100.0
        if abs(numerator) < 1e-10:
            return 100.0
        
        ratio = numerator / denominator
        if ratio <= 0:
            return 0.0 if y_value <= a else 100.0
        
        percentage = p0 - (1.0 / k) * np.log(ratio)
        percentage = np.clip(percentage, 0.0, 100.0)
        
        return percentage
    
    def predict(self, y_value: float) -> float:
        """Predict DOF ratio [0-1] from y-value. This is the main prediction function."""
        percentage = self.y_to_percentage(y_value)
        return percentage / 100.0
    
    @staticmethod
    def y_to_percentage_static(y_value: float, finger_name: str, filepath: str = "sigmoid_coeffs.json") -> float:
        """Convert y-value to percentage [0-100] using saved coefficients."""
        instance = SigmoidRegression.load_coeffs(finger_name, filepath)
        return instance.y_to_percentage(y_value)
    
    def save_coeffs(self, filepath: str = "sigmoid_coeffs.json"):
        """Save coefficients to JSON or NPY file."""
        if self.coeffs is None:
            raise ValueError("No coefficients to save. Call fit() first.")
        if self.finger_name is None:
            raise ValueError("finger_name must be set to save coefficients.")
        
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath)) if os.path.exists(filepath) else {}
            all_data[self.finger_name] = {
                'num_points': int(self.num_points),
                'coeffs': self.coeffs.tolist()
            }
            json.dump(all_data, open(filepath, 'w'), indent=2)
        elif filepath.endswith('.npy'):
            np.save(filepath.replace('.npy', f'_{self.finger_name}.npy'), self.coeffs)
        else:
            raise ValueError("Filepath must end with .json or .npy")
    
    @classmethod
    def load_coeffs(cls, finger_name: str, filepath: str = "sigmoid_coeffs.json"):
        """Load coefficients from JSON or NPY file."""
        if filepath.endswith('.json'):
            all_data = json.load(open(filepath))
            if finger_name not in all_data:
                raise ValueError(f"Finger '{finger_name}' not found. Available: {list(all_data.keys())}")
            data = all_data[finger_name]
            coeffs = np.array(data['coeffs'])
            num_points = data.get('num_points', 10000)
        elif filepath.endswith('.npy'):
            npy_path = filepath.replace('.npy', f'_{finger_name}.npy')
            coeffs = np.load(npy_path)
            num_points = 10000
        else:
            raise ValueError("Filepath must end with .json or .npy")
        
        instance = cls(np.zeros((10, 2)), num_points, finger_name=finger_name)
        instance.coeffs = coeffs
        instance.curve_func = lambda x: cls._sigmoid(x, *coeffs)
        x_fitted = np.linspace(0, 100, num_points)
        instance.fitted_points = np.column_stack([x_fitted, instance.curve_func(x_fitted)])
        return instance


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("rh56_urdf/non-wrist-inspire-right.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    y_values = []
    percentages = []

    for i in np.arange(0, 100, 0.01):
        index_mcp = i / 100 * 1.34
        index_dip = i / 100 * 1.6
        data.qpos[6] = index_mcp
        data.qpos[7] = index_dip
        mujoco.mj_forward(model, data)
        site_id = model.site("right_index_tip_sphere").id
        y_values.append(data.site_xpos[site_id][1])
        percentages.append(i)

    np.save("percentage_to_y.npy", np.array([percentages, y_values]))

    percentages, y_values = np.load("percentage_to_y.npy")
    print(percentages.shape, y_values.shape)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, y_values)
    plt.xlabel('Percentage (%)')
    plt.ylabel('Y position')
    plt.title('Index Finger Tip Y Position vs Percentage')
    plt.grid(True)
    plt.savefig('percentage_to_y.png', dpi=300, bbox_inches='tight')
    print("Plot saved to percentage_to_y.png")

    points = np.column_stack([percentages, y_values])
    sigmoid_regression = SigmoidRegression(points, num_points=10000, finger_name="index_percentage_to_y")
    sigmoid_regression.fit()
    sigmoid_regression.print_equation()
    sigmoid_regression.save_coeffs("sigmoid_coeffs.json")
    sigmoid_regression.plot(save_path='sigmoid_fit.png', show=False)
    
    test_y = 0.15
    try:
        test_percentage = SigmoidRegression.y_to_percentage_static(test_y, "index_percentage_to_y", "sigmoid_coeffs.json")
        print(f"\nExample: y = {test_y} → percentage = {test_percentage:.2f}%")
    except ValueError as e:
        print(f"\nError: {e}")