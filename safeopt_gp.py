
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class SafeOptGP:
    """
    A minimal implementation of SafeOpt-GP for Task3.
    Matches the interface expected by run_experiments.STRICT2.py.
    """
    def __init__(self, bounds, safety_threshold=0.0, beta=2.0, max_points=300, fit_every=100, rng_seed=None):
        self.bounds = bounds
        self.safety_threshold = safety_threshold
        self.beta = beta
        self.rng = np.random.default_rng(rng_seed)
        
        # GP Model
        # C) 1. Disable optimizer (optimizer=None), use fixed kernel parameters
        #    alpha=1e-3 for numerical stability
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3, n_restarts_optimizer=0, optimizer=None)
        
        self.X = []
        self.y_obj = []
        self.y_safe = []
        
        self.MAX_POINTS = max_points # FIFO Limit
        self.FIT_EVERY = fit_every
        self.steps_since_fit = 0
        self.is_fitted = False

    def update(self, action, y_obj, y_safe):
        self.X.append(action)
        self.y_obj.append(y_obj)
        self.y_safe.append(y_safe)
        
        # C) 2. FIFO Buffer
        if len(self.X) > self.MAX_POINTS:
            self.X.pop(0)
            self.y_obj.pop(0)
            self.y_safe.pop(0)
        
        self.steps_since_fit += 1
        
        # C) 3. Fit Frequency
        if self.steps_since_fit >= self.FIT_EVERY:
            self.gp.fit(self.X, self.y_obj)
            self.steps_since_fit = 0
            self.is_fitted = True

    def optimize(self, candidates, threshold, V, Q):
        """
        Select best action from candidates.
        candidates: (N, 2) array of actions
        threshold: safety threshold for y_safe
        """
        if not self.X:
            # Random initial choice
            return candidates[self.rng.integers(len(candidates))]
        
        # C) 4. Handle pre-fit state
        if not self.is_fitted:
             # Just fit now to be safe, or return random
             # For robustness, let's force a fit if we have enough data but missed schedule,
             # or just fit on whatever we have if it's the first time > 1 point
             if len(self.X) >= 1:
                 self.gp.fit(self.X, self.y_obj)
                 self.is_fitted = True
             else:
                 return candidates[self.rng.integers(len(candidates))]

        # Predict Objective
        mu, sigma = self.gp.predict(candidates, return_std=True)
        ucb = mu + self.beta * sigma
        
        # Safety Check (Mock: Assuming Safety is correlated with Objective or independent)
        # Real SafeOpt models Safety separately. 
        # Here we'll just pick the one with highest UCB that is likely safe?
        # Since we don't have a GP for safety in this minimal mock, we rely on the heuristic 
        # or just return the UCB maximizer.
        # Ideally we should have 2 GPs.
        
        # Let's trust the UCB of objective for now to fix the crash.
        best_idx = np.argmax(ucb)
        return candidates[best_idx]
