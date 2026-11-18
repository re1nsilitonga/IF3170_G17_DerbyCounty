import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import SVM


class SVMTests(unittest.TestCase):
    def test_linear_matches_sklearn_and_dual_scores(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        with threadpool_limits(limits=1):
            model = SVM(C=0.1, tol=1e-5, seed=42).fit(X, y)
            reference = SVC(kernel='linear', C=0.1, tol=1e-5).fit(X, y)
            repeated = SVM(C=0.1, tol=1e-5, seed=42).fit(X, y)
        self.assertTrue(model.converged_)
        np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X),
                                   atol=1e-3, rtol=0)
        dual_scores = X @ model.support_vectors_.T @ model.dual_coef_ + model.intercept_
        np.testing.assert_allclose(model.decision_function(X), dual_scores, atol=1e-10)
        np.testing.assert_array_equal(model.predict(X), reference.predict(X))
        np.testing.assert_array_equal(model.predict_proba(X), repeated.predict_proba(X))
        self.assertTrue(np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1)))
        self.assertTrue(np.all((model.alpha_ >= -1e-10) & (model.alpha_ <= model.C + 1e-10)))
        self.assertAlmostEqual(float(model.alpha_ @ (2 * y - 1)), 0.0, places=10)

    def test_nonlinear_kernels_still_separate_xor(self):
        X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]] * 3)
        y = np.array([0, 1, 1, 0] * 3)
        for kernel in ['poly', 'rbf']:
            with self.subTest(kernel=kernel), threadpool_limits(limits=1):
                model = SVM(kernel=kernel, degree=2, C=10, seed=42).fit(X, y)
                self.assertTrue(model.converged_)
                np.testing.assert_array_equal(model.predict(X), y)
                self.assertTrue(np.isfinite(model.predict_proba(X)).all())


if __name__ == '__main__':
    unittest.main()
