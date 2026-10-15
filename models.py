"""Integration models shared by the simulator (oracle-structure) and the
student harness (required baselines).

Every fit takes standardized source scores S (n, K), optional window contrasts
Z (n, K*W), covariates C (n, p) and the phenotype, and returns a predict
function.  Covariates are always in the model, so reported R2 is incremental.
"""
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression


def _design(C, *parts):
    return np.column_stack([C] + [p for p in parts if p is not None])


def incremental_r2(y, pred, C):
    """R2 of covariates + predictor, minus R2 of covariates alone."""
    base = LinearRegression().fit(C, y)
    full = LinearRegression().fit(_design(C, pred.reshape(len(pred), -1)), y)
    ss = ((y - y.mean()) ** 2).sum()
    r2b = 1 - ((y - base.predict(C)) ** 2).sum() / ss
    r2f = 1 - ((y - full.predict(_design(C, pred.reshape(len(pred), -1)))) ** 2).sum() / ss
    return r2f - r2b


def fit_covariate_only(S, Z, C, y):
    m = LinearRegression().fit(C, y)
    return lambda S2, Z2, C2: m.predict(C2)


def fit_best_single(S, Z, C, y, return_index=False):
    best, bi = None, 0
    for k in range(S.shape[1]):
        r = incremental_r2(y, S[:, k], C)
        if best is None or r > best:
            best, bi = r, k
    m = LinearRegression().fit(_design(C, S[:, [bi]]), y)
    f = lambda S2, Z2, C2: m.predict(_design(C2, S2[:, [bi]]))
    return (f, bi) if return_index else f


def fit_single(k):
    def f(S, Z, C, y):
        m = LinearRegression().fit(_design(C, S[:, [k]]), y)
        return lambda S2, Z2, C2: m.predict(_design(C2, S2[:, [k]]))
    return f


def fit_linear_stack(S, Z, C, y, alphas=(0.1, 1.0, 10.0, 100.0)):
    best, bm = None, None
    n = len(y)
    idx = np.arange(n)
    tr, va = idx[: int(0.75 * n)], idx[int(0.75 * n):]
    for a in alphas:
        m = Ridge(alpha=a).fit(_design(C[tr], S[tr]), y[tr])
        r = -((y[va] - m.predict(_design(C[va], S[va]))) ** 2).mean()
        if best is None or r > best:
            best, bm = r, a
    m = Ridge(alpha=bm).fit(_design(C, S), y)
    return lambda S2, Z2, C2: m.predict(_design(C2, S2))


def fit_window_weights(S, Z, C, y, alphas=(10.0, 100.0, 1000.0, 5000.0)):
    """BLocPRS-style: unpenalized genome-wide terms plus ridge on window contrasts."""
    best, bm = None, None
    n = len(y)
    idx = np.arange(n)
    tr, va = idx[: int(0.75 * n)], idx[int(0.75 * n):]
    for a in alphas:
        m = Ridge(alpha=a).fit(_design(C[tr], S[tr], Z[tr]), y[tr])
        r = -((y[va] - m.predict(_design(C[va], S[va], Z[va]))) ** 2).mean()
        if best is None or r > best:
            best, bm = r, a
    m = Ridge(alpha=bm).fit(_design(C, S, Z), y)
    return lambda S2, Z2, C2: m.predict(_design(C2, S2, Z2))


def fit_moe_gating(S, Z, C, y, seed=0):
    """PC-only Stage 2 EM reference, fixed linear gate and PRS experts."""
    from moeprs import MoEPRS
    m = MoEPRS(seed=seed, n_pc=10, l2_gate=0.01).fit(S, C, y)
    return lambda S2, Z2, C2: m.predict(S2, C2)


def fit_boosting(S, Z, C, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = _design(C, S)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                      max_depth=3, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.2,
                                      random_state=seed).fit(X, y)
    return lambda S2, Z2, C2: m.predict(_design(C2, S2))


def fit_boosting_windows(S, Z, C, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = _design(C, S, Z)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                      max_depth=3, l2_regularization=1.0,
                                      early_stopping=True, validation_fraction=0.2,
                                      random_state=seed).fit(X, y)
    return lambda S2, Z2, C2: m.predict(_design(C2, S2, Z2))


REGISTRY = {
    "best_single": fit_best_single,
    "linear_stack": fit_linear_stack,
    "moe_gating": fit_moe_gating,
    "window_weights": fit_window_weights,
    "boosting": fit_boosting,
    "boosting_windows": fit_boosting_windows,
}
