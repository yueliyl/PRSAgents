"""Source PRS estimators: VIPRS (mean-field VI, as in Assignment 1) and LASSO."""
import numpy as np


def standardize(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd < 1e-8] = 1.0
    return (X - mu) / sd, mu, sd


def summary_stats(Xs, y):
    """Marginal effect sizes and in-sample LD, exactly as in Assignment 1."""
    n = Xs.shape[0]
    yc = (y - y.mean()) / (y.std() + 1e-12)
    beta_marg = Xs.T @ yc / n
    R = Xs.T @ Xs / n
    return beta_marg, R, n


def viprs(beta_marg, R, n, block_of_snp=None, h2=0.5, pi=0.01,
          n_iter=120, tol=1e-7):
    """Mean-field variational inference for a spike-and-slab PRS.

    Coordinate ascent over per-variant posterior inclusion gamma_j, posterior
    mean mu_j and variance s2_j, with sigma_beta^2 and pi updated by empirical
    Bayes.  Returns the posterior mean effect vector gamma * mu.

    LD is block diagonal here (the blocks are unlinked), so the sweep runs
    within each block against that block's LD submatrix.  This is both exact
    and an order of magnitude faster than sweeping against the full matrix.
    """
    m = len(beta_marg)
    sigma_b2 = h2 / (m * pi)
    sigma_e2 = 1.0 - h2
    if block_of_snp is None:
        block_of_snp = np.zeros(m, dtype=int)
    groups = [np.where(block_of_snp == b)[0] for b in np.unique(block_of_snp)]
    Rb = [R[np.ix_(g, g)] for g in groups]

    mu = np.zeros(m)
    gamma = np.full(m, pi)
    q = gamma * mu
    s2 = sigma_e2 / (n + sigma_e2 / sigma_b2)

    for _ in range(n_iter):
        q_old = q.copy()
        for g, Rg in zip(groups, Rb):
            qg = q[g]
            Rq = Rg @ qg
            for jj in range(len(g)):
                Rq -= Rg[:, jj] * qg[jj]
                s2 = sigma_e2 / (n + sigma_e2 / sigma_b2)
                mu_j = (n * s2 / sigma_e2) * (beta_marg[g[jj]] - Rq[jj])
                u = (np.log(pi / (1 - pi)) + 0.5 * np.log(s2 / sigma_b2)
                     + mu_j ** 2 / (2 * s2))
                u = np.clip(u, -30, 30)
                gj = 1.0 / (1.0 + np.exp(-u))
                gamma[g[jj]] = gj
                mu[g[jj]] = mu_j
                qg[jj] = gj * mu_j
                Rq += Rg[:, jj] * qg[jj]
            q[g] = qg
        # Empirical-Bayes updates, kept gently bounded for stability.
        pi = float(np.clip(gamma.mean(), 1e-4, 0.5))
        sigma_b2 = float(max((gamma * (mu ** 2 + s2)).sum() / (gamma.sum() + 1e-9), 1e-8))
        if np.max(np.abs(q - q_old)) < tol:
            break
    return q


def lasso_weights(Xs, y, seed=0):
    """LASSO on individual-level genotypes, alpha chosen on a held-out fifth."""
    from sklearn.linear_model import Lasso
    rng = np.random.default_rng(seed)
    n = Xs.shape[0]
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr, va = idx[:cut], idx[cut:]
    yc = (y - y.mean()) / (y.std() + 1e-12)
    best, best_r = None, -np.inf
    amax = np.abs(Xs[tr].T @ yc[tr]).max() / len(tr)
    for a in amax * np.array([0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]):
        mdl = Lasso(alpha=a, fit_intercept=True, max_iter=5000)
        mdl.fit(Xs[tr], yc[tr])
        pred = Xs[va] @ mdl.coef_
        if pred.std() < 1e-10:
            continue
        r = np.corrcoef(pred, yc[va])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best = r, mdl.coef_.copy()
    return best if best is not None else np.zeros(Xs.shape[1])
