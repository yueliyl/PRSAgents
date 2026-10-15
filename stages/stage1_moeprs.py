"""Compatibility adapters and optional Stage 3 variants.

The provided Stage 2 model is in moeprs.py. The historical filename is
retained so existing agent scripts continue to work.
"""
import numpy as np
from moeprs import MoEPRS


class MultiPRS:
    """The paper's baseline: one global set of mixing weights."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, S, C, y):
        from sklearn.linear_model import RidgeCV
        X = np.column_stack([C, S])
        self.m = RidgeCV(alphas=np.logspace(-1, 4, 20)).fit(X, y)
        return self

    def predict(self, S, C):
        return self.m.predict(np.column_stack([C, S]))


# ---- harness adapters -------------------------------------------------------
def build_moeprs(S, Z, C, y, ctx=None, **kw):
    m = MoEPRS(**kw).fit(S, C, y)
    return lambda S2, Z2, C2: m.predict(S2, C2)


def build_moeprs_nlgate(S, Z, C, y, ctx=None):
    return build_moeprs(S, Z, C, y, gate="quadratic")


def build_moeprs_nlexpert(S, Z, C, y, ctx=None):
    return build_moeprs(S, Z, C, y, expert="power")


def build_moeprs_both(S, Z, C, y, ctx=None):
    return build_moeprs(S, Z, C, y, gate="quadratic", expert="power")


def build_multiprs(S, Z, C, y, ctx=None):
    m = MultiPRS().fit(S, C, y)
    return lambda S2, Z2, C2: m.predict(S2, C2)


# ---- further within-framework variants (Stage 3 search space) -----------
def build_moeprs_tuned(S, Z, C, y, ctx=None):
    """Same model, gate regularization chosen on a held-out quarter."""
    n = len(y)
    tr, va = np.arange(int(0.75 * n)), np.arange(int(0.75 * n), n)
    best, bm = None, None
    for l2 in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        m = MoEPRS(l2_gate=l2).fit(S[tr], C[tr], y[tr])
        sc = -((y[va] - m.predict(S[va], C[va])) ** 2).mean()
        if best is None or sc > best:
            best, bm = sc, l2
    m = MoEPRS(l2_gate=bm).fit(S, C, y)
    f = lambda S2, Z2, C2: m.predict(S2, C2)
    f.l2 = bm
    return f


def build_moeprs_npc(S, Z, C, y, ctx=None):
    """Same model, number of gate PCs chosen on a held-out quarter."""
    n = len(y)
    tr, va = np.arange(int(0.75 * n)), np.arange(int(0.75 * n), n)
    best, bm = None, None
    for npc in (2, 3, 5, 10):
        m = MoEPRS(n_pc=npc).fit(S[tr], C[tr], y[tr])
        sc = -((y[va] - m.predict(S[va], C[va])) ** 2).mean()
        if best is None or sc > best:
            best, bm = sc, npc
    m = MoEPRS(n_pc=bm).fit(S, C, y)
    f = lambda S2, Z2, C2: m.predict(S2, C2)
    f.n_pc = bm
    return f


def build_moeprs_window_experts(S, Z, C, y, ctx=None, l2_gate=1.0):
    """Expand the expert set: every (source, window) pair becomes an expert.

    Still a mixture of experts with a linear gate on the PCs. The experts are
    now locus-specific, so the gate can put weight on one source at one part of
    the genome and another elsewhere.
    """
    K = S.shape[1]
    W = Z.shape[1] // K
    Sx = np.column_stack([S, Z])
    m = MoEPRS(l2_gate=l2_gate).fit(Sx, C, y)

    def predict(S2, Z2, C2):
        return m.predict(np.column_stack([S2, Z2]), C2)
    return predict


def build_moeprs_restarts(S, Z, C, y, ctx=None, n_restart=6):
    """Same model, several EM restarts, keep the best training log-likelihood.

    The EM surface for a mixture of experts is not convex, so the Stage 2 fit
    can land in a poor local optimum depending on how gamma is initialized.
    """
    best = None
    for s in range(n_restart):
        m = MoEPRS(seed=s).fit(S, C, y)
        if best is None or m.loglik_ > best.loglik_:
            best = m
    return lambda S2, Z2, C2: best.predict(S2, C2)
