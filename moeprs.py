"""Compact teaching MoEPRS for Stage 2 and Stage 3 (NumPy only).

Conceptual reference: https://github.com/shz9/moe-prs-paper/blob/main/model/moe.py
This independent teaching implementation is not a copy of the paper code.
It uses Gaussian experts with one shared residual variance, scalar PRS
calibration, global covariates, and a PC-only softmax gate. Inputs are fixed
source scores S and C=[age, sex, PC1, ..., PC10]; fit preprocessing on training
individuals only. The conditional mean is:
    a0 + C @ alpha + sum_k pi_k(PC) * gamma_k * S_k.
Stage 2 uses gate="linear", expert="linear". Quadratic PC features and a
fixed monotone power link are optional Stage 3 extensions.
"""
import numpy as np


def _softmax(A):
    A = A - A.max(axis=1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(axis=1, keepdims=True)


def fit_gate(G, R, l2=1e-2, steps=400, lr=0.1, theta=None):
    """Multinomial logistic fit to soft targets R (the E-step responsibilities)."""
    n, p = G.shape
    K = R.shape[1]
    theta = np.zeros((p, K)) if theta is None else theta.copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    for t in range(1, steps + 1):
        P = _softmax(G @ theta)
        grad = G.T @ (P - R) / n + l2 * theta
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        theta -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return theta


class MoEPRS:
    """Mixture of experts over pre-fit PRS, gate on the leading PCs."""

    def __init__(self, n_pc=10, l2_gate=1e-2, n_em=40, tol=1e-6, seed=0,
                 gate="linear", expert="linear"):
        self.n_pc, self.l2_gate, self.n_em, self.tol, self.seed = (
            n_pc, l2_gate, n_em, tol, seed)
        self.gate, self.expert = gate, expert

    def _gate_features(self, C):
        """PC-only gate; quadratic features are a Stage 3 option."""
        PC = C[:, 2:2 + self.n_pc]
        if self.gate == "linear":
            return np.column_stack([np.ones(len(C)), PC])
        rad = np.sqrt((PC ** 2).sum(1, keepdims=True))
        return np.column_stack([np.ones(len(C)), PC, PC ** 2, rad])

    def fit(self, S, C, y):
        n, K = S.shape
        G = self._gate_features(C)
        Cb = np.column_stack([np.ones(n), C])
        rng = np.random.default_rng(self.seed)

        # Initialize from the global (MultiPRS) solution plus a little noise, so
        # the experts are not identical at the first E-step.
        gl = np.linalg.lstsq(np.column_stack([Cb, S]), y, rcond=None)[0]
        alpha = gl[:Cb.shape[1]]
        gamma = gl[Cb.shape[1]:] * K + 0.01 * rng.standard_normal(K)
        theta = np.zeros((G.shape[1], K))
        sigma2 = np.var(y - Cb @ alpha) + 1e-8
        prev = -np.inf
        self.loglik_history_ = []

        for _ in range(self.n_em):
            base = Cb @ alpha
            mean = base[:, None] + self._expert_basis(S) * gamma[None, :]
            logpi = np.log(_softmax(G @ theta) + 1e-12)
            logp = logpi - (y[:, None] - mean) ** 2 / (2 * sigma2) \
                - 0.5 * np.log(2 * np.pi * sigma2)
            mx = logp.max(axis=1, keepdims=True)
            W = np.exp(logp - mx)
            ll = float((mx.ravel() + np.log(W.sum(axis=1))).sum())
            self.loglik_history_.append(ll)
            R = W / W.sum(axis=1, keepdims=True)

            # M-step: scalings, covariate effects, noise, then the gate.
            E = self._expert_basis(S)
            num = (R * E * (y[:, None] - base[:, None])).sum(axis=0)
            den = (R * E ** 2).sum(axis=0) + 1e-9
            gamma = num / den
            alpha = np.linalg.lstsq(Cb, y - (R * E * gamma[None, :]).sum(axis=1),
                                    rcond=None)[0]
            base = Cb @ alpha
            mean = base[:, None] + E * gamma[None, :]
            sigma2 = float((R * (y[:, None] - mean) ** 2).sum() / n) + 1e-10
            theta = fit_gate(G, R, l2=self.l2_gate, theta=theta, steps=120)

            if ll - prev < self.tol * abs(prev):
                break
            prev = ll

        self.alpha, self.gamma, self.theta, self.sigma2 = alpha, gamma, theta, sigma2
        # Record likelihood at the returned parameters, after the final M-step.
        logp = np.log(self.mixing(C) + 1e-12) - (y[:, None] - mean) ** 2 / (2 * sigma2) - 0.5 * np.log(2 * np.pi * sigma2)
        mx = logp.max(axis=1, keepdims=True)
        self.loglik_ = float((mx.ravel() + np.log(np.exp(logp - mx).sum(axis=1))).sum())
        self.n_iter_ = len(self.loglik_history_)
        self.loglik_history_.append(self.loglik_)
        return self

    def _expert_basis(self, S):
        """Fixed PRS; optional signed-power basis for Stage 3."""
        if self.expert == "linear":
            return S
        return np.sign(S) * np.abs(S) ** 1.5

    def mixing(self, C):
        return _softmax(self._gate_features(C) @ self.theta)

    def predict(self, S, C):
        Cb = np.column_stack([np.ones(len(C)), C])
        E = self._expert_basis(S)
        return Cb @ self.alpha + (self.mixing(C) * E * self.gamma[None, :]).sum(1)

