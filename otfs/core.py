"""Numerical core of OTFS.

The implementation follows the algorithm used for the paper's main results.
It intentionally keeps the final Sinkhorn reconstruction identical to the
experimental code to preserve the paper's selection procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OTFSConfig:
    """Hyperparameters for transport and subset extraction."""

    budget: int = 32
    epsilon: float = 0.05
    gamma: float = 0.1
    source_prior: tuple[float, ...] = (0.5, 0.4, 0.1)
    iterations: int = 100
    tolerance: float = 1e-8
    coverage_lambda: float = 0.5
    coverage_gap: int = 4

    def validate(self, n_atoms: int, n_frames: int) -> None:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if n_frames <= 0:
            raise ValueError("at least one candidate frame is required")
        if self.epsilon <= 0 or self.gamma <= 0:
            raise ValueError("epsilon and gamma must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.coverage_lambda < 0 or self.coverage_gap <= 0:
            raise ValueError("coverage_lambda must be nonnegative and coverage_gap positive")
        prior = np.asarray(self.source_prior, dtype=np.float64)
        if prior.shape != (n_atoms,):
            raise ValueError(f"source_prior has {len(prior)} entries but evidence has {n_atoms} atoms")
        if not np.isfinite(prior).all() or np.any(prior <= 0) or prior.sum() <= 0:
            raise ValueError("source_prior must contain finite positive values")


@dataclass(frozen=True)
class OTFSResult:
    """Outputs produced by :func:`otfs_select`."""

    selected_positions: list[int]
    selected_frame_indices: list[int]
    frame_weights: FloatArray
    transport_plan: FloatArray
    normalized_evidence: FloatArray
    iterations_run: int


def minmax_normalize(values: ArrayLike) -> FloatArray:
    """Map an array to [0, 1], returning zeros for a constant array."""

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    if not np.isfinite(arr).all():
        raise ValueError("evidence contains NaN or infinity")
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def prepare_paper_evidence(
    semantic: ArrayLike,
    utility: ArrayLike,
    transition: ArrayLike,
) -> FloatArray:
    """Prepare the three evidence atoms as in the paper experiments.

    BLIP ITM semantic scores are already probabilities and are retained as-is.
    Utility and transition scores are independently min-max normalized.  The
    stacked compatibility matrix is then globally min-max normalized, matching
    the numerical path used to produce the paper's frame subsets.
    """

    sem = np.asarray(semantic, dtype=np.float64)
    util = np.asarray(utility, dtype=np.float64)
    trans = np.asarray(transition, dtype=np.float64)
    if sem.ndim != 1 or util.ndim != 1 or trans.ndim != 1:
        raise ValueError("semantic, utility, and transition must be one-dimensional")
    if not (len(sem) == len(util) == len(trans)):
        raise ValueError("all evidence sequences must have the same length")
    if len(sem) == 0:
        raise ValueError("evidence sequences cannot be empty")
    if not np.isfinite(sem).all() or np.any(sem < -1e-8) or np.any(sem > 1 + 1e-8):
        raise ValueError("semantic BLIP probabilities must lie in [0, 1]")
    stacked = np.stack((sem, minmax_normalize(util), minmax_normalize(trans)))
    return minmax_normalize(stacked)


def semi_unbalanced_sinkhorn(
    cost: ArrayLike,
    source_prior: ArrayLike,
    *,
    epsilon: float = 0.05,
    gamma: float = 0.1,
    iterations: int = 100,
    tolerance: float = 1e-8,
) -> tuple[FloatArray, int]:
    """Solve the semi-unbalanced entropic OT problem used by OTFS.

    The source marginal is constrained by the ``u`` update; the uniform target
    marginal is KL-relaxed through exponent ``gamma / (gamma + epsilon)``.
    A fixed iteration count is the paper default, with the same every-ten-step
    convergence check as the experiment implementation.
    """

    c = np.asarray(cost, dtype=np.float64)
    a = np.asarray(source_prior, dtype=np.float64)
    if c.ndim != 2 or c.size == 0:
        raise ValueError("cost must be a non-empty [atoms, frames] matrix")
    if a.shape != (c.shape[0],):
        raise ValueError("source_prior shape does not match the number of atoms")
    if not np.isfinite(c).all() or not np.isfinite(a).all():
        raise ValueError("cost and source_prior must be finite")
    if np.any(a <= 0) or a.sum() <= 0:
        raise ValueError("source_prior must be positive")
    if epsilon <= 0 or gamma <= 0 or iterations <= 0:
        raise ValueError("epsilon, gamma, and iterations must be positive")

    a = a / a.sum()
    n_frames = c.shape[1]
    b = np.full(n_frames, 1.0 / n_frames, dtype=np.float64)
    kernel = np.exp(-c / epsilon)
    u = np.ones(c.shape[0], dtype=np.float64)
    v = np.ones(n_frames, dtype=np.float64)
    exponent = gamma / (gamma + epsilon)
    iterations_run = iterations

    # 1e-8 is deliberately retained for bit-level compatibility with the
    # experiment implementation rather than using machine epsilon.
    for step in range(iterations):
        old_u, old_v = u.copy(), v.copy()
        u = a / (kernel @ v + 1e-8)
        v = (b / (kernel.T @ u + 1e-8)) ** exponent
        if step > 0 and step % 10 == 0:
            if np.max(np.abs(u - old_u)) < tolerance and np.max(np.abs(v - old_v)) < tolerance:
                iterations_run = step + 1
                break

    plan = u[:, None] * kernel * v[None, :]
    return plan, iterations_run


def coverage_objective(
    selected: Sequence[int], weights: ArrayLike, coverage_lambda: float, coverage_gap: int
) -> float:
    """Evaluate the paper's additive mass-plus-coverage objective."""

    w = np.asarray(weights, dtype=np.float64)
    chosen = list(selected)
    if not chosen:
        return 0.0
    value = float(w[chosen].sum())
    value += coverage_lambda * sum(
        min(right - left, coverage_gap) for left, right in zip(chosen[:-1], chosen[1:])
    )
    return value


def coverage_dp(
    weights: ArrayLike,
    budget: int,
    *,
    coverage_lambda: float = 0.5,
    coverage_gap: int = 4,
) -> tuple[list[int], float]:
    """Exactly maximize OT mass plus saturated temporal coverage in O(Mk).

    The implementation uses the equivalent penalty form
    ``sum(w) - lambda * sum(max(0, gap - distance))``.  For fixed ``budget``
    this differs from the paper objective by a constant only, so it has exactly
    the same optimizer.
    """

    scores = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("weights must be a finite one-dimensional array")
    n = len(scores)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if coverage_lambda < 0 or coverage_gap <= 0:
        raise ValueError("invalid coverage parameters")
    if budget >= n:
        selected = list(range(n))
        return selected, coverage_objective(selected, scores, coverage_lambda, coverage_gap)

    neg_inf = -np.inf
    previous = scores.copy()
    backpointer = np.full((budget + 1, n), -1, dtype=np.int64)

    from collections import deque

    for count in range(2, budget + 1):
        current = np.full(n, neg_inf, dtype=np.float64)
        prefix_value = np.full(n, neg_inf, dtype=np.float64)
        prefix_index = np.full(n, -1, dtype=np.int64)
        best_value, best_index = neg_inf, -1
        for j in range(n):
            if j >= count - 2 and previous[j] > best_value:
                best_value, best_index = previous[j], j
            prefix_value[j], prefix_index[j] = best_value, best_index

        near: deque[int] = deque()
        for i in range(count - 1, n):
            new_j = i - 1
            if new_j >= count - 2:
                new_value = previous[new_j] - coverage_lambda * new_j
                while near and previous[near[-1]] - coverage_lambda * near[-1] <= new_value:
                    near.pop()
                near.append(new_j)

            left = max(count - 2, i - coverage_gap + 1)
            while near and near[0] < left:
                near.popleft()

            candidate, predecessor = neg_inf, -1
            far_right = i - coverage_gap
            if far_right >= count - 2:
                far_j = int(prefix_index[far_right])
                if far_j >= 0:
                    candidate, predecessor = previous[far_j], far_j
            if near:
                near_j = near[0]
                near_value = previous[near_j] - coverage_lambda * (coverage_gap - (i - near_j))
                if near_value > candidate:
                    candidate, predecessor = near_value, near_j
            if predecessor >= 0:
                current[i] = scores[i] + candidate
                backpointer[count, i] = predecessor
        previous = current

    last = int(np.argmax(previous[budget - 1 :])) + budget - 1
    selected: list[int] = []
    cursor = last
    for count in range(budget, 0, -1):
        selected.append(cursor)
        if count > 1:
            cursor = int(backpointer[count, cursor])
            if cursor < 0:
                raise RuntimeError("DP backtracking failed")
    selected.reverse()
    return selected, coverage_objective(selected, scores, coverage_lambda, coverage_gap)


def otfs_select(
    evidence: ArrayLike,
    frame_indices: Sequence[int] | None = None,
    config: OTFSConfig | None = None,
    *,
    evidence_is_prepared: bool = False,
) -> OTFSResult:
    """Run OTFS-Sinkhorn and exact coverage-aware subset extraction."""

    cfg = config or OTFSConfig()
    raw = np.asarray(evidence, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("evidence must have shape [atoms, frames]")
    cfg.validate(raw.shape[0], raw.shape[1])
    normalized = raw.copy() if evidence_is_prepared else minmax_normalize(raw)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized evidence is not finite")
    cost = 1.0 - normalized
    plan, iterations_run = semi_unbalanced_sinkhorn(
        cost,
        cfg.source_prior,
        epsilon=cfg.epsilon,
        gamma=cfg.gamma,
        iterations=cfg.iterations,
        tolerance=cfg.tolerance,
    )
    weights = plan.sum(axis=0)
    selected, _ = coverage_dp(
        weights,
        cfg.budget,
        coverage_lambda=cfg.coverage_lambda,
        coverage_gap=cfg.coverage_gap,
    )
    original = list(range(raw.shape[1])) if frame_indices is None else [int(x) for x in frame_indices]
    if len(original) != raw.shape[1]:
        raise ValueError("frame_indices length does not match evidence")
    return OTFSResult(
        selected_positions=selected,
        selected_frame_indices=[original[i] for i in selected],
        frame_weights=weights,
        transport_plan=plan,
        normalized_evidence=normalized,
        iterations_run=iterations_run,
    )
