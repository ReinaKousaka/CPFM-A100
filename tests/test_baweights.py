"""BAWeights bound-projection regression test (review 2026-08-06).

The old clip-then-renormalize produced max weight 0.758 > hi = c/L = 0.5556 on
a one-tiny-norm profile. The bisection projection must keep every weight inside
[1/(cL), c/L] (tol 1e-9), sum to 1, and reduce to uniform when norms are equal.
Run: .venv/bin/python -m tests.test_baweights
"""

import sys

sys.path.insert(0, ".")
from ba_pfm.perceptual import BAWeights, BLOCKS  # noqa: E402


def run():
    L = len(BLOCKS)
    lo, hi = 1.0 / (5.0 * L), 5.0 / L

    # regression case: one block with a tiny grad norm dominates inverse weights
    # (raw normalized weights [0.9, 0.0125 x 8] -> old code returned 0.758 > hi)
    baw = BAWeights()
    norms = {b: 72.0 for b in BLOCKS}
    norms[BLOCKS[0]] = 1.0
    for _ in range(200):  # saturate the EMA
        baw.update(norms)
    w, n_eff = baw.weights()
    assert abs(sum(w.values()) - 1.0) < 1e-9, sum(w.values())
    for b, v in w.items():
        assert lo - 1e-9 <= v <= hi + 1e-9, (b, v, lo, hi)
    assert w[BLOCKS[0]] == max(w.values())
    assert 1.0 <= n_eff <= L

    # random profiles: bounds + simplex always hold
    import random
    rng = random.Random(0)
    for trial in range(500):
        baw = BAWeights()
        baw.update({b: 10 ** rng.uniform(-4, 4) for b in BLOCKS})
        w, _ = baw.weights()
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(lo - 1e-9 <= v <= hi + 1e-9 for v in w.values()), (trial, w)

    # equal norms -> uniform
    baw = BAWeights()
    baw.update({b: 3.14 for b in BLOCKS})
    w, n_eff = baw.weights()
    assert all(abs(v - 1.0 / L) < 1e-9 for v in w.values())
    assert abs(n_eff - L) < 1e-6
    print("test_baweights: ALL PASS")


if __name__ == "__main__":
    run()
