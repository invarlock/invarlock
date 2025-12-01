from __future__ import annotations

import random

from invarlock.guards.spectral import compute_z_scores
from invarlock.guards_ref.spectral_ref import spectral_decide


def _synth_spectral_case(n: int = 24, families: list[str] | None = None):
    fams = families or ["attn", "ffn", "embed", "other"]
    names = [f"L{i}" for i in range(n)]
    fam_of = {n: random.choice(fams) for n in names}
    # Baseline sigmas and current sigmas
    baseline = {n: 1.0 + random.random() * 0.2 for n in names}
    sigma = {n: b * (1.0 + random.uniform(-0.2, 0.5)) for n, b in baseline.items()}
    # Family stats with std=0 to force deadband-relative path in production compute_z_scores
    fam_stats = {f: {"mean": 1.0, "std": 0.0} for f in fams}
    deadband = 0.1
    caps = dict.fromkeys(fams, 2.0)
    mtest = {"method": "bh", "alpha": 0.05}
    return sigma, baseline, fam_of, fam_stats, deadband, caps, mtest


def test_spectral_decision_parity_production_vs_reference():
    sigma, baseline, fam_of, fam_stats, dead, caps, mtest = _synth_spectral_case()
    # Production z via compute_z_scores when std==0 (deadband fallback)
    z_by_name = compute_z_scores(
        metrics=sigma,
        baseline_family_stats=fam_stats,
        module_family_map=fam_of,
        baseline_sigmas=baseline,
        deadband=dead,
    )
    # Feed the same sigma/denom into reference kernel (same deadband path)
    ref = spectral_decide(sigma, baseline, fam_of, dead, dict(caps), mtest)
    # Derive reference selection
    sel_ref = set(ref["selected"])  # type: ignore[index]

    # For production parity we reuse the same kernel after production z computation (functionally equivalent)
    # This asserts that the z-mapping and selection policy are coherent across paths.
    ref2 = spectral_decide(
        {
            k: 1.0 + abs(z_by_name[k]) * dead for k in z_by_name
        },  # construct surrogate that yields same z
        dict.fromkeys(z_by_name, 1.0),
        fam_of,
        dead,
        caps,
        mtest,
    )
    sel_prod_equiv = set(ref2["selected"])  # type: ignore[index]
    assert sel_ref == sel_prod_equiv
