from __future__ import annotations

from hypothesis import strategies as st


def families(min_families: int = 2, max_families: int = 5):
    fams = st.lists(
        st.sampled_from(["attn", "ffn", "embed", "other", "misc"]),
        min_size=min_families,
        max_size=max_families,
        unique=True,
    )
    return fams


def spectral_inputs():
    fams = families()
    return st.builds(
        _build_spectral,
        fams,
        st.integers(min_value=5, max_value=50),
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=1.0, max_value=10.0),
        st.floats(min_value=0.0, max_value=1.0),
    )


def _build_spectral(
    families: list[str], n_names: int, deadband: float, kappa_hi: float, alpha: float
):
    import random

    names = [f"m{i}" for i in range(n_names)]
    family_of = {n: random.choice(families) for n in names}
    sigma = {n: 1.0 + random.random() * 0.5 for n in names}
    denom = dict.fromkeys(names, 1.0)
    caps = dict.fromkeys(families, 1.0 + kappa_hi / 2.0)
    mtest = {"method": "bh", "alpha": float(alpha or 0.05)}
    return sigma, denom, family_of, float(deadband), caps, mtest


def rmt_inputs():
    fams = families()
    return st.builds(
        _build_rmt,
        fams,
        st.integers(min_value=0, max_value=100),
        st.floats(min_value=0.0, max_value=0.5),
    )


def _build_rmt(families: list[str], count_hi: int, eps_hi: float):
    import random

    bare = {f: random.randint(0, count_hi) for f in families}
    guarded = {f: max(0, bare[f] + random.randint(-2, 3)) for f in families}
    eps = dict.fromkeys(families, eps_hi)
    return bare, guarded, eps


def variance_inputs():
    return st.builds(
        _build_var,
        st.floats(min_value=-0.01, max_value=0.01),
        st.floats(min_value=-0.02, max_value=0.0),
        st.floats(min_value=0.0, max_value=0.02),
        st.sampled_from(["lower", "higher"]),
        st.floats(min_value=0.0, max_value=0.02),
        st.booleans(),
    )


def _build_var(
    mu: float, lo: float, hi: float, direction: str, me: float, one_sided: bool
):
    lo2 = min(lo, hi)
    hi2 = max(lo, hi)
    return mu, (lo2, hi2), direction, float(me), bool(one_sided)
