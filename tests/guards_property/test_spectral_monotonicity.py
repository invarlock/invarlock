from __future__ import annotations

from hypothesis import given

from invarlock.guards_ref.spectral_ref import spectral_decide
from tests.guards_property.strategies import spectral_inputs


@given(spectral_inputs())
def test_spectral_monotone_deadband(data):
    sigma, denom, fam_of, dead, caps, mtest = data
    selected0 = spectral_decide(sigma, denom, fam_of, dead, caps, mtest)["selected"]
    selected1 = spectral_decide(sigma, denom, fam_of, dead * 2.0 + 0.01, caps, mtest)[
        "selected"
    ]
    assert len(selected1) <= len(selected0)


@given(spectral_inputs())
def test_spectral_caps_monotone_locality(data):
    sigma, denom, fam_of, dead, caps, mtest = data
    # Increase cap for one family; ensure other family selections do not increase
    fams = set(fam_of.values())
    if not fams:
        return
    target = next(iter(fams))
    sel0 = spectral_decide(sigma, denom, fam_of, dead, caps, mtest)
    caps2 = dict(caps)
    caps2[target] = caps2.get(target, 1.0) + 10.0
    sel1 = spectral_decide(sigma, denom, fam_of, dead, caps2, mtest)
    count0 = sum(1 for n in sel0["selected"] if fam_of[n] != target)
    count1 = sum(1 for n in sel1["selected"] if fam_of[n] != target)
    assert count1 <= count0
