import math

from invarlock.guards import spectral as S


def test_normalize_family_caps_paths():
    # default
    caps = S._normalize_family_caps(None)
    assert isinstance(caps, dict) and "ffn" in caps

    # dict with numeric
    caps2 = S._normalize_family_caps({"ffn": {"kappa": 2.1}, "attn": 3.0})
    assert caps2["ffn"]["kappa"] == 2.1
    assert caps2["attn"]["kappa"] == 3.0

    # invalid → default when default=True
    caps3 = S._normalize_family_caps({"x": object()}, default=True)
    assert "ffn" in caps3

    # invalid → empty when default=False
    caps4 = S._normalize_family_caps({"x": object()}, default=False)
    assert caps4 == {}


def test_summarize_sigmas_and_z_scores():
    sigmas = {"a": 1.0, "b": 2.0, "c": 4.0}
    summary = S._summarize_sigmas(sigmas)
    assert summary["max_spectral_norm"] == 4.0
    assert math.isclose(summary["mean_spectral_norm"], (1.0 + 2.0 + 4.0) / 3.0)
    assert summary["min_spectral_norm"] == 1.0

    # z-score with std>0 path
    family_stats = {"ffn": {"mean": 2.0, "std": 1.0}}
    fam_map = {"a": "ffn", "b": "ffn", "c": "ffn"}
    zs = S.compute_z_scores(sigmas, family_stats, fam_map, sigmas, deadband=0.1)
    assert all(isinstance(v, float) for v in zs.values())

    # z-score fallback path (std=0)
    family_stats2 = {"ffn": {"mean": 2.0, "std": 0.0}}
    zs2 = S.compute_z_scores(sigmas, family_stats2, fam_map, sigmas, deadband=0.1)
    assert all(isinstance(v, float) for v in zs2.values())

    # summarize family z-scores
    fam_summary = S.summarize_family_z_scores(zs2, fam_map, {"ffn": {"kappa": 1.5}})
    assert "ffn" in fam_summary and fam_summary["ffn"]["count"] == 3
