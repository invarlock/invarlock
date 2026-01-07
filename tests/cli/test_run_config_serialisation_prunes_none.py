from invarlock.cli.commands import run
from invarlock.cli.config import load_config


def test_run_config_guard_overrides_prune_none(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
guards:
  variance:
    calibration:
      windows: 6
      seed: 42
      min_coverage: 4
""".lstrip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    variance_cfg = cfg.guards.variance

    as_dict = run._to_serialisable_dict(variance_cfg)
    assert as_dict["mode"] == "ci"
    assert as_dict["calibration"]["windows"] == 6

    # These fields exist on the dataclass with default None values, but they must not
    # be passed through as explicit policy overrides (they would clobber tier defaults).
    for key in (
        "clamp",
        "deadband",
        "min_gain",
        "min_rel_gain",
        "min_abs_adjust",
        "max_scale_step",
        "min_effect_lognll",
        "predictive_one_sided",
        "topk_backstop",
        "max_adjusted_modules",
        "predictive_gate",
        "target_modules",
        "scope",
    ):
        assert key not in as_dict

