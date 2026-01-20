from __future__ import annotations

from pathlib import Path


def test_certify_edit_config_includes_guard_order(tmp_path: Path, monkeypatch):
    """`invarlock certify --edit-config` should not require guards.order in YAML."""
    monkeypatch.chdir(tmp_path)
    preset = tmp_path / "preset.yaml"
    preset.write_text(
        "\n".join(
            [
                "dataset:",
                "  provider: synthetic",
                "  seq_len: 8",
                "  stride: 8",
                "  preview_n: 1",
                "  final_n: 1",
                "eval:",
                "  metric:",
                "    kind: ppl_causal",
                "",
            ]
        ),
        encoding="utf-8",
    )

    edit_cfg = tmp_path / "edit.yaml"
    edit_cfg.write_text(
        "\n".join(
            [
                "edit:",
                "  name: quant_rtn",
                "  plan:",
                "    bitwidth: 8",
                "    per_channel: true",
                "    group_size: 128",
                "    clamp_ratio: 0.005",
                "    scope: attn",
                "",
            ]
        ),
        encoding="utf-8",
    )

    def _fake_run_command(*, config: str, out: str, **kwargs):  # noqa: ARG001
        cfg_path = Path(config)
        import yaml

        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if cfg_path.name == "edited_merged.yaml":
            guards = payload.get("guards")
            assert isinstance(guards, dict)
            order = guards.get("order")
            assert (
                isinstance(order, list)
                and order
                and all(isinstance(item, str) for item in order)
            )

        # Create a minimal report.json so certify can locate it via _latest_run_report().
        out_dir = Path(out) / "000000"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text("{}", encoding="utf-8")

    from invarlock.cli.commands import certify as certify_mod
    from invarlock.cli.commands import run as run_mod

    monkeypatch.setattr(run_mod, "run_command", _fake_run_command)
    monkeypatch.setattr(certify_mod, "_report", lambda **_: None)

    # Should not raise; the edited merged config must include guards.order.
    certify_mod.certify_command(
        source="gpt2",
        edited="gpt2",
        adapter="hf_causal",
        profile="dev",
        tier="balanced",
        preset=str(preset),
        out=str(tmp_path / "runs"),
        cert_out=str(tmp_path / "reports"),
        edit_config=str(edit_cfg),
        banner=False,
        quiet=True,
    )
