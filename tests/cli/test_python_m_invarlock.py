import sys


def test_python_m_invarlock_shows_help(monkeypatch, capsys):
    # Keep imports light during test
    monkeypatch.setenv("INVARLOCK_LIGHT_IMPORT", "1")
    monkeypatch.setenv("INVARLOCK_DISABLE_PLUGIN_DISCOVERY", "1")

    # Simulate: python -m invarlock --help
    from invarlock.__main__ import main

    old_argv = sys.argv[:]
    try:
        sys.argv = ["invarlock", "--help"]
        try:
            main()
        except SystemExit:
            pass
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    # Smoke‑check: hero usage + grouped commands present
    assert "Usage:" in out
    assert "certify" in out and "report" in out and "plugins" in out
