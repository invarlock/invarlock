import importlib


def test_run_module_imports_and_exposes_run_command():
    mod = importlib.import_module("invarlock.cli.commands.run")
    assert hasattr(mod, "run_command"), "run_command symbol must be present"
