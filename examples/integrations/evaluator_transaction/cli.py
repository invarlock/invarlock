"""Command-line entrypoint for the maintained evaluator transaction."""

try:
    from examples.integrations.evaluator_transaction.transaction import main
except ModuleNotFoundError as exc:  # pragma: no cover - container module execution
    if not exc.name or not exc.name.startswith("examples"):
        raise
    from evaluator_transaction.transaction import main


if __name__ == "__main__":
    raise SystemExit(main())
