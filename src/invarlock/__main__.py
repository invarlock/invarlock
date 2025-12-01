from .cli.app import app


def main() -> None:
    """Entry point for `python -m invarlock`."""
    app()


if __name__ == "__main__":
    main()
