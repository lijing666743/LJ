"""Student-facing entry point for the U2U-MEC experiment launcher."""

from src.cli import main as _cli_main


def main() -> int:
    """Run interactive or direct CLI configuration through the common runner."""

    return _cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
