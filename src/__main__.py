"""Entry point: ``uv run python -m src <command> [options]``."""
import fire

from .cli import Arguments


if __name__ == "__main__":
    fire.Fire(Arguments)
