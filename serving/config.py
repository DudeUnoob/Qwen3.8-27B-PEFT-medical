"""Read the CLI's shared dotenv configuration without changing the environment."""

from pathlib import Path

from dotenv import dotenv_values


def load_credentials(root: Path) -> dict[str, str]:
    path = root / ".env"
    if not path.is_file():
        raise RuntimeError("Create a local .env from .env.example before using the client")
    values = {key: value for key, value in dotenv_values(path).items() if value is not None}
    if not values.get("MODAL_KEY") or not values.get("MODAL_SECRET"):
        raise RuntimeError("Set MODAL_KEY and MODAL_SECRET in the local .env file")
    return values
