from pathlib import Path


DEFAULT_VALID_SUFFIXES = [".jpg", ".jpeg", ".png"]


def collect_images(root: Path, valid_suffixes: list[str] | None = None) -> list[Path]:
    if not root.exists():
        return []
    suffixes = DEFAULT_VALID_SUFFIXES if valid_suffixes is None else valid_suffixes
    image_paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    ]
    image_paths.sort()
    return image_paths
