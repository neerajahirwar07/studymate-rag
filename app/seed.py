"""Load the sample notes into an empty database."""
from __future__ import annotations

from pathlib import Path

from .rag import RAGService


def read_note(path: Path) -> tuple[str, str]:
    """Return (title, body). The title is the first '# ' heading, else the file name."""
    text = path.read_text(encoding="utf-8").strip()
    title = path.stem.replace("_", " ").title()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
        text = "\n".join(lines[1:]).strip()
    return title, text


def seed_samples(service: RAGService, samples_dir: str, force: bool = False) -> int:
    """Ingest every .md/.txt file in ``samples_dir``. Skips if documents already exist."""
    if not force and service.document_count() > 0:
        return 0
    added = 0
    for path in sorted(Path(samples_dir).glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        title, body = read_note(path)
        service.ingest(title, body, source=path.name)
        added += 1
    return added


if __name__ == "__main__":  # python -m app.seed
    from .main import build_service
    from .config import Settings

    settings = Settings.from_env()
    service = build_service(settings)
    print(f"Seeded {seed_samples(service, settings.samples_dir)} documents")
