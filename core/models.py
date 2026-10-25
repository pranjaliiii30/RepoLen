"""Shared data models for repository indexing."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A code or documentation fragment with source coordinates."""

    path: str
    symbol: str
    start_line: int
    end_line: int
    content: str
    language: str


@dataclass
class RepoIndex:
    """Persistent metadata describing one indexed repository snapshot."""

    repo_id: str
    name: str
    url: str
    branch: str
    collection_name: str
    source_dir: str
    indexed_at: str
    file_count: int
    line_count: int
    chunk_count: int
    language_counts: dict[str, int] = field(default_factory=dict)
    active: bool = True
    overview: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable repository metadata."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "RepoIndex":
        """Build a model from registry data while applying old-version defaults."""
        return cls(
            repo_id=str(values["repo_id"]),
            name=str(values.get("name", values["repo_id"])),
            url=str(values["url"]),
            branch=str(values.get("branch", "HEAD")),
            collection_name=str(values["collection_name"]),
            source_dir=str(values["source_dir"]),
            indexed_at=str(values.get("indexed_at", "")),
            file_count=int(values.get("file_count", 0)),
            line_count=int(values.get("line_count", 0)),
            chunk_count=int(values.get("chunk_count", 0)),
            language_counts={
                str(key): int(value)
                for key, value in values.get("language_counts", {}).items()
            },
            active=bool(values.get("active", True)),
            overview=str(values.get("overview", "")),
        )


@dataclass(frozen=True)
class SourceFile:
    """A text file discovered in a repository checkout."""

    path: str
    content: str
    language: str


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk and its hybrid rank and vector similarity."""

    chunk: Chunk
    score: float
    similarity: float
