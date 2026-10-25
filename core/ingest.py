"""Repository cloning, text-file discovery, and source snapshot storage."""

import base64
from contextlib import contextmanager
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator

from core.models import SourceFile

logger = logging.getLogger(__name__)

EXTENSIONS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "c_sharp",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "restructuredtext",
    ".xml": "xml",
    ".vue": "vue",
    ".svelte": "svelte",
}
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "node_modules",
    "vendor",
    "dist",
    "build",
    "__pycache__",
    ".next",
    ".idea",
    ".vscode",
    "coverage",
}
MAX_FILE_BYTES = 1_000_000


def sanitize_repository_url(url: str) -> str:
    """Remove embedded URL credentials before persisting or displaying a URL."""
    if "://" in url:
        scheme, remainder = url.split("://", 1)
        if "@" in remainder:
            remainder = remainder.rsplit("@", 1)[1]
        return f"{scheme}://{remainder}"
    return url


def repository_id(url: str, branch: str) -> str:
    """Create a stable, non-secret identifier from a repository URL and branch."""
    safe_url = sanitize_repository_url(url)
    return hashlib.sha256(f"{safe_url.strip().lower()}|{branch.strip()}".encode()).hexdigest()[:16]


def repository_name(url: str) -> str:
    """Derive a display name from an HTTPS or SSH-style Git URL."""
    safe_url = sanitize_repository_url(url)
    return PurePosixPath(safe_url.rstrip("/").removesuffix(".git").replace(":", "/")).name


def _clone_env(token: str | None) -> dict[str, str]:
    env = os.environ.copy()
    if token:
        credentials = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "http.extraheader"
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {credentials}"
    return env


@contextmanager
def clone_repository(
    url: str,
    branch: str = "",
    token: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Iterator[Path]:
    """Clone a repository into a temporary directory that is removed on exit."""
    temp_dir = tempfile.TemporaryDirectory(prefix="repo-explorer-")
    try:
        destination = Path(temp_dir.name) / "checkout"
        if progress_callback:
            progress_callback(10, "Cloning repository into a temporary folder.")
        command = ["git", "clone", "--depth", "1"]
        if branch.strip():
            command.extend(["--branch", branch.strip()])
        command.extend([url.strip(), str(destination)])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300,
            env=_clone_env(token),
            check=False,
        )
        if result.returncode:
            safe_error = result.stderr.replace(url, sanitize_repository_url(url))
            if token:
                safe_error = safe_error.replace(token, "[redacted]")
            raise RuntimeError(f"Git clone failed: {safe_error.strip() or 'unknown git error'}")
        yield destination
    finally:
        temp_dir.cleanup()


def collect_source_files(repo_path: Path) -> list[SourceFile]:
    """Read supported UTF-8 text files while skipping generated and binary files."""
    files: list[SourceFile] = []
    for root, directories, filenames in os.walk(repo_path):
        directories[:] = sorted(name for name in directories if name not in IGNORED_DIRS)
        for filename in sorted(filenames):
            path = Path(root) / filename
            language = EXTENSIONS.get(path.suffix.lower())
            if not language:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    logger.info("Skipping oversized source file: %s", path)
                    continue
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as error:
                logger.warning("Skipping unreadable source file %s: %s", path, error)
                continue
            relative = path.relative_to(repo_path).as_posix()
            files.append(SourceFile(path=relative, content=content, language=language))
    return files


def language_counts(files: list[SourceFile]) -> dict[str, int]:
    """Count source files by detected language."""
    return dict(Counter(source.language for source in files))


def persist_source_files(
    files: list[SourceFile], destination: Path
) -> Path:
    """Write the indexed text files into a new, versioned source snapshot."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = destination.joinpath(*PurePosixPath(source.path).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.content, encoding="utf-8")
    return destination


def remove_snapshot(snapshot: Path, repository_root: Path) -> None:
    """Remove a snapshot only when its resolved path is inside the data directory."""
    root = repository_root.resolve()
    target = snapshot.resolve()
    if target == root or root not in target.parents:
        raise ValueError("Refusing to remove a path outside the repository data directory")
    if target.exists():
        shutil.rmtree(target)
