"""Experiment project scaffolding CLI."""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import textwrap
import urllib.request
import zipfile
from collections.abc import Iterable
from pathlib import Path

TEMPLATE_REPO = "poulet-lab/experiment-template"
TEMPLATE_BRANCH = "main"
USAGE = "Usage: build-exp init <name>"


def _ensure_single_directory(root: Path) -> Path:
    directories = [path for path in root.iterdir() if path.is_dir()]
    if len(directories) != 1:
        raise RuntimeError(
            "Expected archive to contain a single top-level directory but found "
            f"{len(directories)} entries."
        )
    return directories[0]


def download_template(dest: Path, repo: str = TEMPLATE_REPO, branch: str = TEMPLATE_BRANCH) -> Path:
    """Download and unpack the experiment template archive.

    The GitHub repository and branch can be overridden for testing or alternative templates.
    The extracted template root directory path is returned.
    """

    zip_url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
    with urllib.request.urlopen(zip_url) as response:
        archive = response.read()

    with zipfile.ZipFile(io.BytesIO(archive)) as archive_zip:
        archive_zip.extractall(dest)

    return _ensure_single_directory(dest)


def _update_readme(target: Path, name: str) -> None:
    readme = target / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        text = text.replace("experiment-template", name)
        readme.write_text(text, encoding="utf-8")


def _remove_git_directory(target: Path) -> None:
    git_dir = target / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)


def create_experiment(
    name: str, *, repo: str = TEMPLATE_REPO, branch: str = TEMPLATE_BRANCH
) -> None:
    """Create a new experiment project using the template repository."""

    target = Path(name).resolve()
    if target.exists():
        print(f"Error: target directory {target} already exists.", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        template_root = download_template(tmpdir, repo=repo, branch=branch)
        target.parent.mkdir(parents=True, exist_ok=True)
        _copy_directory(template_root.iterdir(), target)

    _remove_git_directory(target)
    _update_readme(target, target.name)

    print(f"✅ Experiment created in {target}")
    print(
        textwrap.dedent(
            f"""
            Next steps:
              cd {target}
              git init
              git add .
              git commit -m \"Initial commit (from poulet-py experiment init)\"
            """
        ).strip()
    )


def _copy_directory(entries: Iterable[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for entry in entries:
        target_path = destination / entry.name
        if entry.is_dir():
            _copy_directory(entry.iterdir(), target_path)
        else:
            target_path.write_bytes(entry.read_bytes())


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        sys.exit(0)

    if argv[0] != "init" or len(argv) < 2:
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    name = argv[1]
    create_experiment(name)


if __name__ == "__main__":
    main()
