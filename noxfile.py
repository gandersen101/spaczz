"""Nox sessions."""
import os
import platform
import tempfile
from typing import Any

import nox
from nox.sessions import Session

nox.options.sessions = "lint", "mypy", "tests"

PACKAGE = "spaczz"
LOCATIONS = "src", "tests", "./noxfile.py", "docs/conf.py"
PYTHON = "3.11"
PYTHONS = ["3.11", "3.10", "3.9", "3.8", "3.7"]
MYPY_EXTRAS = ["jinja2", "nox", "numpy", "pytest", "rapidfuzz", "spacy"]


def install_with_constraints(session: Session, *args: str, **kwargs: Any) -> None:
    """Install packages constrained by Poetry's lock file.

    This function is a wrapper for nox.sessions.Session.install. It
    invokes pip to install packages inside of the session's virtualenv.
    Additionally, pip is passed a constraints file generated from
    Poetry's lock file, to ensure that the packages are pinned to the
    versions specified in poetry.lock. This allows you to manage the
    packages as Poetry development dependencies.

    Arguments:
        session: The Session object.
        args: Command-line arguments for pip.
        kwargs: Additional keyword arguments for Session.install.
    """
    if platform.system() == "Windows":
        req_path = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        session.run(
            "poetry",
            "export",
            "--dev",
            "--without-hashes",
            "--format=requirements.txt",
            f"--output={req_path}",
            external=True,
        )
        session.install(f"--constraint={req_path}", *args, **kwargs)
        os.unlink(req_path)
    else:
        with tempfile.NamedTemporaryFile() as requirements:
            session.run(
                "poetry",
                "export",
                "--dev",
                "--format=requirements.txt",
                "--without-hashes",
                f"--output={requirements.name}",
                external=True,
            )
            session.install(f"--constraint={requirements.name}", *args, **kwargs)


@nox.session(python=PYTHON)
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python=PYTHON)
def coverage(session: Session) -> None:
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml")
    session.run("codecov", *session.posargs)


@nox.session(python=PYTHON)
def docs(session: Session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")


@nox.session(python=PYTHON)
def isort(session: Session) -> None:
    """Run isort import formatter."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "isort")
    session.run("isort", *args)


@nox.session(python=PYTHONS)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or LOCATIONS
    install_with_constraints(
        session,
        "flake8",
        "flake8-annotations",
        "flake8-bandit",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@nox.session(python=PYTHONS)
def mypy(session: Session) -> None:
    """Type-check using mypy."""
    args = session.posargs or LOCATIONS
    install_with_constraints(session, "mypy", *MYPY_EXTRAS)
    session.run("mypy", *args)


@nox.session(python=PYTHON)
def readme(session: Session) -> None:
    """Run the README notebook and convert it to Markdown."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "nbconvert")
    session.run(
        "jupyter",
        "nbconvert",
        "--to=markdown",
        "--execute",
        "notebooks/README.ipynb",
    )
    session.run("mv", "-f", "notebooks/README.md", "README.md", external=True)


@nox.session(python=PYTHONS)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "--cov-append"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
    )
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("pytest", *args)


@nox.session(python=PYTHONS)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "xdoctest")
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("python", "-m", "xdoctest", PACKAGE, *args)
