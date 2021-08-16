"""Nox sessions."""
import os
import platform
import tempfile
from typing import Any

import nox
from nox.sessions import Session


package = "spaczz"
nox.options.sessions = "lint", "mypy", "safety", "tests"
locations = "src", "tests", "noxfile.py", "docs/conf.py"
min_cov = 98
current_spacy = "3.1.1"


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
    spacy_version = kwargs.pop("spacy_version", current_spacy)
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
        if spacy_version < "3.0.0":
            session.install("--upgrade", f"spacy=={spacy_version}")
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
            if spacy_version < "3.0.0":
                session.install("--upgrade", f"spacy=={spacy_version}")


@nox.session(python="3.9")
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    install_with_constraints(session, "black")
    session.run("black", *args)


@nox.session(python="3.9")
def coverage(session: Session) -> None:
    """Upload coverage data."""
    install_with_constraints(session, "coverage[toml]", "codecov")
    session.run("coverage", "xml", f"--fail-under={min_cov}")
    session.run("codecov", *session.posargs)


@nox.session(python="3.9")
def docs(session: Session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "sphinx", "sphinx-autodoc-typehints")
    session.run("sphinx-build", "docs", "docs/_build")


@nox.session(python=["3.9", "3.8", "3.7"])
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
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


@nox.session(python=["3.9", "3.8", "3.7"])
@nox.parametrize("spacy", [current_spacy, "2.3.5"])
def mypy(session: Session, spacy: str) -> None:
    """Type-check using mypy."""
    args = session.posargs or locations
    install_with_constraints(session, "mypy", "pytest", "nox", spacy_version=spacy)
    session.run("mypy", *args)


@nox.session(python="3.9")
def safety(session: Session) -> None:
    """Scan dependencies for insecure packages."""
    if platform.system() == "Windows":
        req_path = os.path.join(tempfile.gettempdir(), os.urandom(24).hex())
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={req_path}",
            external=True,
        )
        install_with_constraints(session, "safety")
        session.run(
            "safety",
            "check",
            f"--file={req_path}",
            "--full-report",
        )
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
            install_with_constraints(session, "safety")
            session.run(
                "safety",
                "check",
                f"--file={requirements.name}",
                "--full-report",
            )


@nox.session(python=["3.9", "3.8", "3.7"])
@nox.parametrize("spacy", [current_spacy, "2.3.5"])
def tests(session: Session, spacy: str) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "--cov-append"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(
        session,
        "coverage[toml]",
        "pytest",
        "pytest-cov",
        spacy_version=spacy,
    )
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("pytest", *args)


@nox.session(python=["3.9", "3.8", "3.7"])
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--no-dev", external=True)
    install_with_constraints(session, "xdoctest")
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("python", "-m", "xdoctest", package, *args)
