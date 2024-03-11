"""Nox sessions.

Even though `nox` is part of the full spaczz dev-install, nothing outside of the stdlib
and `nox` itself should be imported in this module. This is because CI won't have any
other dependencies installed at the time of calling `nox`.
"""
from itertools import product

import nox
from nox.sessions import Session

nox.options.sessions = "lint", "mypy", "tests"

PACKAGE = "spaczz"
LOCATIONS = "src", "tests", "./noxfile.py", "docs/conf.py"
PYTHON = "3.11"
PYTHONS = ["3.11", "3.10", "3.9", "3.8", "3.7"]
SPACY_VERSION = "3.7.4"
SPACY_MYPY_VERSIONS = {SPACY_VERSION: PYTHONS, "3.2.6": PYTHONS[1:]}
SPACY_TEST_VERIONS = {SPACY_VERSION: PYTHONS, "3.0.9": PYTHONS[1:]}
RAPIDFUZZ_VERSION = "3.4.0"
RAPIDFUZZ_VERSIONS = {
    "3.6.2": PYTHONS[:-1],
    # rapidfuzz dropped Python 3.7 support at v3.5, but the spaczz
    # locked version is v3.4, which still supports Python 3.7 like spaczz does.
    RAPIDFUZZ_VERSION: PYTHONS,
    "2.15.1": PYTHONS,
    "1.9.1": PYTHONS[1:],
}


# Formatting


@nox.session(python=PYTHON)
def isort(session: Session) -> None:
    """Run isort import formatter."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--no-root", "--only", "isort", external=True)
    session.run("isort", *args)


@nox.session(python=PYTHON)
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--no-root", "--only", "black", external=True)
    session.run("black", *args)


# Linting


@nox.session(python=PYTHONS)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--no-root", "--only", "lint", external=True)
    session.run("flake8", *args)


# Typechecking


@nox.session(python=PYTHONS)
@nox.parametrize("spacy", list(SPACY_MYPY_VERSIONS.keys()))
def mypy(session: Session, spacy: str) -> None:
    """Type-check using mypy."""
    if session.python not in SPACY_MYPY_VERSIONS[spacy]:
        return None

    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--only", "main,mypy", external=True)
    session.run("python", "-m", "pip", "install", "--upgrade", f"spacy=={spacy}")
    session.run("mypy", *args)


# Testing


@nox.session(python=PYTHONS)
@nox.parametrize(
    ["spacy", "rapidfuzz"],
    list(product(SPACY_TEST_VERIONS.keys(), RAPIDFUZZ_VERSIONS.keys())),
)
def tests(session: Session, spacy: str, rapidfuzz: str) -> None:
    """Run the test suite."""
    if (
        session.python not in SPACY_TEST_VERIONS[spacy]
        or session.python not in RAPIDFUZZ_VERSIONS[rapidfuzz]
    ):
        return None

    args = session.posargs or ["-rxs", "--cov"]
    session.run("poetry", "install", "--only", "main,test", external=True)
    session.run(
        "python",
        "-m",
        "pip",
        "install",
        "--upgrade",
        f"spacy=={spacy}",
        f"rapidfuzz=={rapidfuzz}",
    )
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("pytest", *args)


# Docs


@nox.session(python=PYTHONS)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--only", "main,xdoctest", external=True)
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("xdoctest", PACKAGE, *args)


@nox.session(python=PYTHON)
def readme(session: Session) -> None:
    """Run the README notebook and convert it to Markdown."""
    session.run("poetry", "install", "--only", "main,readme", external=True)
    session.run(
        "jupyter",
        "nbconvert",
        "--to=markdown",
        "--execute",
        "notebooks/README.ipynb",
    )
    session.run("mv", "-f", "notebooks/README.md", "README.md", external=True)


@nox.session(python=PYTHON)
def docs(session: Session) -> None:
    """Build the documentation."""
    session.run("poetry", "install", "--only", "main,docs", external=True)
    session.run("sphinx-build", "docs", "docs/_build")
