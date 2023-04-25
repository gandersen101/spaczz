"""Nox sessions."""
from itertools import product

import nox
from nox.sessions import Session

nox.options.sessions = "lint", "mypy", "test"

PACKAGE = "spaczz"
LOCATIONS = "src", "tests", "./noxfile.py", "docs/conf.py"
PYTHON = "3.11"
PYTHONS = ["3.11", "3.10", "3.9", "3.8", "3.7"]
SPACY_VERIONS = {"3.5.2": PYTHONS, "3.0.9": ["3.10", "3.9", "3.8", "3.7"]}
RAPIDFUZZ_VERSIONS = {
    "3.0.0": PYTHONS,
    "2.15.1": PYTHONS,
    "1.9.1": ["3.10", "3.9", "3.8", "3.7"],
}


# Formatting


@nox.session(python=PYTHON)
def isort(session: Session) -> None:
    """Run isort import formatter."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--only", "isort", external=True)
    session.run("isort", *args)


@nox.session(python=PYTHON)
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--only", "black", external=True)
    session.run("black", *args)


# Linting


@nox.session(python=PYTHONS)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--only", "lint", external=True)
    session.run("flake8", *args)


# Typechecking


@nox.session(python=PYTHONS)
@nox.parametrize("spacy_version", list(SPACY_VERIONS.keys()))
def mypy(session: Session, spacy_version: str) -> None:
    """Type-check using mypy."""
    if session.python not in SPACY_VERIONS[spacy_version]:
        return None

    args = session.posargs or LOCATIONS
    session.run("poetry", "install", "--only", "main,mypy", external=True)
    session.run("mypy", *args)


# Testing


@nox.session(python=PYTHONS)
@nox.parametrize(
    ["spacy_version", "rapidfuzz_version"],
    list(product(SPACY_VERIONS.keys(), RAPIDFUZZ_VERSIONS.keys())),
)
def test(session: Session, spacy_version: str, rapidfuzz_version: str) -> None:
    """Run the test suite."""
    if (
        session.python not in SPACY_VERIONS[spacy_version]
        or session.python not in RAPIDFUZZ_VERSIONS[rapidfuzz_version]
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
        f"spacy=={spacy_version}",
        f"rapidfuzz=={rapidfuzz_version}",
    )
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("pytest", *args)


@nox.session(python=PYTHON)
def coverage(session: Session) -> None:
    """Upload coverage data."""
    session.run("poetry", "install", "--only", "test", external=True)
    session.run("coverage", "xml")


# Docs


@nox.session(python=PYTHONS)
def xdoctest(session: Session) -> None:
    """Run examples with xdoctest."""
    args = session.posargs or ["all"]
    session.run("poetry", "install", "--only", "main,xdoctest", external=True)
    session.run("python", "-m", "spacy", "download", "en_core_web_md")
    session.run("python", "-m", "xdoctest", PACKAGE, *args)


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
