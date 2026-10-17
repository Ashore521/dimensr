# Publishing this repository

The repository contains original implementation code, documentation, dependency metadata, tests, synthetic benchmark results and upstream notices. It excludes local virtual environments, paper PDFs, the private design document, downloaded upstream source and credentials.

Before publishing, run:

```bash
python -m pytest -q
python -m build
git status --short
git ls-files
```

The confirmed publication destination is [Ashore521/dimensr](https://github.com/Ashore521/dimensr), a public repository. For a fresh checkout without a configured remote, use:

```bash
git remote add origin https://github.com/Ashore521/dimensr.git
git push -u origin main
```

If using an existing repository, inspect its contents and history first. Do not overwrite or force-push unrelated work. GitHub Actions should run the included Windows/Linux test matrix after publication; successful local tests do not imply hosted CI has already passed.

The built wheel can be installed directly with `python -m pip install path/to/dimensr-0.1.0-py3-none-any.whl`. Publishing to PyPI is a separate operation and has not been performed. The confirmed repository URL is included in package metadata and CITATION.cff.
