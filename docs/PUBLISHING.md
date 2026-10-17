# Publishing this repository

The repository contains original implementation code, documentation, dependency metadata, tests, synthetic benchmark results and upstream notices. It excludes local virtual environments, paper PDFs, the private design document, downloaded upstream source and credentials.

Before publishing, run:

```bash
python -m pytest -q
python -m build
git status --short
git ls-files
```

Create an empty public GitHub repository under the intended account, then push this directory's `main` branch. For example, after replacing `OWNER` with your account:

```bash
git remote add origin https://github.com/OWNER/dimensr.git
git push -u origin main
```

If using an existing repository, inspect its contents and history first. Do not overwrite or force-push unrelated work. GitHub Actions should run the included Windows/Linux test matrix after publication; successful local tests do not imply hosted CI has already passed.

The built wheel can be installed directly with `python -m pip install path/to/dimensr-0.1.0-py3-none-any.whl`. Publishing to PyPI is a separate operation and has not been performed. Set the repository URL in package metadata after its destination is confirmed.
