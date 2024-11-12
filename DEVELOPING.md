# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the GenStudio codebase.

### Jupyter notes

A typical and recommended workflow is to use genstudio with VS Code's Python Interactive Window. With the VS Code jupyter extension installed, one can use ordinary `.py` files with `# %%` markers to separate cells, then run the `Jupyter: Run Current Cell` command. Results, including plots, will be rendered with VS Code.

Of course, one can also use genstudio from within Jupyter Labs and Colab.

If jupyter has trouble finding a kernel to evaluate from, you can install one (using poetry) via:

```bash
poetry run python -m ipykernel install --user --name genstudio
```

### Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```
