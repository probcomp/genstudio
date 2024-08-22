# GenStudio

GenStudio is a Python library for creating interactive, JavaScript-based visualizations. It provides a simple, composable way to generate plots, animations, and custom user interfaces from within a Python environment.

[Getting Started](getting-started.py){ .md-button  .md-button--primary }

## Key Features

- Seamless integration with [Observable Plot](https://observablehq.com/plot/), a layered grammar-of-graphics based library, for creating rich, interactive plots
- Compose plots, HTML elements, and interactive widgets using a simple, declarative syntax
- Animate plots with built-in support for sliders and frame-by-frame animations
- Embed visualizations in Jupyter notebooks or standalone HTML files
- Customize every aspect of your visualization with JavaScript when needed

### Installation

!!! note
    GenStudio is currently private. To configure your machine to access the package,
    - Run `\invite-genjax <google-account-email>` in any channel in the the probcomp Slack, or [file a ticket requesting access to the GenJAX-Users
    group](https://github.com/probcomp/genjax/issues/new?assignees=sritchie&projects=&template=access.md&title=%5BACCESS%5D)
    - [install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install)
    - follow the instructions on the [installation page](https://cloud.google.com/sdk/docs/install)
    - run `gcloud auth application-default login` as described [in this guide](https://cloud.google.com/sdk/docs/initializing).

If you're using [GenJAX](https://www.github.com/probcomp/genjax) and have already followed the [installation instructions](https://genjax.gen.dev/#quickstart), you can add `genstudio` as an "extra" while installing GenJAX: `genjax[genstudio]`.

To install GenStudio using `pip`:

```bash
pip install keyring keyrings.google-artifactregistry-auth
pip install genstudio --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

If you're using `poetry`:

```bash
poetry self update && poetry self add keyrings.google-artifactregistry-auth
poetry source add --priority=explicit gcp https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
poetry add genstudio --source gcp
```
