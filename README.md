# Gen Studio
_Visualization tools for GenJAX._

Current version: `2024.11.007`

-----

`genstudio.plot` provides a composable way to create interactive plots using [Observable Plot](https://observablehq.com/plot/).

Key features:

- Functional, composable plot creation built on Observable Plot (with near 1:1 API correspondence between Python and JavaScript)
- Support for sliders & animations
- Works in Jupyter / Google Colab
- HTML mode which persists plots across kernel restart/shutdown, and a Widget mode which supports Python<>JavaScript interactivity
- Terse layout syntax for organizing plots into rows and columns
- Hiccup implementation for interspersing arbitrary HTML

For detailed usage instructions and examples, refer to the [Gen Studio User Guide](https://studio.gen.dev).

## Installation

GenStudio is currently private. To configure your machine to access the package,

- Run `\invite-genjax <google-account-email>` in any channel in the the probcomp Slack, or [file a ticket requesting access to the GenJAX-Users
group](https://github.com/probcomp/genjax/issues/new?assignees=sritchie&projects=&template=access.md&title=%5BACCESS%5D)
- [install the Google Cloud command line tools](https://cloud.google.com/sdk/docs/install)
- follow the instructions on the [installation page](https://cloud.google.com/sdk/docs/install)
- run `gcloud auth application-default login` as described [in this guide](https://cloud.google.com/sdk/docs/initializing).

To install GenStudio using `pip`:```bash
pip install keyring keyrings.google-artifactregistry-auth
pip install genstudio --extra-index-url https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
```

If you're using Poetry:

```bash
poetry self update && poetry self add keyrings.google-artifactregistry-auth
poetry source add --priority=explicit gcp https://us-west1-python.pkg.dev/probcomp-caliban/probcomp/simple/
poetry add genstudio --source gcp
```

## Development

Run `yarn watch` to compile the JavaScript bundle.

## Credits

- [AnyWidget](https://github.com/manzt/anywidget) provides a nice Python<>JavaScript widget API
- [pyobsplot](https://github.com/juba/pyobsplot) was the inspiration for our Python->JavaScript approach
