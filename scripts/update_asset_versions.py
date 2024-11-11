from pathlib import Path
import sys

REPLACEMENTS = {
    "mkdocs.yml": {"stylesheets/custom.css": "stylesheets/custom.css?v={VERSION}"},
}


def update_asset_versions(version: str) -> None:
    """Update asset version query parameters in widget.py and mkdocs.yml.

    Args:
        version: Version string to append to asset URLs
    """
    for file_name, replacements in REPLACEMENTS.items():
        file_path = Path(file_name)
        content = file_path.read_text()
        for old, new in replacements.items():
            content = content.replace(old, new.format(VERSION=version))
        file_path.write_text(content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_asset_versions.py VERSION")
        sys.exit(1)

    update_asset_versions(sys.argv[1])
