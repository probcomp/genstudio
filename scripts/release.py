import re
from datetime import datetime
import subprocess
import toml
import sys


def check_working_directory():
    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    )
    if result.stdout:
        print("Error: There are uncommitted changes in the working directory.")
        print("Please commit or stash these changes before running the release script.")
        sys.exit(1)


def get_next_version():
    today = datetime.now()
    year_month = today.strftime("%Y.%m")

    # Read all tags
    tags = (
        subprocess.check_output(["git", "tag", "-l", f"{year_month}.*"])
        .decode()
        .split()
    )

    # Filter out dev versions
    release_tags = [tag for tag in tags if not tag.endswith(".dev")]

    if not release_tags:
        return f"{year_month}.001"

    # Extract the highest patch number
    patch_numbers = [int(tag.split(".")[-1]) for tag in release_tags]
    next_patch = max(patch_numbers) + 1

    return f"{year_month}.{next_patch:03d}"


def update_pyproject_toml(new_version):
    with open("pyproject.toml", "r") as f:
        data = toml.load(f)

    data["tool"]["poetry"]["version"] = new_version

    with open("pyproject.toml", "w") as f:
        toml.dump(data, f)

    print(f"Updated pyproject.toml with new version: {new_version}")


def update_readme(new_version):
    with open("README.md", "r") as f:
        content = f.read()

    # Update version in README (assuming there's a line like "Current version: X.Y.Z")
    updated_content = re.sub(
        r"Current version: [\d\.]+", f"Current version: {new_version}", content
    )

    with open("README.md", "w") as f:
        f.write(updated_content)


def update_changelog(new_version):
    # Get commit messages since last tag
    last_tag = (
        subprocess.check_output(["git", "describe", "--tags", "--abbrev=0"])
        .decode()
        .strip()
    )
    commit_messages = subprocess.check_output(
        ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%s"]
    ).decode()

    # Categorize commits (this is a basic implementation and might need refinement)
    features = [msg for msg in commit_messages.split("\n") if msg.startswith("feat:")]
    fixes = [msg for msg in commit_messages.split("\n") if msg.startswith("fix:")]
    others = [
        msg
        for msg in commit_messages.split("\n")
        if not (msg.startswith("feat:") or msg.startswith("fix:"))
    ]

    # Prepare changelog entry
    changelog_entry = f"## [{new_version}] - {datetime.now().strftime('%B %d, %Y')}\n\n"

    if features:
        changelog_entry += "### New Features\n"
        changelog_entry += "\n".join(f"- {feature}" for feature in features)
        changelog_entry += "\n\n"

    if fixes:
        changelog_entry += "### Bug Fixes\n"
        changelog_entry += "\n".join(f"- {fix}" for fix in fixes)
        changelog_entry += "\n\n"

    if others:
        changelog_entry += "### Other Changes\n"
        changelog_entry += "\n".join(f"- {other}" for other in others)
        changelog_entry += "\n\n"

    # Prepend to CHANGELOG.md
    with open("docs/CHANGELOG.md", "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(changelog_entry + content)

    # Print the new changelog entry to the terminal
    print("\nNew changelog entry:")
    print(changelog_entry)

    # Pause for user to review and potentially edit
    input(
        "\nReview the changelog entry. Edit CHANGELOG.md if needed, then press Enter to continue..."
    )


def main():
    # Check for uncommitted changes
    check_working_directory()

    new_version = get_next_version()

    update_pyproject_toml(new_version)
    update_readme(new_version)
    update_changelog(new_version)

    # Commit changes
    subprocess.run(["git", "add", "pyproject.toml", "README.md", "docs/CHANGELOG.md"])
    subprocess.run(["git", "commit", "-m", f"Release version {new_version}"])

    # Create and push tag
    subprocess.run(
        ["git", "tag", "-a", f"v{new_version}", "-m", f"Release version {new_version}"]
    )
    subprocess.run(["git", "push", "origin", "main", "--tags"])

    print(f"Released version {new_version}")


if __name__ == "__main__":
    main()
