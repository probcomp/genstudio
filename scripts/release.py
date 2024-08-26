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

    # Read all tags, including those with 'v' prefix
    tags = (
        subprocess.check_output(["git", "tag", "-l", f"v{year_month}.[0-9][0-9][0-9]"])
        .decode()
        .strip()
        .split("\n")
    )

    # Filter out dev versions and empty strings, and remove 'v' prefix
    release_tags = [tag[1:] for tag in tags if tag and not tag.endswith(".dev")]

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
        r"Current version: `.+`", f"Current version: `{new_version}`", content
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
    commit_messages = (
        subprocess.check_output(
            ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%s"]
        )
        .decode()
        .split("\n")
    )
    # Define categories and their prefixes
    categories = {
        "New Features": "feat:",
        "Bug Fixes": "fix:",
        "Documentation": "docs:",
        "Other Changes": None,  # This will catch all other commits
    }

    # Categorize commits
    categorized_commits = {category: [] for category in categories}

    for msg in commit_messages:
        categorized = False
        for category, prefix in categories.items():
            if prefix and msg.startswith(prefix):
                categorized_commits[category].append(msg[len(prefix) :].strip())
                categorized = True
                break
        if not categorized:
            categorized_commits["Other Changes"].append(msg.strip())

    # Prepare changelog entry
    changelog_entry = (
        f"#### [{new_version}] - {datetime.now().strftime('%B %d, %Y')}\n\n"
    )

    for category, commits in categorized_commits.items():
        if commits:
            changelog_entry += f"#### {category}\n"
            changelog_entry += "\n".join(f"- {commit}" for commit in commits)
            changelog_entry += "\n\n"

    # Remove empty "Other Changes" section if it's the only one
    if (
        len([c for c in categorized_commits.values() if c]) == 1
        and categorized_commits["Other Changes"]
    ):
        changelog_entry = changelog_entry.replace("#### Other Changes\n", "")

    # Prepend to CHANGELOG.md
    with open("CHANGELOG.md", "r+") as f:
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

    # Add changes
    subprocess.run(["git", "add", "pyproject.toml", "README.md", "CHANGELOG.md"])

    # Run pre-commit
    subprocess.run(["pre-commit", "run", "--all-files"])

    # Add changes again (in case pre-commit made modifications)
    subprocess.run(["git", "add", "pyproject.toml", "README.md", "CHANGELOG.md"])

    # Commit changes
    subprocess.run(["git", "commit", "-m", f"Release version {new_version}"])

    # Create and push tag
    subprocess.run(
        ["git", "tag", "-a", f"v{new_version}", "-m", f"Release version {new_version}"]
    )
    subprocess.run(["git", "push", "origin", "main", "--tags"])

    print(f"Released version {new_version}")


if __name__ == "__main__":
    main()
