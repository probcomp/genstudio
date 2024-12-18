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


def get_next_version(alpha_name=None):
    today = datetime.now()
    year_month = today.strftime("%Y.%m")

    if alpha_name:
        # Read alpha tags for this month and name
        base = f"{year_month}-alpha.{alpha_name}"
        tags = (
            subprocess.check_output(["git", "tag", "-l", f"v{base}-[0-9][0-9][0-9]"])
            .decode()
            .strip()
            .split("\n")
        )

        # Filter out empty strings and remove 'v' prefix
        matching_tags = [tag[1:] for tag in tags if tag]

        if not matching_tags:
            return f"{base}-001"

        # Extract the highest number
        numbers = [int(tag.split("-")[-1]) for tag in matching_tags]
        next_num = max(numbers) + 1
        return f"{base}-{next_num:03d}"
    else:
        # Original version logic for regular releases
        tags = (
            subprocess.check_output(
                ["git", "tag", "-l", f"v{year_month}.[0-9][0-9][0-9]"]
            )
            .decode()
            .strip()
            .split("\n")
        )
        release_tags = [tag[1:] for tag in tags if tag and not tag.endswith(".dev")]

        if not release_tags:
            return f"{year_month}.001"

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
            ["git", "log", f"{last_tag}..HEAD", "--pretty=format:%B"]
        )
        .decode()
        .split("\n\n")
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
        lines = msg.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            categorized = False
            for category, prefix in categories.items():
                if prefix and line.lower().startswith(prefix.lower()):
                    cleaned_msg = line[len(prefix) :].strip()
                    categorized_commits[category].append(cleaned_msg)
                    categorized = True
                    break
            if not categorized:
                categorized_commits["Other Changes"].append(line)
        print("---")

    # Prepare changelog entry
    changelog_entry = (
        f"### [{new_version}] - {datetime.now().strftime('%b %d, %Y')}\n\n"
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

    # Save original content of CHANGELOG.md
    with open("CHANGELOG.md", "r") as f:
        original_content = f.read()

    # Prepend to CHANGELOG.md
    with open("CHANGELOG.md", "w") as f:
        f.write(changelog_entry + original_content)

    # Print the new changelog entry to the terminal
    print("\nNew changelog entry:")
    print(changelog_entry)

    # Pause for user to review and potentially edit
    user_input = input(
        "\nReview the changelog entry. Press Enter to continue or 'q' to cancel: "
    )

    if user_input.lower() == "q":
        # Revert the changelog
        with open("CHANGELOG.md", "w") as f:
            f.write(original_content)
        print("Changelog update cancelled and reverted.")
        return False

    return True


def main():
    # Check for uncommitted changes
    check_working_directory()

    # Add command line argument parsing
    alpha_name = None
    if len(sys.argv) > 2 and sys.argv[1] == "--alpha":
        alpha_name = sys.argv[2]
        skip_changelog = True
    else:
        skip_changelog = False

    new_version = get_next_version(alpha_name)

    # Print version prominently for easy copying
    if alpha_name:
        print("\n" + "=" * 50)
        print(f"Alpha version: {new_version}")
        print("=" * 50 + "\n")

    if not skip_changelog:
        if not update_changelog(new_version):
            print("Release process cancelled.")
            return

    update_pyproject_toml(new_version)
    update_readme(new_version)

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
