import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Initialize Anthropic client
client = anthropic.Anthropic()


def path_in_project(relative_path):
    """
    Resolve a path relative to the genstudio root directory.
    Walks up parent directories until finding 'genstudio', then resolves path from there.
    """
    current = Path(__file__).resolve().parent
    root_dirs = ["genstudio", "src"]  # Look for either genstudio or src directory
    while True:
        if current.name in root_dirs:
            break
        if current == current.parent:
            raise ValueError(
                "Could not find 'genstudio' or 'src' directory in parent path"
            )
        current = current.parent
    return current / relative_path


def count_tokens_in_file(file_path):
    """Count tokens in a single file using Anthropic's tokenizer"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        # Use the beta API's count_tokens method
        return client.beta.messages.count_tokens(
            messages=[{"role": "user", "content": content}],
            model="claude-3-5-sonnet-latest",
        )


print(count_tokens_in_file(path_in_project("docs/llms.py")))
