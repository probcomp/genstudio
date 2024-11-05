import ast
import inspect
import os
import importlib
from pathlib import Path

from griffe import GriffeLoader, Parser
from mkdocs.config import Config
from mkdocs.config.config_options import Type
from mkdocs.plugins import BasePlugin
from mkdocs.structure.files import File, Files


class APIDocPlugin(BasePlugin):
    config_scheme = (("modules", Type(list, default=[])),)

    def on_config(self, config: Config):  # pyright: ignore
        # Add source files to watch list
        if not hasattr(config, "watch"):
            setattr(config, "watch", [])

        # Add source files to watch list
        for module_config in self.config.get("modules", []):
            module_name = module_config["module"]
            module = importlib.import_module(module_name)
            source_path = inspect.getfile(module)
            getattr(config, "watch").append(source_path)

        return config

    def on_files(self, files: Files, *, config: Config) -> Files:
        for module_config in self.config.get("modules", []):
            module_name = module_config["module"]
            output_path = module_config.get(
                "output", f"api/{module_name.split('.')[-1]}.md"
            )

            # Import the module
            module = importlib.import_module(module_name)

            # Get paths for source and output files
            source_path = inspect.getfile(module)
            full_output_path = Path(getattr(config, "docs_dir")) / output_path

            # Generate docs if output doesn't exist or source is newer
            should_generate = True
            if full_output_path.exists():
                should_generate = os.path.getmtime(source_path) > os.path.getmtime(
                    full_output_path
                )

            if should_generate:
                # Initialize the loader with Google docstring parser
                loader = GriffeLoader(docstring_parser=Parser.google)

                # Load the module using Griffe
                griffe_module = loader.load(module_name)

                # Generate documentation
                docs_content = self.generate_docs(griffe_module, module)

                # Write to a markdown file
                full_output_path.parent.mkdir(parents=True, exist_ok=True)
                full_output_path.write_text(docs_content)

            # Add the file to MkDocs' file collection using the correct method
            new_file = File(
                path=output_path,
                src_dir=getattr(config, "docs_dir"),
                dest_dir=getattr(config, "site_dir"),
                use_directory_urls=getattr(config, "use_directory_urls"),
            )
            files.remove(new_file)
            files.append(new_file)

        return files

    def generate_docs(self, griffe_module, source_module) -> str:
        content = [f"# {source_module.__name__} {{: .api .api-title }}\n"]

        # Get the ordered members and groups from __all__
        all_defs = get_all_definition_with_comments(source_module)
        if not all_defs:
            return "\n".join(content)

        for item in all_defs:
            if item["type"] == "comment":
                content.extend(self._render_group_header(item["value"]))
            else:
                content.extend(self._render_member(griffe_module, item["value"]))

        return "\n".join(content)

    def _render_group_header(self, group_name: str) -> list[str]:
        return [f"\n{group_name}\n"]

    def _render_member(self, module, member_name: str) -> list[str]:
        if member_name not in module.members:
            return []

        member = module.members[member_name]
        if not member.is_public:
            return []

        content = []
        content.append(f"### {member_name} {{: .api .api-member }}\n")

        if member.docstring:
            parsed = member.docstring.parsed
            content.extend(self._render_docstring_sections(parsed))
        else:
            print(f"Warning: No docstring found for {member_name}")

        if hasattr(member, "signature"):
            content.append(
                f"```python\n{member.signature}\n``` {{: .api .api-signature }}\n"
            )

        content.append("\n")
        return content

    def _render_docstring_sections(self, parsed) -> list[str]:
        content = []

        # Add description if present (first section is usually the description)
        if parsed and len(parsed) > 0 and parsed[0].kind == "text":
            content.append(f"{parsed[0].value}\n")

        for section in parsed:
            if section.kind == "text" and parsed.index(section) != 0:
                content.append(f"\n{section.value}\n")

            elif section.kind.lower() == "parameters":
                content.extend(self._render_parameters_section(section))

            elif section.kind.lower() == "returns":
                content.append("Returns\n{: .api .api-section }\n\n")
                for ret in section.value:
                    ret_type = f" ({ret.annotation})" if ret.annotation else ""
                    content.append(f"- {ret.description}{ret_type}\n")

            elif section.kind.lower() == "raises":
                content.append("Raises\n{: .api .api-section }\n\n")
                for exc in section.value:
                    content.append(f"- `{exc.annotation}`: {exc.description}\n")

            elif section.kind.lower() == "examples":
                content.append("Examples\n{: .api .api-section }\n\n")
                content.append(f"```python\n{section.value}\n```\n")

            elif section.kind.lower() == "notes":
                content.append("Notes\n{: .api .api-section }\n\n")
                content.append(f"{section.value}\n")

            elif section.kind.lower() == "warnings":
                content.append("Warnings\n{: .api .api-section }\n\n")
                content.append(f"{section.value}\n")

        return content

    def _render_parameters_section(self, section) -> list[str]:
        content = []
        content.append("Parameters\n{: .api .api-section }\n\n")
        for param in section.value:
            param_type = f" ({param.annotation})" if param.annotation else ""
            # Handle parameter description, checking if it contains newlines
            desc_lines = param.description.split("\n")
            content.append(f"- `{param.name}`{param_type}: {desc_lines[0]}\n")
            # If there are additional lines, add them indented
            for line in desc_lines[1:]:
                content.append(f"    {line}\n")
        return content


def get_all_definition_with_comments(module):
    # Get the source code of the module
    source = inspect.getsource(module)

    # Parse the source into an AST
    tree = ast.parse(source)

    # Find the __all__ assignment
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        result = []
                        lines = source.split("\n")
                        list_start = node.lineno
                        list_end = node.end_lineno

                        # Process lines within __all__ list
                        for line in lines[list_start - 1 : list_end]:
                            line = line.strip()
                            if line.startswith("#"):
                                # Line is a pure comment
                                result.append(
                                    {"type": "comment", "value": line[1:].strip()}
                                )
                            elif "'" in line or '"' in line:
                                # Extract all members from the line
                                parts = line.split(",")
                                for part in parts:
                                    part = part.strip()
                                    if "'" in part or '"' in part:
                                        # Extract member between quotes
                                        member = (
                                            part.split("'")[1]
                                            if "'" in part
                                            else part.split('"')[1]
                                        )
                                        result.append(
                                            {"type": "member", "value": member}
                                        )

                                # Check for inline comment
                                if "#" in line:
                                    comment = line[line.index("#") + 1 :].strip()
                                    result.append({"type": "comment", "value": comment})
                        return result
    return None


# Example code to parse genstudio.plot.events docstring
if __name__ == "__main__":
    loader = GriffeLoader(docstring_parser=Parser.google)
    module = loader.load("genstudio.plot")

    if "events" in module.members:
        events = module.members["events"]
        if events.docstring:
            parsed = events.docstring.parsed
            print("Docstring sections for genstudio.plot.events:")
            for section in parsed:
                print(f"\nSection kind: {section.kind}")

                if section.kind == "text":
                    print(f"Text content: {section.value}")

                elif section.kind == "parameters":
                    print("Parameters:")
                    for param in section.value:
                        print(f"  - {param.name}: {param.description}")
                        if param.annotation:
                            print(f"    Type: {param.annotation}")

                elif section.kind == "returns":
                    print("Returns:")
                    for ret in section.value:
                        print(f"  Description: {ret.description}")
                        if ret.annotation:
                            print(f"  Type: {ret.annotation}")

                elif section.kind == "raises":
                    print("Raises:")
                    for exc in section.value:
                        print(f"  - {exc.annotation}: {exc.description}")

                elif section.kind == "examples":
                    print("Examples:")
                    print(section.value)

                elif section.kind == "notes":
                    print("Notes:")
                    print(section.value)

                elif section.kind == "warnings":
                    print("Warnings:")
                    print(section.value)
