import os

folders = [
    ".vscode",
    ".github/workflows",
    "src",
    "notebooks",
    "tests",
    "scripts",
]

files = {
    ".vscode/settings.json": "",
    ".github/workflows/unittests.yml": "",
    ".gitignore": "",
    "requirements.txt": "",
    "README.md": "",
    "notebooks/__init__.py": "",
    "notebooks/README.md": "",
    "tests/__init__.py": "",
    "scripts/__init__.py": "",
    "scripts/README.md": "",
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created folder: {folder}")

for file_path, content in files.items():
    with open(file_path, "w") as file:
        file.write(content)
    print(f"Created file: {file_path}")

print("Project structure created successfully!")