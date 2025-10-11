# Shell Commands Collection

This directory contains a collection of useful bash commands, each demonstrated in a separate `.sh` file. Each file includes the command, a short description, usage examples, and notes.

## File Format

Each command file should follow this structure:

- Shebang: `#!/bin/bash`
- Comments with:
  - Command name
  - Short description
  - Usage examples
  - Notes (cautions, tips, etc.)
- Demonstration code or examples

Files are named with a number prefix for ordering, e.g., `01_command.sh`.

## Contribution Guidelines

- **One command per file**: Focus on a single command or closely related commands (e.g., tar and zip).
- **Include usage and examples**: Provide practical examples that users can run or adapt.
- **Add notes**: Include any important caveats, permissions required, or alternatives.
- **Test your code**: Ensure the script runs without errors (comment out destructive operations if needed).
- **Follow naming convention**: Use descriptive names with number prefix.
- **Reference sources**: If applicable, link to official documentation or tutorials.

## Submission Guidelines

1. Fork the repository.
2. Create a new branch for your contribution.
3. Add your command file to the `shell_commands/` directory.
4. Update this README if necessary (e.g., add new categories).
5. Submit a pull request with a clear description of the command and its use case.

## Label Suggestions for PRs

When submitting a pull request, consider using these labels:

- `shell`: For shell scripting contributions
- `hacktoberfest`: To participate in Hacktoberfest
- `good first issue`: For beginner-friendly contributions
- `documentation`: For improvements to docs like this README

## Categories

- File management (find, grep, etc.)
- Text processing (sed, awk, etc.)
- Archiving (tar, zip, etc.)
- System info (netstat, ps, etc.)
- Data parsing (CSV, etc.)

Contributions are welcome! Let's build a comprehensive collection of bash commands.
