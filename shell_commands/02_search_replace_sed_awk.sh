#!/bin/bash

# Command: sed and awk
# Short Description: Demonstrate basic search and replace using sed, and text processing with awk.
# Usage Examples:
#   - Sed for replacement: sed 's/old_text/new_text/g' input.txt > output.txt
#   - Awk for printing columns: awk '{print $1, $3}' input.txt
#   - Awk for conditional processing: awk '$2 > 50 {print $1}' input.txt
# Notes:
#   - Sed is stream editor for filtering and transforming text.
#   - Awk is a programming language for text processing, good for columnar data.
#   - Always backup files before in-place edits (sed -i).

echo "Example: Using sed to replace 'hello' with 'hi' in a string"
echo "hello world" | sed 's/hello/hi/'

echo "Example: Using awk to print the first column from a CSV-like input"
echo -e "name,age,city\nJohn,25,NYC\nJane,30,LA" | awk -F',' '{print $1}'

echo "Demonstration complete."
