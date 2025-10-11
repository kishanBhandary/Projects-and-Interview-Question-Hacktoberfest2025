#!/bin/bash

# Command: tar and zip
# Short Description: Quick commands for archiving and extracting files using tar and zip.
# Usage Examples:
#   - Create tar.gz archive: tar -czf archive.tar.gz directory/
#   - Extract tar.gz: tar -xzf archive.tar.gz
#   - Create zip archive: zip -r archive.zip directory/
#   - Extract zip: unzip archive.zip
# Notes:
#   - tar options: c=create, x=extract, z=gzip, f=file, v=verbose (add v for progress).
#   - zip is cross-platform, tar is Unix-like.
#   - Ensure sufficient disk space for archives.

echo "Example: Creating a tar.gz archive of current directory"
# tar -czf example.tar.gz .  # Commented to avoid actual creation

echo "Example: Extracting a tar.gz archive"
# tar -xzf example.tar.gz  # Commented

echo "Example: Creating a zip archive"
# zip -r example.zip .  # Commented

echo "Example: Extracting a zip archive"
# unzip example.zip  # Commented

echo "Commands demonstrated (run with actual files)."
