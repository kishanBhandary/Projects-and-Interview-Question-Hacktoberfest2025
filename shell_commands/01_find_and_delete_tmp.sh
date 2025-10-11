#!/bin/bash

# Command: find
# Short Description: Find and delete old temporary files in /tmp directory that are older than 7 days.
# Usage Examples:
#   - To find and list files older than 7 days: find /tmp -type f -mtime +7
#   - To delete them: find /tmp -type f -mtime +7 -delete
#   - This script combines both: first lists, then deletes.
# Notes:
#   - Be cautious with -delete; always test with -print first to avoid accidental deletion.
#   - Requires appropriate permissions to delete files in /tmp.
#   - Adjust the path and mtime as needed for your use case.

echo "Listing files older than 7 days in /tmp:"
find /tmp -type f -mtime +7

echo "Deleting files older than 7 days in /tmp..."
find /tmp -type f -mtime +7 -delete

echo "Cleanup complete."
