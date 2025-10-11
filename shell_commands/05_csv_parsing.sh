#!/bin/bash

# Command: awk, cut
# Short Description: Bash one-liners for parsing CSV files.
# Usage Examples:
#   - Print first column with awk: awk -F',' '{print $1}' file.csv
#   - Print first column with cut: cut -d',' -f1 file.csv
#   - Sum a column: awk -F',' '{sum += $2} END {print sum}' file.csv
#   - Filter rows: awk -F',' '$2 > 100 {print $1}' file.csv
# Notes:
#   - Assumes simple CSV without quoted fields or escapes.
#   - For complex CSV, use tools like csvkit or python.
#   - awk is powerful for calculations and filtering.

echo "Example CSV data:"
cat << EOF > example.csv
name,age,city
John,25,NYC
Jane,30,LA
Bob,22,SF
EOF

echo "Printing first column (names):"
awk -F',' '{print $1}' example.csv

echo "Summing ages:"
awk -F',' 'NR>1 {sum += $2} END {print sum}' example.csv

echo "Filtering: names where age > 24"
awk -F',' '$2 > 24 {print $1}' example.csv

rm example.csv
echo "Demonstration complete."
