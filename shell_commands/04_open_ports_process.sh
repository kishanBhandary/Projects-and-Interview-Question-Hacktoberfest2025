#!/bin/bash

# Command: netstat, ss, ps, lsof
# Short Description: Show open ports and process information.
# Usage Examples:
#   - List listening ports with netstat: netstat -tlnp
#   - List listening ports with ss: ss -tlnp
#   - Show process info: ps aux | head
#   - List open files/network: lsof -i
# Notes:
#   - netstat is older, ss is modern replacement.
#   - ps shows processes, lsof shows open files including network sockets.
#   - May require sudo for full info.

echo "Example: Listing listening TCP ports with ss"
ss -tlnp

echo "Example: Showing process information"
ps aux | head -10

echo "Example: Listing open network connections with lsof"
lsof -i | head -10

echo "Demonstration complete."
