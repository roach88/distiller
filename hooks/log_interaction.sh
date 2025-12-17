#!/bin/bash
#
# Distiller: Claude Code Logging Hook
#
# This hook captures Claude Code interactions and logs them to the Distiller database.
# It reads JSON event data from stdin and passes it to the Python logging module.
#
# Installation:
#   1. Make this script executable: chmod +x hooks/log_interaction.sh
#   2. Add to ~/.claude/settings.json (see INSTALL.md)
#
# Environment Variables:
#   DISTILLER_DB_PATH - Override default database path
#   DISTILLER_ROOT    - Override Distiller project root
#

set -e

# Determine Distiller root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISTILLER_ROOT="${DISTILLER_ROOT:-$(dirname "$SCRIPT_DIR")}"

# Python script path
LOG_SCRIPT="$DISTILLER_ROOT/scripts/log_event.py"

# Check if Python script exists
if [ ! -f "$LOG_SCRIPT" ]; then
    echo '{"status": "error", "message": "log_event.py not found"}'
    exit 0  # Exit 0 to not break Claude Code
fi

# Read stdin and pass to Python
# Use timeout to prevent hanging
if command -v timeout &> /dev/null; then
    timeout 5 python3 "$LOG_SCRIPT"
elif command -v gtimeout &> /dev/null; then
    gtimeout 5 python3 "$LOG_SCRIPT"
else
    python3 "$LOG_SCRIPT"
fi

# Always exit 0 to not break Claude Code
exit 0
