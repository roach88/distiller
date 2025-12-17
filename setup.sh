#!/bin/bash
#
# Distiller Setup Script
# Installs dependencies and adds shell commands
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELL_NAME="$(basename "$SHELL")"

echo "=== Distiller Setup ==="
echo "Directory: $SCRIPT_DIR"
echo "Shell: $SHELL_NAME"
echo

# 1. Create virtual environment if needed
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

# 2. Install dependencies
echo "Installing dependencies..."
"$SCRIPT_DIR/.venv/bin/pip" install -q pyyaml requests boto3

# 3. Initialize database if needed
if [ ! -f "$SCRIPT_DIR/data/raw_logs.db" ]; then
    echo "Initializing database..."
    "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/scripts/init_db.py" 2>/dev/null || true
fi

# 4. Detect shell config file
case "$SHELL_NAME" in
    zsh)
        SHELL_RC="$HOME/.zshrc"
        ;;
    bash)
        if [ -f "$HOME/.bash_profile" ]; then
            SHELL_RC="$HOME/.bash_profile"
        else
            SHELL_RC="$HOME/.bashrc"
        fi
        ;;
    *)
        SHELL_RC="$HOME/.profile"
        ;;
esac

# 5. Generate shell commands
SHELL_COMMANDS="
# Distiller - Claude Code training data pipeline
export DISTILLER_HOME=\"$SCRIPT_DIR\"
alias distiller=\"\$DISTILLER_HOME/.venv/bin/python \$DISTILLER_HOME/pipeline.py\"
alias distiller-run=\"distiller --incremental --llm-judge\"
alias distiller-full=\"distiller --llm-judge\"
alias distiller-schedule=\"\$DISTILLER_HOME/.venv/bin/python \$DISTILLER_HOME/scripts/setup_schedule.py\"
"

# 6. Check if already installed
if grep -q "DISTILLER_HOME" "$SHELL_RC" 2>/dev/null; then
    echo "Shell commands already in $SHELL_RC"
    echo "Updating..."
    # Remove old distiller block
    sed -i.bak '/# Distiller - Claude Code/,/alias distiller-schedule/d' "$SHELL_RC"
fi

# 7. Add to shell config
echo "$SHELL_COMMANDS" >> "$SHELL_RC"
echo "Added commands to $SHELL_RC"

# 8. Summary
echo
echo "=== Setup Complete ==="
echo
echo "Commands available (restart shell or run: source $SHELL_RC):"
echo
echo "  distiller           Run pipeline with custom options"
echo "  distiller-run       Quick incremental run with LLM scoring"
echo "  distiller-full      Full reprocess with LLM scoring"
echo "  distiller-schedule  Set up weekly automated runs"
echo
echo "Examples:"
echo "  distiller --help"
echo "  distiller-run"
echo "  distiller-run --s3-bucket my-bucket"
echo "  distiller-schedule --s3-bucket my-bucket"
echo
