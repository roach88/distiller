#!/usr/bin/env python3
"""
Distiller CLI - Simple interface for Claude Code training data extraction.

Commands:
    distiller init       First-time setup (creates weekly cron)
    distiller status     Show system status and stats
    distiller run        Run incremental extraction
    distiller reprocess  Full reprocess (e.g., after changing judge LLM)
    distiller paths      Show file locations
"""

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# Paths
def get_distiller_root() -> Path:
    """Get the Distiller installation root."""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """Get the database path."""
    env_path = os.environ.get("DISTILLER_DB_PATH")
    if env_path:
        return Path(env_path)
    return get_distiller_root() / "data" / "raw_logs.db"


def get_output_dir() -> Path:
    """Get the output directory."""
    return get_distiller_root() / "output"


def get_state_file() -> Path:
    """Get the pipeline state file."""
    return get_output_dir() / ".distiller_state.json"


def get_config_path() -> Path:
    """Get the PII config path."""
    return get_distiller_root() / "config" / "pii_patterns.yaml"


# Helpers
def format_size(bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024
    return f"{bytes:.1f} TB"


def format_ago(timestamp: float) -> str:
    """Format timestamp as relative time."""
    delta = datetime.now().timestamp() - timestamp
    if delta < 60:
        return "just now"
    elif delta < 3600:
        mins = int(delta / 60)
        return f"{mins} minute{'s' if mins != 1 else ''} ago"
    elif delta < 86400:
        hours = int(delta / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = int(delta / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"


def setup_weekly_cron(hour: int = 3, day: int = 1) -> bool:
    """Set up weekly cron job. Returns True on success."""
    root = get_distiller_root()

    # Check if setup_schedule.py exists
    setup_script = root / "scripts" / "setup_schedule.py"
    if not setup_script.exists():
        return False

    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(setup_script),
        "--hour", str(hour),
        "--weekday", str(day),
    ]

    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    return result.returncode == 0


# Ollama/Flow-Judge setup
FLOW_JUDGE_MODEL = "flow-judge:latest"
FLOW_JUDGE_GGUF_URL = "https://huggingface.co/flowaicom/Flow-Judge-v0.1-GGUF/resolve/main/Flow-Judge-v0.1-Q8_0.gguf"
FLOW_JUDGE_GGUF_NAME = "Flow-Judge-v0.1-Q8_0.gguf"


def check_ollama_installed() -> bool:
    """Check if Ollama is installed."""
    result = subprocess.run(["which", "ollama"], capture_output=True)
    return result.returncode == 0


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_ollama_models() -> list[str]:
    """Get list of installed Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return []

        models = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                model_name = line.split()[0]
                models.append(model_name)
        return models
    except:
        return []


def install_ollama() -> bool:
    """Install Ollama. Returns True on success."""
    import platform

    console.print("   [dim]Installing Ollama...[/dim]")

    if platform.system() == "Darwin":
        # macOS - try brew first, fall back to curl
        result = subprocess.run(
            ["brew", "install", "ollama"],
            capture_output=True
        )
        if result.returncode == 0:
            return True

        # Fall back to curl installer
        result = subprocess.run(
            ["curl", "-fsSL", "https://ollama.com/install.sh", "-o", "/tmp/ollama_install.sh"],
            capture_output=True
        )
        if result.returncode == 0:
            result = subprocess.run(["sh", "/tmp/ollama_install.sh"], capture_output=True)
            return result.returncode == 0

    elif platform.system() == "Linux":
        result = subprocess.run(
            ["curl", "-fsSL", "https://ollama.com/install.sh", "-o", "/tmp/ollama_install.sh"],
            capture_output=True
        )
        if result.returncode == 0:
            result = subprocess.run(["sh", "/tmp/ollama_install.sh"], capture_output=True)
            return result.returncode == 0

    return False


def start_ollama() -> bool:
    """Start Ollama server. Returns True on success."""
    # Start ollama serve in background
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

    # Wait for it to start
    import time
    for _ in range(10):
        time.sleep(1)
        if check_ollama_running():
            return True

    return False


def setup_flow_judge_model() -> bool:
    """Download and create Flow-Judge model. Returns True on success."""
    root = get_distiller_root()
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    gguf_path = models_dir / FLOW_JUDGE_GGUF_NAME
    modelfile_path = root / "config" / "flow-judge.modelfile"

    # Download GGUF if not present
    if not gguf_path.exists():
        console.print(f"   [dim]Downloading Flow-Judge model (~2.4GB)...[/dim]")

        try:
            # Try huggingface-cli first (faster, supports resume)
            result = subprocess.run(
                ["huggingface-cli", "download", "flowaicom/Flow-Judge-v0.1-GGUF",
                 FLOW_JUDGE_GGUF_NAME, "--local-dir", str(models_dir)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Fall back to curl
                result = subprocess.run(
                    ["curl", "-L", "-o", str(gguf_path), FLOW_JUDGE_GGUF_URL],
                    capture_output=False  # Show progress
                )
                if result.returncode != 0:
                    return False
        except FileNotFoundError:
            # huggingface-cli not found, use curl
            result = subprocess.run(
                ["curl", "-L", "-o", str(gguf_path), FLOW_JUDGE_GGUF_URL],
                capture_output=False
            )
            if result.returncode != 0:
                return False

    # Create Ollama model from GGUF
    console.print(f"   [dim]Creating Ollama model...[/dim]")

    # Write modelfile pointing to downloaded GGUF
    temp_modelfile = models_dir / "Modelfile"
    temp_modelfile.write_text(f"""FROM {gguf_path}
TEMPLATE {{{{ .Prompt }}}}
PARAMETER temperature 0.1
PARAMETER stop <|endoftext|>
""")

    result = subprocess.run(
        ["ollama", "create", "flow-judge", "-f", str(temp_modelfile)],
        capture_output=True,
        text=True
    )

    # Clean up temp modelfile
    temp_modelfile.unlink(missing_ok=True)

    return result.returncode == 0


def ensure_ollama_ready(quiet: bool = False) -> bool:
    """Ensure Ollama is running and Flow-Judge model is available.

    Returns True if ready, False if setup failed.
    """
    # Check if Ollama is installed
    if not check_ollama_installed():
        if not quiet:
            console.print("[yellow]![/yellow] Ollama not installed. Run [cyan]distiller init[/cyan] first.")
            console.print("    Or use [cyan]--no-llm-judge[/cyan] for heuristic-only scoring.")
        return False

    # Start Ollama if not running
    if not check_ollama_running():
        if not quiet:
            console.print("[dim]Starting Ollama server...[/dim]")
        if not start_ollama():
            if not quiet:
                console.print("[yellow]![/yellow] Could not start Ollama. Start manually: [cyan]ollama serve[/cyan]")
            return False
        if not quiet:
            console.print("[green]✓[/green] Ollama server started")

    # Check if Flow-Judge model exists
    models = get_ollama_models()
    if FLOW_JUDGE_MODEL not in models and "flow-judge" not in [m.split(":")[0] for m in models]:
        if not quiet:
            console.print("[yellow]![/yellow] Flow-Judge model not found. Run [cyan]distiller setup-model[/cyan]")
            console.print("    Or use [cyan]--no-llm-judge[/cyan] for heuristic-only scoring.")
        return False

    return True


def run_pipeline(full: bool = False, llm_judge: bool = True,
                 project: Optional[str] = None, min_score: float = 0.6,
                 s3_bucket: Optional[str] = None, json_output: bool = False) -> int:
    """Run the pipeline. Returns exit code."""
    root = get_distiller_root()

    # If using LLM judge, ensure Ollama is ready
    if llm_judge:
        if not ensure_ollama_ready(quiet=json_output):
            if not json_output:
                console.print()
                console.print("[yellow]Falling back to heuristic-only scoring.[/yellow]")
                console.print()
            llm_judge = False

    # Step 1: Backfill from Claude Code's cache
    backfill_cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(root / "scripts" / "ingest.py"),
        "--backfill", str(Path.home() / ".claude")
    ]
    if project:
        backfill_cmd.extend(["--project", project])

    subprocess.run(backfill_cmd, capture_output=True, text=True)

    # Step 2: Run pipeline
    cmd = [str(root / ".venv" / "bin" / "python"), str(root / "pipeline.py")]

    if not full:
        cmd.append("--incremental")

    if llm_judge:
        cmd.append("--llm-judge")

    if project:
        cmd.extend(["--project", project])

    if min_score != 0.6:
        cmd.extend(["--min-score", str(min_score)])

    if s3_bucket:
        cmd.extend(["--s3-bucket", s3_bucket])

    if json_output:
        cmd.append("--json")

    result = subprocess.run(cmd, cwd=root)
    return result.returncode


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="distiller")
@click.pass_context
def cli(ctx):
    """Distiller - Extract training data from Claude Code sessions.

    Quick start:

    \b
      distiller init        # One-time setup (creates weekly cron)
      distiller run         # Manual run anytime
      distiller status      # Check progress
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.option("--hour", type=int, default=3, help="Hour for weekly cron (0-23, default: 3am)")
@click.option("--day", type=int, default=1, help="Day for weekly cron (1=Mon, 7=Sun, default: Monday)")
@click.option("--skip-cron", is_flag=True, help="Skip automatic cron setup")
@click.option("--skip-ollama", is_flag=True, help="Skip Ollama/Flow-Judge setup")
def init(hour: int, day: int, skip_cron: bool, skip_ollama: bool):
    """First-time setup.

    Sets up the database, Ollama with Flow-Judge model for quality scoring,
    and creates a weekly cron job to automatically extract training data.

    \b
    What this does:
      1. Creates/verifies the database
      2. Imports any existing Claude Code sessions
      3. Installs Ollama (if needed)
      4. Downloads Flow-Judge model for quality scoring
      5. Sets up weekly cron (Mondays at 3am by default)

    After setup, distiller runs automatically. Use 'distiller run' for
    manual runs or 'distiller reprocess' to redo everything.
    """
    console.print()
    console.print(Panel.fit(
        "[bold blue]Distiller Setup[/bold blue]\n\n"
        "Extract training data from your Claude Code sessions.",
        border_style="blue"
    ))
    console.print()

    root = get_distiller_root()
    db_path = get_db_path()

    # Step 1: Initialize database
    console.print("[bold]1. Database[/bold]")

    if db_path.exists():
        size = format_size(db_path.stat().st_size)
        console.print(f"   [green]✓[/green] Exists: {db_path} ({size})")
    else:
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            from scripts.init_db import init_database
            init_database(db_path)
            console.print(f"   [green]✓[/green] Created: {db_path}")
        except Exception as e:
            console.print(f"   [red]✗[/red] Failed: {e}")
            return
    console.print()

    # Step 2: Check for existing Claude Code sessions
    console.print("[bold]2. Claude Code Sessions[/bold]")

    claude_dir = Path.home() / ".claude" / "projects"
    if claude_dir.exists():
        session_count = sum(1 for _ in claude_dir.rglob("*.jsonl") if not _.name.startswith("agent-"))
        console.print(f"   [green]✓[/green] Found {session_count} session files")

        if session_count > 0:
            console.print(f"   [dim]   Importing existing sessions...[/dim]")

            # Run initial backfill
            backfill_cmd = [
                str(root / ".venv" / "bin" / "python"),
                str(root / "scripts" / "ingest.py"),
                "--backfill", str(Path.home() / ".claude")
            ]
            result = subprocess.run(backfill_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Check what we imported
                try:
                    conn = sqlite3.connect(db_path)
                    db_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
                    db_turns = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
                    conn.close()
                    console.print(f"   [green]✓[/green] Imported: {db_sessions} sessions, {db_turns:,} turns")
                except:
                    console.print(f"   [green]✓[/green] Import complete")
            else:
                console.print(f"   [yellow]![/yellow] Import had issues (non-fatal)")
    else:
        console.print(f"   [dim]-[/dim] No sessions yet (that's ok, they'll be captured)")
    console.print()

    # Step 3: Ollama setup
    console.print("[bold]3. Ollama (for LLM quality scoring)[/bold]")

    ollama_ready = False
    if skip_ollama:
        console.print(f"   [dim]-[/dim] Skipped (heuristic scoring only)")
    else:
        # Check if Ollama is installed
        if check_ollama_installed():
            console.print(f"   [green]✓[/green] Ollama installed")

            # Check if running
            if check_ollama_running():
                console.print(f"   [green]✓[/green] Ollama server running")
                ollama_ready = True
            else:
                console.print(f"   [dim]   Starting Ollama server...[/dim]")
                if start_ollama():
                    console.print(f"   [green]✓[/green] Ollama server started")
                    ollama_ready = True
                else:
                    console.print(f"   [yellow]![/yellow] Could not start Ollama")
                    console.print(f"       Run manually: [cyan]ollama serve[/cyan]")
        else:
            console.print(f"   [dim]   Ollama not found, installing...[/dim]")
            if install_ollama():
                console.print(f"   [green]✓[/green] Ollama installed")
                if start_ollama():
                    console.print(f"   [green]✓[/green] Ollama server started")
                    ollama_ready = True
            else:
                console.print(f"   [yellow]![/yellow] Could not install Ollama")
                console.print(f"       Install manually: [cyan]brew install ollama[/cyan]")
                console.print(f"       Or visit: [cyan]https://ollama.com[/cyan]")
    console.print()

    # Step 4: Flow-Judge model
    console.print("[bold]4. Flow-Judge Model[/bold]")

    if skip_ollama:
        console.print(f"   [dim]-[/dim] Skipped (Ollama not set up)")
    elif not ollama_ready:
        console.print(f"   [yellow]![/yellow] Skipped (Ollama not running)")
        console.print(f"       Run later: [cyan]distiller setup-model[/cyan]")
    else:
        # Check if model exists
        models = get_ollama_models()
        if FLOW_JUDGE_MODEL in models or "flow-judge" in [m.split(":")[0] for m in models]:
            console.print(f"   [green]✓[/green] Flow-Judge model ready")
        else:
            console.print(f"   [dim]   Setting up Flow-Judge model...[/dim]")
            if setup_flow_judge_model():
                console.print(f"   [green]✓[/green] Flow-Judge model installed")
            else:
                console.print(f"   [yellow]![/yellow] Could not set up Flow-Judge")
                console.print(f"       Run later: [cyan]distiller setup-model[/cyan]")
    console.print()

    # Step 5: Set up cron
    console.print("[bold]5. Weekly Schedule[/bold]")

    if skip_cron:
        console.print(f"   [dim]-[/dim] Skipped (use 'distiller run' manually)")
    else:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if setup_weekly_cron(hour, day):
            console.print(f"   [green]✓[/green] Cron installed: {days[day-1]}s at {hour}:00")
        else:
            console.print(f"   [yellow]![/yellow] Cron setup failed (use 'distiller run' manually)")
    console.print()

    # Summary
    console.print(Panel(
        "[bold green]Setup Complete![/bold green]\n\n"
        "Distiller will automatically extract training data weekly.\n\n"
        "[bold]Commands:[/bold]\n"
        "  [cyan]distiller status[/cyan]      - Check extraction stats\n"
        "  [cyan]distiller run[/cyan]         - Run extraction now\n"
        "  [cyan]distiller run --llm-judge[/cyan] - Run with LLM quality scoring\n"
        "  [cyan]distiller reprocess[/cyan]   - Full re-extraction\n"
        "  [cyan]distiller paths[/cyan]       - Show file locations",
        border_style="green"
    ))


@cli.command()
def status():
    """Show extraction status and statistics.

    Displays database size, session counts, last run info, and output files.
    """
    console.print()
    db_path = get_db_path()
    output_dir = get_output_dir()
    state_file = get_state_file()

    # Header
    console.print(Panel.fit("[bold]Distiller Status[/bold]", border_style="blue"))
    console.print()

    # Database info
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    if db_path.exists():
        size = format_size(db_path.stat().st_size)
        table.add_row("Database", f"{db_path}")
        table.add_row("Size", size)

        try:
            conn = sqlite3.connect(db_path)
            session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            turn_count = conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
            tool_count = conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
            conn.close()

            table.add_row("Sessions", str(session_count))
            table.add_row("Turns", f"{turn_count:,}")
            table.add_row("Tool Calls", f"{tool_count:,}")
        except Exception as e:
            table.add_row("Status", f"[red]Error: {e}[/red]")
    else:
        table.add_row("Database", "[yellow]Not initialized[/yellow]")
        table.add_row("", "[dim]Run: distiller init[/dim]")

    console.print(table)
    console.print()

    # Last run info
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
            last_run = state.get("last_run", "Unknown")
            last_output = state.get("last_output", "Unknown")
            last_timestamp = state.get("last_run_timestamp", 0)

            console.print("[bold]Last Run[/bold]")
            run_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            run_table.add_column("Key", style="dim")
            run_table.add_column("Value")

            if last_timestamp:
                run_table.add_row("When", f"{last_run} ({format_ago(last_timestamp)})")
            else:
                run_table.add_row("When", last_run)
            run_table.add_row("Output", str(last_output))

            console.print(run_table)
            console.print()
        except:
            pass

    # Output files
    if output_dir.exists():
        jsonl_files = sorted(output_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if jsonl_files:
            console.print("[bold]Recent Outputs[/bold]")
            out_table = Table(box=box.SIMPLE, padding=(0, 2))
            out_table.add_column("File", style="cyan")
            out_table.add_column("Size", justify="right")
            out_table.add_column("Modified", style="dim")

            for f in jsonl_files[:5]:
                size = format_size(f.stat().st_size)
                mtime = format_ago(f.stat().st_mtime)
                out_table.add_row(f.name, size, mtime)

            console.print(out_table)

            if len(jsonl_files) > 5:
                console.print(f"  [dim]...and {len(jsonl_files) - 5} more files[/dim]")
            console.print()


@cli.command()
@click.option("--no-llm-judge", is_flag=True, help="Skip LLM scoring, use heuristics only")
@click.option("--project", "-p", help="Filter to specific project")
@click.option("--min-score", type=float, default=0.6, help="Minimum quality score (0-1)")
@click.option("--s3-bucket", help="Upload output to S3 bucket")
@click.option("--json", "json_output", is_flag=True, help="Output stats as JSON")
def run(no_llm_judge: bool, project: Optional[str], min_score: float,
        s3_bucket: Optional[str], json_output: bool):
    """Run incremental extraction.

    Imports any new Claude Code sessions and extracts training data.
    Only processes sessions that haven't been extracted yet.

    By default, uses Flow-Judge LLM for quality scoring. Ollama will be
    started automatically if needed.

    \b
    This is what the weekly cron runs. Use 'distiller reprocess'
    if you need to re-extract everything (e.g., after changing settings).

    \b
    Examples:
        distiller run                 # With LLM quality scoring (default)
        distiller run --no-llm-judge  # Heuristic scoring only (faster)
        distiller run -p myproject    # Only extract from specific project
    """
    if not json_output:
        console.print()
        console.print(Panel.fit("[bold]Running Distiller[/bold]", border_style="blue"))
        console.print()
        console.print("[dim]Importing new sessions and extracting training data...[/dim]")
        console.print()

    exit_code = run_pipeline(
        full=False,
        llm_judge=not no_llm_judge,
        project=project,
        min_score=min_score,
        s3_bucket=s3_bucket,
        json_output=json_output
    )

    if exit_code == 0 and not json_output:
        console.print()
        console.print("[green]Done![/green] Run [cyan]distiller status[/cyan] to see results.")


@cli.command()
@click.option("--no-llm-judge", is_flag=True, help="Skip LLM scoring, use heuristics only")
@click.option("--project", "-p", help="Filter to specific project")
@click.option("--min-score", type=float, default=0.6, help="Minimum quality score (0-1)")
@click.option("--s3-bucket", help="Upload output to S3 bucket")
@click.confirmation_option(prompt="This will reprocess ALL sessions. Continue?")
def reprocess(no_llm_judge: bool, project: Optional[str], min_score: float,
              s3_bucket: Optional[str]):
    """Full re-extraction of all sessions.

    Reprocesses everything from scratch. Use this when you've changed
    settings that affect extraction (like the judge LLM or quality thresholds).

    By default, uses Flow-Judge LLM for quality scoring.

    \b
    Examples:
        distiller reprocess                 # Redo with LLM scoring (default)
        distiller reprocess --no-llm-judge  # Redo with heuristics only
        distiller reprocess --min-score 0.8 # Redo with higher quality bar
    """
    console.print()
    console.print(Panel.fit("[bold]Reprocessing All Sessions[/bold]", border_style="yellow"))
    console.print()
    console.print("[dim]This may take a while depending on how many sessions you have...[/dim]")
    console.print()

    exit_code = run_pipeline(
        full=True,
        llm_judge=not no_llm_judge,
        project=project,
        min_score=min_score,
        s3_bucket=s3_bucket,
        json_output=False
    )

    if exit_code == 0:
        console.print()
        console.print("[green]Reprocessing complete![/green] Run [cyan]distiller status[/cyan] to see results.")


@cli.command()
def paths():
    """Show important file paths.

    Displays the locations of the database, output directory, and other paths.
    """
    console.print()
    root = get_distiller_root()

    console.print(Panel.fit("[bold]Distiller Paths[/bold]", border_style="blue"))
    console.print()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column("Name", style="dim")
    table.add_column("Path")
    table.add_column("Status", justify="right")

    paths_to_check = [
        ("Installation", root, None),
        ("Database", get_db_path(), "file"),
        ("Output Dir", get_output_dir(), "dir"),
        ("PII Config", get_config_path(), "file"),
        ("State File", get_state_file(), "file"),
        ("Claude Sessions", Path.home() / ".claude" / "projects", "dir"),
    ]

    for name, path, check_type in paths_to_check:
        exists = path.exists()
        if check_type is None:
            status = ""
        elif exists:
            status = "[green]exists[/green]"
        else:
            status = "[dim]not found[/dim]"

        table.add_row(name, str(path), status)

    console.print(table)


@cli.command("setup-model")
def setup_model():
    """Install or reinstall the Flow-Judge model for LLM quality scoring.

    Downloads the Flow-Judge GGUF from HuggingFace and creates an Ollama model.
    Use this if the model wasn't set up during init or needs to be reinstalled.
    """
    console.print()
    console.print(Panel.fit("[bold]Flow-Judge Model Setup[/bold]", border_style="blue"))
    console.print()

    # Check Ollama
    if not check_ollama_installed():
        console.print("[red]Error:[/red] Ollama is not installed.")
        console.print("Install with: [cyan]brew install ollama[/cyan]")
        console.print("Or visit: [cyan]https://ollama.com[/cyan]")
        return

    if not check_ollama_running():
        console.print("[dim]Starting Ollama server...[/dim]")
        if not start_ollama():
            console.print("[red]Error:[/red] Could not start Ollama server.")
            console.print("Start manually with: [cyan]ollama serve[/cyan]")
            return
        console.print("[green]✓[/green] Ollama server started")

    # Check if model already exists
    models = get_ollama_models()
    if FLOW_JUDGE_MODEL in models or "flow-judge" in [m.split(":")[0] for m in models]:
        console.print("[yellow]Flow-Judge model already exists.[/yellow]")
        if not click.confirm("Reinstall it?", default=False):
            return

    # Setup model
    console.print("[dim]Downloading and setting up Flow-Judge model (~2.4GB)...[/dim]")
    console.print()

    if setup_flow_judge_model():
        console.print()
        console.print("[green]✓[/green] Flow-Judge model installed successfully!")
        console.print()
        console.print("LLM scoring is now enabled by default with [cyan]distiller run[/cyan]")
    else:
        console.print()
        console.print("[red]✗[/red] Failed to set up Flow-Judge model.")
        console.print()
        console.print("Try manually:")
        console.print("  1. Download GGUF from HuggingFace:")
        console.print(f"     [cyan]{FLOW_JUDGE_GGUF_URL}[/cyan]")
        console.print("  2. Create Ollama model:")
        console.print("     [cyan]ollama create flow-judge -f Modelfile[/cyan]")


@cli.command()
@click.option("--hour", type=int, help="Hour to run (0-23)")
@click.option("--day", type=int, help="Day of week (1=Monday, 7=Sunday)")
@click.option("--remove", is_flag=True, help="Remove the scheduled cron job")
def cron(hour: Optional[int], day: Optional[int], remove: bool):
    """View or update the weekly cron schedule.

    Without options, shows the current schedule.
    Use --hour and --day to update when the cron runs.
    Use --remove to disable automatic runs.

    \b
    Examples:
        distiller cron                  # Show current schedule
        distiller cron --hour 9         # Change to 9am (same day)
        distiller cron --day 5          # Change to Fridays (same hour)
        distiller cron --hour 18 --day 7  # Sundays at 6pm
        distiller cron --remove         # Disable automatic runs
    """
    root = get_distiller_root()
    setup_script = root / "scripts" / "setup_schedule.py"
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    console.print()

    if remove:
        # Remove cron
        if not setup_script.exists():
            console.print("[yellow]Schedule script not found.[/yellow]")
            return

        cmd = [
            str(root / ".venv" / "bin" / "python"),
            str(setup_script),
            "--uninstall",
        ]
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

        if result.returncode == 0:
            console.print("[green]✓[/green] Weekly cron removed.")
            console.print("[dim]Run 'distiller cron --hour H --day D' to set up again.[/dim]")
        else:
            console.print("[yellow]No cron job found to remove.[/yellow]")
        return

    if hour is not None or day is not None:
        # Update cron - need both values
        # Get current values if not provided
        current_hour = hour if hour is not None else 3
        current_day = day if day is not None else 1

        if setup_weekly_cron(current_hour, current_day):
            console.print(f"[green]✓[/green] Schedule updated: {days[current_day-1]}s at {current_hour}:00")
        else:
            console.print("[red]✗[/red] Failed to update schedule.")
        return

    # Show current schedule
    console.print(Panel.fit("[bold]Cron Schedule[/bold]", border_style="blue"))
    console.print()

    # Try to read current schedule from launchd plist or crontab
    import platform
    if platform.system() == "Darwin":
        plist_path = Path.home() / "Library" / "LaunchAgents" / "com.distiller.pipeline.plist"
        if plist_path.exists():
            try:
                import plistlib
                with open(plist_path, 'rb') as f:
                    plist = plistlib.load(f)
                calendar = plist.get('StartCalendarInterval', {})
                h = calendar.get('Hour', '?')
                d = calendar.get('Weekday', '?')
                if isinstance(d, int) and 0 <= d <= 6:
                    # launchd uses 0=Sunday, convert to our 1=Monday format
                    day_name = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][d]
                else:
                    day_name = "Unknown"
                console.print(f"   [green]✓[/green] Scheduled: {day_name}s at {h}:00")
                console.print()
                console.print("[bold]Update schedule:[/bold]")
                console.print("   distiller cron --hour H --day D")
                console.print()
                console.print("[dim]   --hour: 0-23 (e.g., 9 = 9am, 18 = 6pm)[/dim]")
                console.print("[dim]   --day:  1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun[/dim]")
                console.print()
                console.print("[dim]Remove with:[/dim] distiller cron --remove")
            except:
                console.print("   [yellow]![/yellow] Could not read schedule")
        else:
            console.print("   [dim]-[/dim] No cron job configured")
            console.print()
            console.print("[bold]Set up schedule:[/bold]")
            console.print("   distiller cron --hour H --day D")
            console.print()
            console.print("[dim]   --hour: 0-23 (e.g., 9 = 9am, 18 = 6pm)[/dim]")
            console.print("[dim]   --day:  1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun[/dim]")
    else:
        # Linux - check crontab
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if "distiller" in result.stdout:
            console.print("   [green]✓[/green] Cron job exists (check 'crontab -l' for details)")
        else:
            console.print("   [dim]-[/dim] No cron job configured")
            console.print()
            console.print("[bold]Set up schedule:[/bold]")
            console.print("   distiller cron --hour H --day D")
            console.print()
            console.print("[dim]   --hour: 0-23 (e.g., 9 = 9am, 18 = 6pm)[/dim]")
            console.print("[dim]   --day:  1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat, 7=Sun[/dim]")


@cli.command()
def uninstall():
    """Remove the weekly cron job.

    Stops automatic extraction. Your data is preserved.
    """
    root = get_distiller_root()
    setup_script = root / "scripts" / "setup_schedule.py"

    if not setup_script.exists():
        console.print("[yellow]Schedule script not found.[/yellow]")
        return

    cmd = [
        str(root / ".venv" / "bin" / "python"),
        str(setup_script),
        "--uninstall",
    ]

    console.print()
    result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("[green]✓[/green] Weekly cron removed.")
        console.print("[dim]Run 'distiller init' to set it up again.[/dim]")
    else:
        console.print("[yellow]No cron job found to remove.[/yellow]")


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
