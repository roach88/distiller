#!/usr/bin/env python3
"""
Set up scheduled weekly runs of the Distiller pipeline.

Supports macOS (launchd) and Linux (cron).

Usage:
    python scripts/setup_schedule.py                    # Local output
    python scripts/setup_schedule.py --s3-bucket bucket # S3 upload
    python scripts/setup_schedule.py --uninstall        # Remove schedule
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


LAUNCHD_PLIST = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.distiller.pipeline</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{pipeline_path}</string>
        <string>--incremental</string>
        <string>--llm-judge</string>
{extra_args}
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>{weekday}</integer>
        <key>Hour</key>
        <integer>{hour}</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{log_path}/distiller.log</string>
    <key>StandardErrorPath</key>
    <string>{log_path}/distiller.error.log</string>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{venv_bin}</string>
    </dict>
</dict>
</plist>
"""


def get_cron_line(pipeline_path: str, python_path: str, extra_args: str, hour: int, weekday: int) -> str:
    """Generate cron line for weekly schedule."""
    # Cron weekday: 0=Sunday, launchd weekday: 1=Monday
    cron_weekday = weekday % 7  # Convert if needed
    return f"0 {hour} * * {cron_weekday} cd {Path(pipeline_path).parent} && {python_path} {pipeline_path} --incremental --llm-judge {extra_args} >> ~/.distiller/distiller.log 2>&1"


def setup_macos(
    pipeline_path: Path,
    python_path: Path,
    s3_bucket: str = None,
    s3_prefix: str = "training-data",
    hour: int = 3,
    weekday: int = 1,  # Monday
) -> None:
    """Set up launchd on macOS."""
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.distiller.pipeline.plist"

    log_path = Path.home() / ".distiller"
    log_path.mkdir(parents=True, exist_ok=True)

    venv_bin = python_path.parent

    # Build extra args
    extra_args_lines = []
    if s3_bucket:
        extra_args_lines.append(f"        <string>--s3-bucket</string>")
        extra_args_lines.append(f"        <string>{s3_bucket}</string>")
        extra_args_lines.append(f"        <string>--s3-prefix</string>")
        extra_args_lines.append(f"        <string>{s3_prefix}</string>")

    plist_content = LAUNCHD_PLIST.format(
        python_path=python_path,
        pipeline_path=pipeline_path,
        extra_args="\n".join(extra_args_lines),
        weekday=weekday,
        hour=hour,
        log_path=log_path,
        working_dir=pipeline_path.parent,
        venv_bin=venv_bin,
    )

    plist_path.write_text(plist_content)
    print(f"Created: {plist_path}")

    # Load the launch agent
    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True)

    if result.returncode == 0:
        print(f"Loaded launchd agent")
        print(f"\nSchedule: Every {'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()[weekday-1]} at {hour}:00")
        print(f"Logs: {log_path}/distiller.log")
    else:
        print(f"Error loading agent: {result.stderr.decode()}")


def setup_linux(
    pipeline_path: Path,
    python_path: Path,
    s3_bucket: str = None,
    s3_prefix: str = "training-data",
    hour: int = 3,
    weekday: int = 1,
) -> None:
    """Set up cron on Linux."""
    extra_args = ""
    if s3_bucket:
        extra_args = f"--s3-bucket {s3_bucket} --s3-prefix {s3_prefix}"

    cron_line = get_cron_line(str(pipeline_path), str(python_path), extra_args, hour, weekday)

    # Get current crontab
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    current_crontab = result.stdout if result.returncode == 0 else ""

    # Remove any existing distiller entries
    lines = [l for l in current_crontab.strip().split("\n") if "distiller" not in l.lower() and l.strip()]
    lines.append(cron_line)

    # Install new crontab
    new_crontab = "\n".join(lines) + "\n"
    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(new_crontab)

    if process.returncode == 0:
        print(f"Installed cron job")
        print(f"\nSchedule: Every week at {hour}:00 on day {weekday}")
        print(f"Cron line: {cron_line}")
    else:
        print("Error installing cron job")


def uninstall_macos() -> None:
    """Remove launchd agent on macOS."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.distiller.pipeline.plist"
    if plist_path.exists():
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        plist_path.unlink()
        print(f"Removed: {plist_path}")
    else:
        print("No launchd agent found")


def uninstall_linux() -> None:
    """Remove cron job on Linux."""
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        print("No crontab found")
        return

    lines = [l for l in result.stdout.strip().split("\n") if "distiller" not in l.lower() and l.strip()]
    new_crontab = "\n".join(lines) + "\n" if lines else ""

    process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
    process.communicate(new_crontab)
    print("Removed distiller cron job")


def main():
    parser = argparse.ArgumentParser(description="Set up scheduled Distiller pipeline runs")
    parser.add_argument("--s3-bucket", help="S3 bucket for output")
    parser.add_argument("--s3-prefix", default="training-data", help="S3 key prefix")
    parser.add_argument("--hour", type=int, default=3, help="Hour to run (0-23, default: 3)")
    parser.add_argument("--weekday", type=int, default=1, help="Day of week (1=Mon, 7=Sun, default: 1)")
    parser.add_argument("--uninstall", action="store_true", help="Remove scheduled job")

    args = parser.parse_args()

    is_macos = platform.system() == "Darwin"

    if args.uninstall:
        if is_macos:
            uninstall_macos()
        else:
            uninstall_linux()
        return

    # Find paths
    pipeline_path = Path(__file__).parent.parent / "pipeline.py"
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python"

    if venv_python.exists():
        python_path = venv_python
    else:
        python_path = Path(sys.executable)

    print(f"Pipeline: {pipeline_path}")
    print(f"Python: {python_path}")

    if is_macos:
        setup_macos(
            pipeline_path,
            python_path,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            hour=args.hour,
            weekday=args.weekday,
        )
    else:
        setup_linux(
            pipeline_path,
            python_path,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            hour=args.hour,
            weekday=args.weekday,
        )

    print("\nTo test manually:")
    print(f"  {python_path} {pipeline_path} --incremental --llm-judge")


if __name__ == "__main__":
    main()
