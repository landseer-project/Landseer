#!/usr/bin/env python3
"""
Lightweight worker watchdog for local Landseer runs.

This script polls backend health/progress/workers and tails worker logs to
surface early warnings:
  - stale worker heartbeats
  - task failures (including rises in tasks_failed)
  - disk quota / no-space errors in worker logs
  - pipeline stall (running+pending but no progress movement)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_LOG_DIR = Path("logs")
ERROR_PATTERNS = (
    "disk quota exceeded",
    "errno 122",
    "no space left on device",
    "task ",
    " execution error",
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # API emits naive local ISO strings; treat as local then convert UTC.
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def fetch_json(base_url: str, endpoint: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    with urlopen(url, timeout=5.0) as resp:  # nosec - local trusted endpoint
        return json.loads(resp.read().decode("utf-8"))


def read_recent_matches(log_path: Path, max_lines: int = 200) -> List[str]:
    if not log_path.exists():
        return []

    try:
        lines = log_path.read_text(errors="replace").splitlines()[-max_lines:]
    except Exception:
        return []

    matches: List[str] = []
    for line in lines:
        l = line.lower()
        if "task " in l and ("execution error" in l or "failed" in l):
            matches.append(line)
            continue
        if any(pattern in l for pattern in ERROR_PATTERNS[:3]):
            matches.append(line)
    return matches[-5:]


def summarize_workers(workers: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for worker in workers:
        wid = worker.get("worker_id", "?")
        gpu = (worker.get("capabilities") or {}).get("gpu_id")
        status = worker.get("status", "?")
        task = worker.get("current_task_id") or "-"
        failed = worker.get("tasks_failed", 0)
        parts.append(f"{wid}(gpu={gpu},status={status},task={task},failed={failed})")
    return " | ".join(parts)


def check_alerts(
    workers: List[Dict[str, Any]],
    progress: Dict[str, Any],
    stale_seconds: int,
    log_dir: Path,
    baseline_failed: Dict[str, int],
    last_progress_key: Optional[Tuple[int, int, int]],
    last_progress_change_at: float,
    stall_seconds: int,
) -> Tuple[List[str], Tuple[int, int, int], float]:
    alerts: List[str] = []
    now = now_utc()

    for w in workers:
        wid = w.get("worker_id", "<unknown>")
        hb = parse_iso(w.get("last_heartbeat", ""))
        if hb:
            age = int((now - hb).total_seconds())
            if age > stale_seconds and w.get("status") != "offline":
                alerts.append(
                    f"STALE_HEARTBEAT worker={wid} age={age}s status={w.get('status')}"
                )

        failed = int(w.get("tasks_failed", 0))
        prev = baseline_failed.get(wid, failed)
        if failed > prev:
            alerts.append(
                f"TASK_FAILURES_INCREASED worker={wid} from={prev} to={failed} "
                f"current_task={w.get('current_task_id')}"
            )
        baseline_failed[wid] = failed

        gpu = (w.get("capabilities") or {}).get("gpu_id")
        if gpu is not None:
            matches = read_recent_matches(log_dir / f"worker_gpu{gpu}.log")
            for m in matches:
                if "disk quota exceeded" in m.lower() or "errno 122" in m.lower():
                    alerts.append(f"DISK_QUOTA worker={wid} gpu={gpu} log={m.strip()}")

    progress_key = (
        int(progress.get("completed", 0)),
        int(progress.get("failed", 0)),
        int(progress.get("running", 0)),
    )
    if last_progress_key is None or progress_key != last_progress_key:
        last_progress_change_at = time.time()
    else:
        stalled_for = int(time.time() - last_progress_change_at)
        pending = int(progress.get("pending", 0))
        running = int(progress.get("running", 0))
        if (pending > 0 or running > 0) and stalled_for >= stall_seconds:
            alerts.append(
                f"PIPELINE_STALLED stalled_for={stalled_for}s "
                f"pending={pending} running={running} completed={progress_key[0]} failed={progress_key[1]}"
            )

    return alerts, progress_key, last_progress_change_at


def run(args: argparse.Namespace) -> int:
    baseline_failed: Dict[str, int] = {}
    last_progress_key: Optional[Tuple[int, int, int]] = None
    last_progress_change_at = time.time()
    had_alert = False

    while True:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            health = fetch_json(args.backend_url, "/health")
            workers_data = fetch_json(args.backend_url, "/workers")
            progress = fetch_json(args.backend_url, "/progress")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"[{ts}] ALERT BACKEND_UNREACHABLE: {e}", flush=True)
            had_alert = True
            if args.once:
                return 2
            time.sleep(args.interval)
            continue

        workers = workers_data.get("workers", [])
        alerts, last_progress_key, last_progress_change_at = check_alerts(
            workers=workers,
            progress=progress,
            stale_seconds=args.stale_seconds,
            log_dir=args.log_dir,
            baseline_failed=baseline_failed,
            last_progress_key=last_progress_key,
            last_progress_change_at=last_progress_change_at,
            stall_seconds=args.stall_seconds,
        )

        status_line = (
            f"[{ts}] health={health.get('status')} total={progress.get('total')} "
            f"pending={progress.get('pending')} running={progress.get('running')} "
            f"completed={progress.get('completed')} failed={progress.get('failed')}"
        )
        print(status_line, flush=True)
        print(f"[{ts}] workers: {summarize_workers(workers)}", flush=True)

        if alerts:
            had_alert = True
            for alert in alerts:
                print(f"[{ts}] ALERT {alert}", flush=True)
            if args.exit_on_alert:
                return 1
        else:
            print(f"[{ts}] OK no worker alerts", flush=True)

        if args.once:
            return 1 if had_alert else 0
        time.sleep(args.interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor Landseer workers and detect early failure signals."
    )
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--interval", type=int, default=5, help="Poll interval in seconds")
    parser.add_argument("--stale-seconds", type=int, default=90)
    parser.add_argument("--stall-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true", help="Run a single check and exit")
    parser.add_argument(
        "--exit-on-alert",
        action="store_true",
        help="Exit non-zero immediately when an alert is detected",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
