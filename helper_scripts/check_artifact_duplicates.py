#!/usr/bin/env python3
"""Check duplicate objects in a MinIO XL artifacts directory.

This script scans a MinIO backend layout like:
  /data/landseer/lanseer-minio/landseer-artifacts/artifacts

It can report duplicates by:
1) logical object key (same key across multiple artifact IDs)
2) content hash (same bytes across different objects)

Notes:
- Object payloads stored as multipart are reconstructed by concatenating
  part.N files in numeric order.
- Inline MinIO objects (stored directly in xl.meta) are skipped for content
  hashing and reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PART_RE = re.compile(r"^part\.(\d+)$")


@dataclass
class ObjectRecord:
    artifact_id: str
    object_key: str
    object_dir: Path
    data_dir: Optional[Path]
    part_paths: List[Path]
    total_size: int
    inline_only: bool

    @property
    def display_id(self) -> str:
        return f"{self.artifact_id}:{self.object_key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate objects in MinIO XL artifact storage")
    parser.add_argument(
        "--root",
        default="/data/landseer/lanseer-minio/landseer-artifacts/artifacts",
        help="Path to the MinIO artifacts root",
    )
    parser.add_argument(
        "--mode",
        choices=["key", "content", "both"],
        default="both",
        help="Duplicate detection mode",
    )
    parser.add_argument(
        "--hash",
        default="sha256",
        help="Hash algorithm for content mode (default: sha256)",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=50,
        help="Maximum number of duplicate groups to print per section",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON report path",
    )
    return parser.parse_args()


def find_part_files(data_dir: Path) -> List[Path]:
    parts: List[Tuple[int, Path]] = []
    for child in data_dir.iterdir():
        if not child.is_file():
            continue
        match = PART_RE.match(child.name)
        if not match:
            continue
        parts.append((int(match.group(1)), child))
    parts.sort(key=lambda item: item[0])
    return [p for _, p in parts]


def collect_objects(root: Path) -> List[ObjectRecord]:
    records: List[ObjectRecord] = []

    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root path does not exist or is not a directory: {root}")

    for artifact_entry in sorted(root.iterdir()):
        if not artifact_entry.is_dir():
            continue
        artifact_id = artifact_entry.name

        for current, _dirs, files in os.walk(artifact_entry):
            if "xl.meta" not in files:
                continue

            obj_dir = Path(current)
            object_key = str(obj_dir.relative_to(artifact_entry))

            subdirs = [p for p in obj_dir.iterdir() if p.is_dir()]
            multipart_candidates: List[Tuple[Path, List[Path]]] = []
            for subdir in subdirs:
                parts = find_part_files(subdir)
                if parts:
                    multipart_candidates.append((subdir, parts))

            if not multipart_candidates:
                records.append(
                    ObjectRecord(
                        artifact_id=artifact_id,
                        object_key=object_key,
                        object_dir=obj_dir,
                        data_dir=None,
                        part_paths=[],
                        total_size=0,
                        inline_only=True,
                    )
                )
                continue

            for data_dir, part_paths in multipart_candidates:
                total_size = sum(p.stat().st_size for p in part_paths)
                records.append(
                    ObjectRecord(
                        artifact_id=artifact_id,
                        object_key=object_key,
                        object_dir=obj_dir,
                        data_dir=data_dir,
                        part_paths=part_paths,
                        total_size=total_size,
                        inline_only=False,
                    )
                )

    return records


def hash_object_parts(part_paths: Iterable[Path], algo: str) -> str:
    hasher = hashlib.new(algo)
    for path in part_paths:
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicates_by_key(records: Iterable[ObjectRecord]) -> Dict[str, List[ObjectRecord]]:
    grouped: Dict[str, List[ObjectRecord]] = defaultdict(list)
    for record in records:
        grouped[record.object_key].append(record)
    return {k: v for k, v in grouped.items() if len(v) > 1}


def find_duplicates_by_content(records: Iterable[ObjectRecord], algo: str) -> Tuple[Dict[str, List[ObjectRecord]], int]:
    grouped: Dict[str, List[ObjectRecord]] = defaultdict(list)
    hashed_count = 0

    for record in records:
        if record.inline_only or not record.part_paths:
            continue
        digest = hash_object_parts(record.part_paths, algo)
        grouped[digest].append(record)
        hashed_count += 1

    duplicates = {digest: recs for digest, recs in grouped.items() if len(recs) > 1}
    return duplicates, hashed_count


def print_key_duplicates(dupes: Dict[str, List[ObjectRecord]], max_groups: int) -> None:
    print("\n=== Duplicate Object Keys ===")
    if not dupes:
        print("No duplicate object keys found.")
        return

    shown = 0
    for key in sorted(dupes):
        if shown >= max_groups:
            break
        recs = dupes[key]
        print(f"\n- key: {key} (count={len(recs)})")
        for rec in recs:
            size_text = f"{rec.total_size} bytes" if not rec.inline_only else "inline/unknown-size"
            print(f"  - {rec.artifact_id} | {size_text}")
        shown += 1

    if len(dupes) > shown:
        print(f"\n... {len(dupes) - shown} more key-duplicate groups omitted.")


def print_content_duplicates(
    dupes: Dict[str, List[ObjectRecord]],
    max_groups: int,
    algo: str,
) -> None:
    print(f"\n=== Duplicate Object Content ({algo}) ===")
    if not dupes:
        print("No duplicate content groups found.")
        return

    shown = 0
    for digest, recs in sorted(dupes.items(), key=lambda item: len(item[1]), reverse=True):
        if shown >= max_groups:
            break
        size = recs[0].total_size if recs else 0
        print(f"\n- digest: {digest} (count={len(recs)}, size={size} bytes)")
        for rec in recs:
            print(f"  - {rec.artifact_id}:{rec.object_key}")
        shown += 1

    if len(dupes) > shown:
        print(f"\n... {len(dupes) - shown} more content-duplicate groups omitted.")


def to_jsonable_records(records: List[ObjectRecord]) -> List[dict]:
    return [
        {
            "artifact_id": rec.artifact_id,
            "object_key": rec.object_key,
            "object_dir": str(rec.object_dir),
            "data_dir": str(rec.data_dir) if rec.data_dir else None,
            "part_count": len(rec.part_paths),
            "total_size": rec.total_size,
            "inline_only": rec.inline_only,
        }
        for rec in records
    ]


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    try:
        records = collect_objects(root)
    except Exception as exc:
        print(f"Error while scanning root: {exc}", file=sys.stderr)
        return 2

    inline_count = sum(1 for r in records if r.inline_only)
    multipart_count = sum(1 for r in records if not r.inline_only)

    print(f"Scanned root: {root}")
    print(f"Total object records: {len(records)}")
    print(f"Multipart objects: {multipart_count}")
    print(f"Inline-only objects (content hash skipped): {inline_count}")

    key_dupes: Dict[str, List[ObjectRecord]] = {}
    content_dupes: Dict[str, List[ObjectRecord]] = {}
    hashed_count = 0

    if args.mode in {"key", "both"}:
        key_dupes = find_duplicates_by_key(records)
        print_key_duplicates(key_dupes, args.max_groups)

    if args.mode in {"content", "both"}:
        try:
            content_dupes, hashed_count = find_duplicates_by_content(records, args.hash)
        except ValueError as exc:
            print(f"Invalid hash algorithm '{args.hash}': {exc}", file=sys.stderr)
            return 2
        print(f"\nHashed objects for content comparison: {hashed_count}")
        print_content_duplicates(content_dupes, args.max_groups, args.hash)

    if args.json_out:
        report = {
            "root": str(root),
            "summary": {
                "total_records": len(records),
                "multipart_records": multipart_count,
                "inline_only_records": inline_count,
                "hashed_records": hashed_count,
                "key_duplicate_groups": len(key_dupes),
                "content_duplicate_groups": len(content_dupes),
            },
            "key_duplicates": {
                key: to_jsonable_records(recs) for key, recs in sorted(key_dupes.items())
            },
            "content_duplicates": {
                digest: to_jsonable_records(recs)
                for digest, recs in sorted(content_dupes.items(), key=lambda item: len(item[1]), reverse=True)
            },
        }
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
