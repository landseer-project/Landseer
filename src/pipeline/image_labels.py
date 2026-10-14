"""
Landseer image label schema (v1) for tool, dataset-loader and evaluator images.

Every Landseer image carries two groups of labels:

* OCI standard annotations (``org.opencontainers.image.*``) for identity and
  provenance: title, description, vendor, source, base image, and, stamped at
  build time, version / revision / created.
* Landseer metadata (``org.landseer.*``) for what the pipeline needs to know:
  which kind of image it is, which datasets it supports, which framework and
  accelerator it needs, and the type-specific contract (stage for tools,
  dataset/variant/outputs for loaders, metrics/artifacts for evaluators).

List-valued labels are comma-separated with no spaces; ``*`` means "any".
The full reference is ``docs/reference/image-labels.md``.

Run ``python src/pipeline/image_labels.py lint`` (stdlib only) to check the
Dockerfiles under ``containers/`` and ``python -m src.pipeline.image_labels audit``
to check the published images referenced by ``configs/*.yaml``.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCHEMA_VERSION = "1"

OCI = "org.opencontainers.image."
NS = "org.landseer."

# Labels every image must declare in its Dockerfile.
COMMON_REQUIRED = (
    OCI + "title",
    OCI + "description",
    OCI + "vendor",
    OCI + "source",
    OCI + "base.name",
    NS + "schema",
    NS + "type",
    NS + "name",
    NS + "datasets",
    NS + "modality",
    NS + "framework",
    NS + "python",
    NS + "accelerator",
)

# Labels stamped at build time (CI metadata-action or --label); only checked on
# built images, never in Dockerfiles.
BUILD_STAMPED = (
    OCI + "version",
    OCI + "revision",
    OCI + "created",
)

TYPE_REQUIRED: Dict[str, tuple] = {
    "tool": (NS + "stage", NS + "defense_type"),
    "dataset": (NS + "dataset", NS + "variant", NS + "outputs"),
    "evaluator": (NS + "metrics", NS + "required_artifacts", NS + "defense_types"),
    "adapter": (NS + "source_framework", NS + "target_framework"),
}

# Labels that may legitimately be an empty string.
MAY_BE_EMPTY = {
    NS + "required_artifacts",
    NS + "defense_types",
    NS + "inputs",
}

ALLOWED_VALUES: Dict[str, frozenset] = {
    NS + "type": frozenset(TYPE_REQUIRED),
    NS + "accelerator": frozenset({"cpu", "cuda"}),
    NS + "stage": frozenset({"pre_training", "during_training", "post_training", "deployment"}),
}
ALLOWED_LIST_VALUES: Dict[str, frozenset] = {
    NS + "modality": frozenset({"image", "text", "tabular"}),
}

# Old keys still read as fallbacks; lint reports them so they get migrated.
LEGACY_KEYS: Dict[str, str] = {
    OCI + "dataset": NS + "datasets",
    OCI + "stage": NS + "stage",
    OCI + "defense_stage": NS + "stage",
    NS + "category": NS + "type",
}

VENDOR = "Landseer Project"


def _split_list(value: str) -> List[str]:
    return [p.strip() for p in str(value).split(",") if p.strip()]


def supported_datasets_label(labels: Dict[str, str]) -> str:
    """Raw supported-datasets value, preferring the v1 key over the legacy OCI one."""
    return labels.get(NS + "datasets") or labels.get(OCI + "dataset", "")


def stage_label(labels: Dict[str, str]) -> Optional[str]:
    """Raw stage value, preferring the v1 key over the legacy OCI ones."""
    return (
        labels.get(NS + "stage")
        or labels.get(OCI + "stage")
        or labels.get(OCI + "defense_stage")
        or None
    )


@dataclass
class LabelReport:
    subject: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_labels(
    labels: Dict[str, str],
    subject: str = "",
    *,
    built: bool = False,
    expected_type: Optional[str] = None,
) -> LabelReport:
    """
    Check ``labels`` against schema v1.

    ``built`` also requires the build-stamped OCI labels (version, revision,
    created); leave it off when checking a Dockerfile.
    """
    report = LabelReport(subject)
    image_type = labels.get(NS + "type", "")

    required = list(COMMON_REQUIRED) + list(TYPE_REQUIRED.get(image_type, ()))
    if built:
        required += list(BUILD_STAMPED)
    for key in required:
        if key not in labels:
            report.errors.append(f"missing {key}")
        elif not str(labels[key]).strip() and key not in MAY_BE_EMPTY:
            report.errors.append(f"empty {key}")

    if expected_type and image_type and image_type != expected_type:
        report.errors.append(f"{NS}type is {image_type!r}, expected {expected_type!r}")

    schema = labels.get(NS + "schema")
    if schema and schema != SCHEMA_VERSION:
        report.errors.append(f"{NS}schema is {schema!r}, expected {SCHEMA_VERSION!r}")

    vendor = labels.get(OCI + "vendor")
    if vendor and vendor != VENDOR:
        report.warnings.append(f"{OCI}vendor is {vendor!r}, expected {VENDOR!r}")

    for key, allowed in ALLOWED_VALUES.items():
        value = labels.get(key)
        if value and value not in allowed:
            report.errors.append(f"{key}={value!r} not in {sorted(allowed)}")
    for key, allowed in ALLOWED_LIST_VALUES.items():
        for item in _split_list(labels.get(key, "")):
            if item not in allowed:
                report.errors.append(f"{key} item {item!r} not in {sorted(allowed)}")

    if labels.get(NS + "accelerator") == "cuda" and not labels.get(NS + "cuda"):
        report.errors.append(f"{NS}accelerator=cuda requires {NS}cuda (CUDA version)")

    for key in (NS + "datasets", NS + "metrics", NS + "required_artifacts",
                NS + "defense_types", NS + "outputs", NS + "modality"):
        value = labels.get(key, "")
        if value and (", " in value or " ," in value):
            report.errors.append(f"{key} must be comma-separated without spaces: {value!r}")

    if image_type == "dataset":
        dataset = labels.get(NS + "dataset")
        datasets = _split_list(labels.get(NS + "datasets", ""))
        if dataset and datasets and datasets != [dataset]:
            report.errors.append(f"{NS}datasets must equal {NS}dataset on a dataset image")

    for old, new in LEGACY_KEYS.items():
        if old in labels:
            report.warnings.append(f"legacy label {old}; use {new}")

    return report


# ---------------------------------------------------------------------------
# Dockerfile parsing
# ---------------------------------------------------------------------------

def _logical_lines(text: str) -> Iterable[str]:
    buf = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buf and (not stripped or stripped.startswith("#")):
            continue
        if buf and stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        buf += stripped
        yield buf
        buf = ""
    if buf:
        yield buf


def parse_dockerfile_labels(text: str) -> Dict[str, str]:
    """
    Labels declared in the final stage of a Dockerfile.

    Handles multiple ``key=value`` pairs per ``LABEL``, quoting, and line
    continuations. ``${ARG}`` references are kept verbatim.
    """
    labels: Dict[str, str] = {}
    for line in _logical_lines(text):
        parts = line.split(None, 1)
        instruction = parts[0].upper()
        if instruction == "FROM":
            labels = {}
            from_args = shlex.split(parts[1]) if len(parts) > 1 else []
            from_args = [a for a in from_args if not a.startswith("--")]
            if from_args:
                labels["__from__"] = from_args[0]
            continue
        if instruction != "LABEL" or len(parts) < 2:
            continue
        tokens = shlex.split(parts[1])
        if len(tokens) == 2 and "=" not in tokens[0]:
            # Legacy ``LABEL key value`` form.
            labels[tokens[0]] = tokens[1]
            continue
        for token in tokens:
            if "=" in token:
                key, value = token.split("=", 1)
                labels[key] = value
    return labels


def lint_dockerfile(path: Path, expected_type: Optional[str] = None) -> LabelReport:
    labels = parse_dockerfile_labels(path.read_text())
    base = labels.pop("__from__", None)
    report = validate_labels(labels, str(path), built=False, expected_type=expected_type)
    declared_base = labels.get(OCI + "base.name")
    if base and declared_base and declared_base != base:
        report.errors.append(f"{OCI}base.name={declared_base!r} but FROM is {base!r}")
    return report


# Directory under containers/ -> required org.landseer.type.
CONTAINER_DIRS = {
    "datasets": "dataset",
    "evals": "evaluator",
    "tools": "tool",
    "adapters": "adapter",
}


def find_container_dockerfiles(root: Path) -> List[tuple]:
    found = []
    for subdir, image_type in CONTAINER_DIRS.items():
        base = root / "containers" / subdir
        if not base.is_dir():
            continue
        for dockerfile in sorted(base.glob("*/Dockerfile")):
            found.append((dockerfile, image_type))
    return found


def _evaluator_config_by_dir(root: Path) -> Dict[str, List[dict]]:
    """containers/evals/<dir> -> evaluators.yaml entries whose image is evals/<dir>."""
    try:
        import yaml
    except ImportError:
        return {}
    path = root / "configs" / "evaluators.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    by_dir: Dict[str, List[dict]] = {}
    for key, entry in (data.get("evaluators") or {}).items():
        image = ((entry or {}).get("container") or {}).get("image", "")
        match = re.search(r"/evals/([^:/@]+)", image)
        if match:
            by_dir.setdefault(match.group(1), []).append(dict(entry, _key=key))
    return by_dir


def _cross_check_evaluator(report: LabelReport, labels: Dict[str, str], entries: List[dict]) -> None:
    for entry in entries:
        for field_name in ("metrics", "required_artifacts", "defense_types"):
            in_label = set(_split_list(labels.get(NS + field_name, "")))
            in_config = {str(v) for v in (entry.get(field_name) or [])}
            if in_label != in_config:
                report.warnings.append(
                    f"{NS}{field_name} differs from configs/evaluators.yaml "
                    f"[{entry['_key']}]: label only {sorted(in_label - in_config)}, "
                    f"config only {sorted(in_config - in_label)}"
                )


def lint_repo(root: Path) -> List[LabelReport]:
    evaluator_config = _evaluator_config_by_dir(root)
    reports = []
    for path, image_type in find_container_dockerfiles(root):
        report = lint_dockerfile(path, image_type)
        report.subject = str(path.relative_to(root))
        if image_type == "evaluator":
            labels = parse_dockerfile_labels(path.read_text())
            _cross_check_evaluator(report, labels, evaluator_config.get(path.parent.name, []))
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# Published image audit
# ---------------------------------------------------------------------------

def _config_images(root: Path) -> Dict[str, str]:
    """image ref -> expected type, from configs/{tools,datasets,evaluators}.yaml."""
    import yaml

    images: Dict[str, str] = {}
    sources = [
        ("tools.yaml", "tools", "tool"),
        ("tools.yaml", "adapters", "adapter"),
        ("evaluators.yaml", "evaluators", "evaluator"),
    ]
    for filename, section, image_type in sources:
        path = root / "configs" / filename
        if not path.exists():
            continue
        data = yaml.safe_load(path.read_text()) or {}
        for entry in (data.get(section) or {}).values():
            image = ((entry or {}).get("container") or {}).get("image")
            if image:
                images[image] = image_type
    path = root / "configs" / "datasets.yaml"
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        for entry in (data.get("datasets") or {}).values():
            entry = entry or {}
            refs = [entry.get("default_image")]
            refs += [(v or {}).get("image") for v in (entry.get("variants") or {}).values()]
            for image in refs:
                if image:
                    images[image] = "dataset"
    return images


def audit_images(root: Path, images: Optional[Dict[str, str]] = None) -> List[LabelReport]:
    try:
        from .container_labels import get_container_labels_for_image
    except ImportError:  # run as a script: python src/pipeline/image_labels.py
        from container_labels import get_container_labels_for_image

    images = images if images is not None else _config_images(root)
    reports = []
    for image, image_type in sorted(images.items()):
        labels = get_container_labels_for_image(image)
        if not labels:
            report = LabelReport(image, errors=["no labels found (image missing or unreachable)"])
        else:
            report = validate_labels(labels, image, built=True, expected_type=image_type)
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_reports(reports: List[LabelReport], as_json: bool) -> int:
    if as_json:
        print(json.dumps(
            [{"subject": r.subject, "ok": r.ok, "errors": r.errors, "warnings": r.warnings}
             for r in reports],
            indent=2,
        ))
    else:
        for r in reports:
            status = "ok  " if r.ok else "FAIL"
            print(f"{status} {r.subject}")
            for e in r.errors:
                print(f"       error: {e}")
            for w in r.warnings:
                print(f"       warn:  {w}")
        failed = sum(not r.ok for r in reports)
        print(f"\n{len(reports) - failed}/{len(reports)} passed")
    return 1 if any(not r.ok for r in reports) else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="image_labels")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lint = sub.add_parser("lint", help="check Dockerfile labels under containers/ (or given files)")
    p_lint.add_argument("dockerfiles", nargs="*", type=Path)
    p_lint.add_argument("--type", choices=sorted(TYPE_REQUIRED), help="expected type for given files")

    p_audit = sub.add_parser("audit", help="check labels on published or local images")
    p_audit.add_argument("images", nargs="*", help="image refs (default: all images in configs/)")
    p_audit.add_argument("--type", choices=sorted(TYPE_REQUIRED), help="expected type for given images")

    for p in (p_lint, p_audit):
        p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
        p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "lint":
        if args.dockerfiles:
            reports = [lint_dockerfile(p, args.type) for p in args.dockerfiles]
        else:
            reports = lint_repo(args.root)
    else:
        images = {i: args.type for i in args.images} if args.images else None
        reports = audit_images(args.root, images)
    return _print_reports(reports, args.json)


if __name__ == "__main__":
    sys.exit(main())
