"""
De-duplication against a real MinIO server, real processes, a real second
filesystem and a real container. Opt-in; each part skips without its setup.

    LANDSEER_MINIO_TEST_ENDPOINT=127.0.0.1:19000   (a throwaway server: a fresh
    LANDSEER_MINIO_TEST_ACCESS_KEY=...              bucket is created per run
    LANDSEER_MINIO_TEST_SECRET_KEY=...              and removed afterwards)
    LANDSEER_DEDUP_XFS_DIR=/data/...   a dir on another filesystem than tmp
    LANDSEER_RUN_DOCKER_RO_TEST=1      runs alpine with the worker's mounts

Covers:
- dataset + ancestor chain round trip through MinIO: left-out files have no
  objects, outputs are byte-identical, rebuilt files are hard links
- a reader that predates de-dup (c60c7d0) misses Q and R and hits P
- four processes downloading the same artifact into one cache dir
- dataset on another filesystem: real EXDEV, files copied
- a tool cannot write a hard-linked input through the worker's docker mounts
"""

import json
import multiprocessing
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from src.common import dataset_manifest as dsm
from src.store.artifact_manager import ARTIFACT_SOURCE, DATASET_SOURCE, ArtifactManager
from src.store.minio_store import ARTIFACT_COMPLETE_MARKER, MinioConfig, MinioStore
from tests.store.test_artifact_dataset_dedup import LOCATION, _files, _legacy_manager_class

ENDPOINT = os.environ.get("LANDSEER_MINIO_TEST_ENDPOINT")
P, Q, R = "p" * 64, "q" * 64, "r" * 64

needs_minio = pytest.mark.skipif(not ENDPOINT, reason="set LANDSEER_MINIO_TEST_ENDPOINT to a throwaway MinIO")


def _config(bucket):
    return MinioConfig(
        endpoint=ENDPOINT,
        access_key=os.environ.get("LANDSEER_MINIO_TEST_ACCESS_KEY", "minioadmin"),
        secret_key=os.environ.get("LANDSEER_MINIO_TEST_SECRET_KEY", "minioadmin"),
        bucket=bucket,
        secure=False,
    )


@pytest.fixture
def bucket():
    name = f"dedup-test-{uuid.uuid4().hex[:12]}"
    store = MinioStore(_config(name))
    assert store.is_available
    yield name
    for obj in list(store.list_objects("", recursive=True)):
        store.delete(obj.object_name)
    store._client.remove_bucket(name)


def _manager(cache: Path, bucket: str) -> ArtifactManager:
    return ArtifactManager(local_cache_dir=cache, minio_store=MinioStore(_config(bucket)), use_minio=True)


def _make_dataset(root: Path):
    (root / "img_align_celeba").mkdir(parents=True)
    (root / "data.npy").write_bytes(os.urandom(300_000))
    (root / "labels.npy").write_bytes(os.urandom(4_000))
    for i in range(500):
        (root / "img_align_celeba" / f"{i:06d}.jpg").write_bytes(os.urandom(200 + i))
    manifest = dsm.build_manifest(root)
    dsm.write_manifest(root, manifest)
    return root, manifest


def _make_chain(tmp: Path, ds: Path):
    """P filters data.npy and adds wm; Q trains; R is a noop passing everything on."""
    ignore = shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR)
    p = tmp / "P"
    shutil.copytree(ds, p, ignore=ignore)
    (p / "data.npy").write_bytes(os.urandom(200_000))
    (p / "wm_test_data.npy").write_bytes(os.urandom(150_000))
    q = tmp / "Q"
    shutil.copytree(p, q)
    (q / "wm_test_data.npy").unlink()
    (q / "model.pt").write_bytes(os.urandom(50_000))
    r = tmp / "R"
    shutil.copytree(q, r)
    shutil.copy2(p / "wm_test_data.npy", r / "wm_test_data.npy")
    (r / "img_align_celeba" / "000003.jpg").write_bytes(b"triggered")
    return {P: (p, []), Q: (q, [P]), R: (r, [Q])}


def _object_names(bucket, key):
    store = MinioStore(_config(bucket))
    prefix = store.get_artifact_key(key)
    return {o.object_name[len(prefix):] for o in store.list_objects(prefix, recursive=True)}


@pytest.fixture
def published(tmp_path, bucket):
    ds, manifest = _make_dataset(tmp_path / "ds")
    chain = _make_chain(tmp_path, ds)
    writer = _manager(tmp_path / "writer", bucket)
    writer.register_dataset(ds, manifest, LOCATION)
    for key in (P, Q, R):
        writer.store_artifact(key, chain[key][0], key[0], "tool", parent_hashes=chain[key][1])
    return ds, manifest, chain


@needs_minio
def test_chain_round_trip_through_minio(tmp_path, bucket, published):
    ds, manifest, chain = published
    assert _object_names(bucket, P) >= {"data.npy", "wm_test_data.npy"}
    assert not any(n.startswith("img_align_celeba/") for n in _object_names(bucket, P))
    assert _object_names(bucket, Q) == {"model.pt", "manifest.json", ARTIFACT_COMPLETE_MARKER}
    assert _object_names(bucket, R) == {"img_align_celeba/000003.jpg", "manifest.json", ARTIFACT_COMPLETE_MARKER}

    store = MinioStore(_config(bucket))
    marker = json.loads(store.download_bytes(f"{store.get_artifact_key(R)}{ARTIFACT_COMPLETE_MARKER}"))
    assert marker["files"]["data.npy"] == {
        **marker["files"]["data.npy"], "source": ARTIFACT_SOURCE, "artifact_cache_key": P}
    assert marker["files"]["labels.npy"]["source"] == DATASET_SOURCE

    reader = _manager(tmp_path / "reader", bucket)
    reader.register_dataset(ds, manifest, LOCATION, active=False)
    out = reader.get_artifact(R)
    assert out is not None and _files(out) == _files(chain[R][0])
    p_out = reader.get_local_path(P) / "output"
    assert os.stat(out / "data.npy").st_ino == os.stat(p_out / "data.npy").st_ino
    assert os.stat(out / "labels.npy").st_ino == os.stat(ds / "labels.npy").st_ino
    for key in (P, Q):
        assert _files(reader.get_artifact(key)) == _files(chain[key][0])


@needs_minio
def test_old_reader_through_minio(tmp_path, bucket, published):
    legacy = _legacy_manager_class(tmp_path)
    old = legacy(local_cache_dir=tmp_path / "old", minio_store=MinioStore(_config(bucket)), use_minio=True)
    assert old.get_artifact(Q) is None
    assert old.get_artifact(R) is None
    # P left out only dataset files: old readers miss it too.
    assert old.get_artifact(P) is None
    assert not [p for p in (tmp_path / "old").iterdir() if p.name.startswith(".tmp-")]


def _download_in_process(cache, bucket, ds, manifest, queue):
    reader = _manager(Path(cache), bucket)
    reader.register_dataset(Path(ds), manifest, LOCATION, active=False)
    out = reader.get_artifact(R)
    queue.put(_files(out) if out else None)


@needs_minio
def test_processes_racing_on_one_cache(tmp_path, bucket, published):
    ds, manifest, chain = published
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_download_in_process, args=(str(tmp_path / "shared"), bucket, str(ds), manifest, queue))
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    results = [queue.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    expected = _files(chain[R][0])
    assert all(r == expected for r in results)
    assert not [p for p in (tmp_path / "shared").iterdir() if p.name.startswith(".tmp-")]


@needs_minio
def test_dataset_on_other_filesystem(tmp_path, bucket):
    xfs = os.environ.get("LANDSEER_DEDUP_XFS_DIR")
    if not xfs or os.stat(xfs).st_dev == os.stat(tmp_path).st_dev:
        pytest.skip("set LANDSEER_DEDUP_XFS_DIR to a dir on another filesystem than tmp")
    root = Path(xfs) / f"dedup-xfs-{uuid.uuid4().hex[:8]}"
    try:
        ds, manifest = _make_dataset(root / "ds")
        out = tmp_path / "out"
        shutil.copytree(ds, out, ignore=shutil.ignore_patterns(dsm.MANIFEST_NAME, dsm.HASH_SIDECAR))
        (out / "model.pt").write_bytes(b"m")
        writer = _manager(tmp_path / "writer", bucket)
        writer.register_dataset(ds, manifest, LOCATION)
        writer.store_artifact(P, out, "p", "tool")

        reader = _manager(tmp_path / "reader", bucket)
        reader.register_dataset(ds, manifest, LOCATION, active=False)
        got = reader.get_artifact(P)
        assert got is not None and _files(got) == _files(out)
        assert os.stat(got / "labels.npy").st_dev != os.stat(ds / "labels.npy").st_dev
        assert os.stat(ds / "labels.npy").st_nlink == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.mark.skipif(os.environ.get("LANDSEER_RUN_DOCKER_RO_TEST") != "1", reason="set LANDSEER_RUN_DOCKER_RO_TEST=1")
def test_container_cannot_write_hard_linked_input(tmp_path):
    from src.worker.runner import DockerRunner

    cache_file = tmp_path / "cache" / "labels.npy"
    cache_file.parent.mkdir()
    cache_file.write_bytes(b"original")
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    os.link(cache_file, input_dir / "labels.npy")
    os.chmod(input_dir, 0o777)
    output_dir = tmp_path / "output"
    # The runner splits the command on whitespace, so the shell steps live in a script.
    (input_dir / "try_write.sh").write_text(
        "echo changed > /input/labels.npy\n"
        "echo changed > /data/labels.npy\n"
        "echo ok > /output/done\n"
    )

    runner = DockerRunner(workspace_dir=tmp_path / "ws", timeout=120)
    result = runner.run(
        image="alpine:latest",
        command="sh /input/try_write.sh",
        input_dir=input_dir,
        output_dir=output_dir,
    )
    assert cache_file.read_bytes() == b"original"
    assert (output_dir / "done").exists(), result
    assert result[1].count("Read-only file system") == 2, result[1]
    subprocess.run(["docker", "run", "--rm", "-v", f"{output_dir}:/o", "alpine:latest", "rm", "-rf", "/o/done"],
                   check=False, capture_output=True)
