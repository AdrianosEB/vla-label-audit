"""Turn LIBERO's 546,930 AV1-encoded frames into cached image embeddings, once.

Session 2 needs a joint vector index over LIBERO images and their language
labels in order to ask whether an instruction actually describes the trajectory
it is attached to. Everything downstream of that -- nearest neighbours,
per-episode mismatch scores, the ranked worklist -- reads embeddings, never
pixels. So the pixels get read exactly once, here, and the vectors go to disk.

Three facts about this machine shape the whole design:

* **M2 has no AV1 hardware decode** (Apple added it in M3). Decoding is
  software, via libdav1d in PyAV. Decoding the corpus twice is a waste measured
  in tens of minutes, so this script is resumable at video-file granularity and
  writes straight into a memmap: a crash costs one file, not the run.
* **Sequential decode is ~200x cheaper per frame than seeking.** Measured on
  this dataset: a full-file linear decode runs at ~2,300-2,500 img/s, while
  seeking to a keyframe per frame collapses to low tens. Each mp4 concatenates
  ~10k frames from ~35 episodes back to back, so the loop below opens each file
  once, decodes every frame in presentation order, and maps decode position to
  dataset row using the episodes-meta ``from_timestamp`` ranges. Seeking is used
  only in ``--verify``, where it is the *independent* implementation whose job is
  to disagree if the fast path drifted.
* **16 GB of unified memory is shared with the GPU** and
  ``is_amp_available("mps")`` is False, so this is fp32 with a modest batch.

Two encoders, deliberately chosen to fail differently:

* ``dinov2`` -- facebook/dinov2-small, 384-d. Self-supervised, vision-only. The
  primary instrument: it has never seen a caption, so it cannot inherit a
  language prior from the same distribution the labels came from.
* ``clip`` -- openai/clip-vit-base-patch32 image tower, 512-d. Language-
  supervised, i.e. the opposite training recipe. If both encoders flag the same
  episodes, the finding is not an artefact of either one's geometry.

Because MPS has a documented class of silent-garbage bugs (lerobot#496), the
device is never trusted on faith: ``--check-mps`` re-embeds a sample on CPU and
compares, and refuses to bless the run if cosine similarity drops below
``MPS_COSINE_FLOOR``.

Run:
    python scripts/libero_embed.py --check-mps --encoder dinov2
    python scripts/libero_embed.py --encoder dinov2      # both cameras, resumable
    python scripts/libero_embed.py --encoder clip
    python scripts/libero_embed.py --verify --encoder dinov2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data"
LIBERO = CACHE / "libero"

# facebook/dinov2-small is the ViT-S/14 checkpoint chosen in CLAUDE.md over
# ViT-B/14: 384-d halves both embed time and index memory, and at 273k rows the
# exact-search index has to fit alongside the GPU's own allocations.
ENCODERS = {
    "dinov2": "facebook/dinov2-small",
    "clip": "openai/clip-vit-base-patch32",
}
CAMERAS = ["observation.images.image", "observation.images.image2"]

# 128 is where MPS throughput plateaus for both towers (measured: dinov2 117-123
# img/s at 64/128/256, clip 205 at 64 and 202 at 128). Bigger batches buy
# nothing and only raise the peak allocation on memory the GPU shares with the
# rest of the machine.
BATCH_SIZE = 128

# Decoded frames are handed over in chunks, not whole files. A whole mp4 is
# ~10k x 256 x 256 x 3 = 2.0 GB of uint8, and with one file in the queue, one
# being decoded and one being embedded that is ~6 GB of a 16 GB pool the GPU is
# also drawing on. 512-frame chunks cost 100 MB each, so a 4-deep queue keeps
# the GPU continuously fed for 0.4 GB.
DECODE_CHUNK = 512
DECODE_QUEUE_DEPTH = 4

MPS_COSINE_FLOOR = 0.9999
MPS_CHECK_N = 200


# --------------------------------------------------------------------------
# dataset identity
# --------------------------------------------------------------------------


def dataset_tag() -> str:
    """A content hash tying every cache file to this exact dataset revision.

    Hashing 1.94 GB of video on every run would cost more than the embedding
    does, so the digest covers the metadata that *defines* the layout --
    info.json, the episodes table, the task table -- plus the size in bytes of
    every video file. A re-download that changed any frame would change a file
    size or the episode boundaries; a hash collision would require the new
    dataset to have byte-identical metadata and byte-identical file lengths, at
    which point it is the same dataset.
    """
    h = hashlib.sha256()
    for rel in ["meta/info.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"]:
        h.update((LIBERO / rel).read_bytes())
    for cam in CAMERAS:
        for p in sorted((LIBERO / "videos" / cam / "chunk-000").glob("*.mp4")):
            h.update(p.name.encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class Layout:
    """Everything needed to place a decoded frame in the output matrix.

    ``row_start`` is the global dataset index of an episode's first frame.
    Rows are therefore ordered by (episode_index, frame_index) ascending, which
    is the ordering the analysis pass is promised and the same ordering the
    per-frame parquet files use.
    """

    episodes: pd.DataFrame
    n_rows: int
    tasks: list[str]
    task_index: np.ndarray  # per episode


def load_layout() -> Layout:
    ep = pd.read_parquet(LIBERO / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ep = ep.sort_values("episode_index").reset_index(drop=True)

    info = json.loads((LIBERO / "meta" / "info.json").read_text())
    n_rows = int(info["total_frames"])

    starts = ep["dataset_from_index"].to_numpy()
    lengths = ep["length"].to_numpy()
    # The whole row-assignment scheme rests on this: global row index is the
    # running total of episode lengths. If the published dataset_from_index ever
    # disagreed with that, every embedding would land in the wrong row and
    # nothing downstream would notice.
    expected = np.concatenate([[0], np.cumsum(lengths)])[:-1]
    if not np.array_equal(starts, expected):
        raise SystemExit("dataset_from_index is not the cumulative sum of episode lengths")
    if int(lengths.sum()) != n_rows:
        raise SystemExit(f"episode lengths sum to {lengths.sum()}, info.json says {n_rows}")

    tasks_df = pd.read_parquet(LIBERO / "meta" / "tasks.parquet")
    # tasks.parquet is indexed *by the instruction string*, with task_index as
    # its only column -- the inverse of the obvious layout, so invert it here
    # rather than in four places downstream.
    instructions = [""] * len(tasks_df)
    for text, idx in zip(tasks_df.index.astype(str), tasks_df["task_index"].to_numpy()):
        instructions[int(idx)] = text

    return Layout(
        episodes=ep,
        n_rows=n_rows,
        tasks=instructions,
        task_index=episode_task_index(ep),
    )


def episode_task_index(ep: pd.DataFrame) -> np.ndarray:
    """One task_index per episode, read from the per-frame data parquets.

    The episodes table records video offsets but not the task, and the task only
    lives per frame. LIBERO gives one instruction per episode, so this reads the
    two index columns from each of the 377 data shards and asserts the episode's
    task is constant -- an episode with two tasks would mean the label this
    project is auditing is not even well defined for that trajectory.
    """
    out = np.full(len(ep), -1, dtype=np.int64)
    for path in sorted((LIBERO / "data" / "chunk-000").glob("*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index", "task_index"])
        grouped = df.groupby("episode_index")["task_index"].agg(["min", "max"])
        if (grouped["min"] != grouped["max"]).any():
            raise SystemExit(f"{path.name}: an episode carries more than one task_index")
        out[grouped.index.to_numpy()] = grouped["min"].to_numpy()
    if (out < 0).any():
        raise SystemExit(f"{int((out < 0).sum())} episodes had no rows in data/")
    return out


def file_plan(layout: Layout, camera: str) -> list[tuple[int, np.ndarray]]:
    """For each video file: the dataset rows its frames map to, in decode order.

    Returned as an explicit row-index array per file rather than a (start, stop)
    slice. The slice version would be faster and, on this dataset, identical --
    but it silently assumes the episodes sharing a video file are contiguous in
    episode_index, and an off-by-one in that assumption is exactly the bug this
    script is most likely to have. Building the mapping from each episode's own
    ``dataset_from_index`` cannot drift.
    """
    ep = layout.episodes
    fidx = ep[f"videos/{camera}/file_index"].to_numpy()
    fts = ep[f"videos/{camera}/from_timestamp"].to_numpy()
    starts = ep["dataset_from_index"].to_numpy()
    lengths = ep["length"].to_numpy()

    plan = []
    for f in np.unique(fidx):
        sel = np.flatnonzero(fidx == f)
        # Decode order is presentation order, so episodes must be visited in
        # from_timestamp order -- not episode_index order, which merely happens
        # to agree here.
        sel = sel[np.argsort(fts[sel], kind="stable")]
        offsets = np.round(fts[sel] * 10.0).astype(np.int64)
        if not np.array_equal(offsets, np.concatenate([[0], np.cumsum(lengths[sel])])[:-1]):
            raise SystemExit(f"{camera} file {f}: from_timestamps are not back-to-back")
        rows = np.concatenate([np.arange(s, s + n) for s, n in zip(starts[sel], lengths[sel])])
        plan.append((int(f), rows))
    return plan


# --------------------------------------------------------------------------
# decode
# --------------------------------------------------------------------------


def decode_file(path: Path, expected: int):
    """Every frame of one mp4, in presentation order, as uint8 NHWC chunks.

    Yields ``(offset, images)`` where offset is the frame's position within the
    file. Presentation timestamps are checked to be strictly increasing and
    evenly spaced: PyAV yields frames in decode order, which only coincides with
    presentation order when the stream has no reordering, and silently accepting
    a reordered stream would scramble frames within an episode.

    The total frame count is checked against the sum of the lengths of the
    episodes the metadata assigns to this file. A mismatch means the sequential
    mapping has drifted, and the only safe response is to stop -- a short file
    would shift every subsequent row by the deficit.
    """
    seen = 0
    buf: list[np.ndarray] = []
    last_pts = None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            if last_pts is not None and frame.pts <= last_pts:
                raise SystemExit(f"{path.name}: frames are not in presentation order at {seen}")
            last_pts = frame.pts
            buf.append(frame.to_ndarray(format="rgb24"))
            if len(buf) == DECODE_CHUNK:
                yield seen, np.stack(buf)
                seen += len(buf)
                buf = []
    if buf:
        yield seen, np.stack(buf)
        seen += len(buf)
    if seen != expected:
        raise SystemExit(f"{path.name}: decoded {seen} frames, metadata expects {expected}")


def decode_worker(work, out: queue.Queue) -> None:
    """Decode files ahead of the GPU on a background thread.

    PyAV releases the GIL inside libdav1d, so this genuinely overlaps: decode is
    ~2,400 img/s against ~120 img/s for the ViT forward, which turns four
    minutes of otherwise-serial decode into zero. An ``("eof", f)`` marker
    follows each file so the consumer knows when it may record that file as
    complete; recording it earlier would let a resume skip a half-written file.
    """
    try:
        for file_index, path, rows in work:
            for offset, images in decode_file(path, len(rows)):
                out.put(("chunk", file_index, rows[offset : offset + len(images)], images))
            out.put(("eof", file_index, None, None))
    except BaseException as exc:  # surfaced on the consumer side, never swallowed
        out.put(exc)
    else:
        out.put(None)


def seek_frame(path: Path, timestamp: float) -> np.ndarray:
    """One frame at a wall-clock offset, decoded by seeking -- the slow path.

    Used only by ``--verify``. It shares no code with the sequential reader on
    purpose: agreement between the two is evidence, agreement between a function
    and itself is not.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        target = int(round(timestamp / float(stream.time_base)))
        container.seek(target, stream=stream, backward=True)
        for frame in container.decode(video=0):
            if frame.pts >= target:
                return frame.to_ndarray(format="rgb24")
    raise SystemExit(f"{path.name}: no frame at t={timestamp}")


# --------------------------------------------------------------------------
# embedding
# --------------------------------------------------------------------------


class Embedder:
    """One encoder pinned to one device, exposing a single uint8-batch call.

    Preprocessing runs on CPU even when the model is on MPS. The resize is
    bicubic, and MPS's bicubic disagrees with the CPU kernel by up to ~0.35 in
    normalised units on adversarial input -- small, but it would make the
    MPS-vs-CPU check measure the resampler instead of the model, which is the
    thing the check exists to interrogate.
    """

    def __init__(self, key: str, device: str) -> None:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModelWithProjection

        self.key = key
        self.model_id = ENCODERS[key]
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id, backend="torchvision")
        loader = CLIPVisionModelWithProjection if key == "clip" else AutoModel
        self.model = loader.from_pretrained(self.model_id, dtype=torch.float32).to(device).eval()
        self.dim = 512 if key == "clip" else int(self.model.config.hidden_size)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Embed an NHWC uint8 batch to (N, dim) float32.

        One vector per image: CLIP's projected ``image_embeds`` (what
        ``get_image_features`` returns) and DINOv2's ``pooler_output``, which is
        the CLS token after the final layernorm -- the canonical DINOv2 global
        descriptor. Not L2-normalised here; normalisation is a decision for the
        index, and a cache that has already been normalised cannot be un-.
        """
        out = np.empty((len(images), self.dim), dtype=np.float32)
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        with torch.inference_mode():
            for i in range(0, len(images), BATCH_SIZE):
                chunk = tensor[i : i + BATCH_SIZE]
                px = self.processor(images=chunk, return_tensors="pt")["pixel_values"]
                res = self.model(pixel_values=px.to(self.device))
                vec = res.image_embeds if self.key == "clip" else res.pooler_output
                out[i : i + len(chunk)] = vec.float().cpu().numpy()
        return out


# --------------------------------------------------------------------------
# cache paths
# --------------------------------------------------------------------------


def cam_slug(camera: str) -> str:
    return camera.rsplit(".", 1)[-1]


def cache_path(key: str, camera: str, n_rows: int, tag: str) -> Path:
    return CACHE / f"libero_emb_{key}_{cam_slug(camera)}_{n_rows}_{tag}.npy"


def manifest_path(n_rows: int, tag: str) -> Path:
    return CACHE / f"libero_index_{n_rows}_{tag}.npz"


def write_manifest(layout: Layout, tag: str) -> Path:
    """Per-row episode/frame/task alignment, so no consumer re-derives it.

    Row order here *is* the row order of every embedding matrix. Shipping the
    instruction strings alongside the indices means the analysis pass never has
    to reopen the LIBERO metadata, and therefore cannot reopen it differently.
    """
    path = manifest_path(layout.n_rows, tag)
    if path.exists():
        return path
    lengths = layout.episodes["length"].to_numpy()
    episode_index = np.repeat(layout.episodes["episode_index"].to_numpy(), lengths).astype(np.int32)
    frame_index = np.concatenate([np.arange(n) for n in lengths]).astype(np.int32)
    np.savez(
        path,
        episode_index=episode_index,
        frame_index=frame_index,
        task_index=layout.task_index[episode_index].astype(np.int32),
        episode_task_index=layout.task_index.astype(np.int32),
        episode_length=lengths.astype(np.int32),
        episode_row_start=layout.episodes["dataset_from_index"].to_numpy().astype(np.int64),
        tasks=np.array(layout.tasks, dtype=object),
        dataset_tag=np.array(tag),
    )
    return path


# --------------------------------------------------------------------------
# the pass
# --------------------------------------------------------------------------


def embed_camera(key: str, camera: str, layout: Layout, tag: str, device: str) -> Path:
    """Embed every frame of one camera, resumable at video-file granularity.

    Work lands in a ``.part`` memmap with a sibling progress file listing the
    video files already written. Interrupting the run costs at most one file
    (~90 seconds of GPU), and rerunning the command picks up where it stopped --
    which matters because a full pass is over an hour and this machine is also
    someone's laptop.
    """
    final = cache_path(key, camera, layout.n_rows, tag)
    if final.exists():
        print(f"  [{key}/{cam_slug(camera)}] cached at {final.name}")
        return final

    embedder = Embedder(key, device)
    part = final.with_suffix(".part.npy")
    progress = final.with_suffix(".progress.json")
    done: set[int] = set()
    if part.exists() and progress.exists():
        done = set(json.loads(progress.read_text())["files_done"])
        print(f"  [{key}/{cam_slug(camera)}] resuming, {len(done)} video files already embedded")
    mm = np.lib.format.open_memmap(
        part, mode="r+" if part.exists() else "w+", dtype=np.float32,
        shape=(layout.n_rows, embedder.dim),
    )

    plan = [(f, rows) for f, rows in file_plan(layout, camera) if f not in done]
    video_dir = LIBERO / "videos" / camera / "chunk-000"
    work = [(f, video_dir / f"file-{f:03d}.mp4", rows) for f, rows in plan]

    q: queue.Queue = queue.Queue(maxsize=DECODE_QUEUE_DEPTH)
    producer = threading.Thread(target=decode_worker, args=(work, q), daemon=True)
    producer.start()

    t0 = time.perf_counter()
    n_done = 0
    total = sum(len(rows) for _, rows in plan)
    while True:
        item = q.get()
        if isinstance(item, BaseException):
            raise item
        if item is None:
            break
        kind, f, rows, images = item
        if kind == "chunk":
            mm[rows] = embedder(images)
            del images
            n_done += len(rows)
            continue
        mm.flush()
        done.add(f)
        progress.write_text(json.dumps({"files_done": sorted(done)}))
        rate = n_done / (time.perf_counter() - t0)
        eta = (total - n_done) / rate if rate else 0.0
        print(
            f"  [{key}/{cam_slug(camera)}] file {f:3d}  {n_done:>7,}/{total:,} frames"
            f"  {rate:6.1f} img/s  eta {eta / 60:5.1f} min",
            flush=True,
        )

    if n_done != total:
        raise SystemExit(f"embedded {n_done} frames, planned {total}")
    wall = time.perf_counter() - t0
    del mm
    part.rename(final)
    progress.unlink(missing_ok=True)

    sidecar = final.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "encoder": key,
                "model_id": embedder.model_id,
                "dims": embedder.dim,
                "camera": camera,
                "n_rows": layout.n_rows,
                "dtype": "float32",
                "batch_size": BATCH_SIZE,
                "device": device,
                # Scoped to *this* invocation: a resumed run only embeds the
                # files a previous run did not, so reporting these as the cost
                # of the whole camera would understate it. frames_embedded_this_run
                # is the denominator that makes them interpretable.
                "wall_seconds_this_run": round(wall, 1),
                "images_per_second_this_run": round(n_done / wall, 1) if wall else None,
                "frames_embedded_this_run": n_done,
                "resumed": n_done != layout.n_rows,
                "row_order": "(episode_index, frame_index) ascending",
                "pooling": "image_embeds" if key == "clip" else "pooler_output (CLS post-LN)",
                "l2_normalised": False,
                "dataset_tag": tag,
                "manifest": manifest_path(layout.n_rows, tag).name,
            },
            indent=2,
        )
    )
    print(f"  [{key}/{cam_slug(camera)}] wrote {final.name} in {wall / 60:.1f} min")
    return final


# --------------------------------------------------------------------------
# the two things that must be checked before the cache is trusted
# --------------------------------------------------------------------------


def check_mps(key: str, layout: Layout) -> None:
    """Embed the same images on MPS and CPU and refuse to proceed if they differ.

    lerobot#496 is the reference case: an MPS transfer that returned garbage
    with no error. That failure mode is silent, so it has to be tested for
    rather than waited for. fp32 reassociation between two backends is worth a
    few 1e-4 absolute; anything that moves cosine below MPS_COSINE_FLOOR is not
    numerics.
    """
    cam = CAMERAS[0]
    path = LIBERO / "videos" / cam / "chunk-000" / "file-000.mp4"
    with av.open(str(path)) as container:
        images = []
        for frame in container.decode(video=0):
            images.append(frame.to_ndarray(format="rgb24"))
            if len(images) >= MPS_CHECK_N:
                break
    images = np.stack(images)

    print(f"[mps-check] {key}: {len(images)} images, mps vs cpu")
    a = Embedder(key, "mps")(images)
    b = Embedder(key, "cpu")(images)
    max_abs = float(np.abs(a - b).max())
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    min_cos = float(cos.min())
    print(f"  max |mps - cpu| = {max_abs:.3e}")
    print(f"  min cosine      = {min_cos:.8f}")
    if min_cos < MPS_COSINE_FLOOR:
        raise SystemExit(
            f"MPS and CPU disagree (min cosine {min_cos:.6f} < {MPS_COSINE_FLOOR}). "
            "Do not ship this cache."
        )
    print("  verdict: agree to fp32 noise")


def verify_alignment(key: str, layout: Layout, tag: str, n: int, device: str) -> None:
    """Re-derive a handful of cached rows by the independent seek path.

    This is the guard against the one bug that would poison everything without
    showing up anywhere else: an off-by-one in mapping sequential decode
    position to dataset row. A shifted cache still has the right shape, the
    right norms, and plausible neighbours -- it is only wrong. Picks are spread
    across video files, across episodes, and cover both cameras.
    """
    rng = np.random.default_rng(0)
    ep = layout.episodes
    embedders = {cam: None for cam in CAMERAS}

    picks = []
    for i in range(n):
        cam = CAMERAS[i % len(CAMERAS)]
        # Spread over the file axis rather than the episode axis so consecutive
        # picks cannot all land in one mp4.
        fidx = ep[f"videos/{cam}/file_index"].to_numpy()
        target_file = sorted(np.unique(fidx))[int(i * (len(np.unique(fidx)) - 1) / max(n - 1, 1))]
        candidates = np.flatnonzero(fidx == target_file)
        e = int(rng.choice(candidates))
        f = int(rng.integers(0, ep["length"].to_numpy()[e]))
        picks.append((cam, e, f))

    print(f"[verify] {key}: {len(picks)} spot-checks against the seek path")
    for cam, e, f in picks:
        mat = np.load(cache_path(key, cam, layout.n_rows, tag), mmap_mode="r")
        row = int(ep["dataset_from_index"].to_numpy()[e]) + f
        cached = np.asarray(mat[row], dtype=np.float32)

        file_index = int(ep[f"videos/{cam}/file_index"].to_numpy()[e])
        t = float(ep[f"videos/{cam}/from_timestamp"].to_numpy()[e]) + f / 10.0
        img = seek_frame(LIBERO / "videos" / cam / "chunk-000" / f"file-{file_index:03d}.mp4", t)

        if embedders[cam] is None:
            embedders[cam] = Embedder(key, device)
        fresh = embedders[cam](img[None])[0]
        cos = float(
            fresh @ cached / (np.linalg.norm(fresh) * np.linalg.norm(cached) + 1e-12)
        )
        flag = "ok" if cos > 0.9999 else "MISMATCH"
        print(
            f"  {cam_slug(cam):<7} ep {e:>5} frame {f:>4} (mp4 {file_index:>3}, row {row:>7})"
            f"  cosine {cos:.6f}  {flag}"
        )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    ap.add_argument("--camera", choices=CAMERAS, default=None, help="default: both")
    ap.add_argument("--check-mps", action="store_true", help="MPS vs CPU sanity check, then exit")
    ap.add_argument("--verify", action="store_true", help="spot-check cached rows, then exit")
    ap.add_argument("--spot-checks", type=int, default=5)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    layout = load_layout()
    tag = dataset_tag()
    print(f"libero: {layout.n_rows:,} frames x {len(CAMERAS)} cameras, dataset tag {tag}")

    if args.check_mps:
        check_mps(args.encoder, layout)
        return

    if args.verify:
        verify_alignment(args.encoder, layout, tag, args.spot_checks, args.device)
        return

    print(f"manifest: {write_manifest(layout, tag).name}")
    cameras = [args.camera] if args.camera else CAMERAS
    for cam in cameras:
        embed_camera(args.encoder, cam, layout, tag, args.device)


if __name__ == "__main__":
    main()
