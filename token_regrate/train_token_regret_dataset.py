from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import io
import json
import os
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image

try:
    import webdataset as wds
except Exception:
    wds = None


def _default_hf_shard_urls(num_shards=69):
    """Build default Hugging Face WebDataset shard URLs."""
    base_url = (
        "https://huggingface.co/datasets/ProGamerGov/"
        "synthetic-dataset-1m-high-quality-captions/"
        "resolve/main/data/data-{i:06d}.tar"
    )
    return [base_url.format(i=i) for i in range(int(num_shards))]


def _default_cc12m_tsv_path():
    """Return default CC12M TSV manifest location."""
    return os.path.join("dataset", "cc12m.tsv")


def _default_cc12m_cache_dir():
    """Return default on-disk cache directory for downloaded CC12M images."""
    return os.path.join("dataset", "cc12m_image_cache")


def _looks_like_cc12m_manifest(source):
    """Check whether a path-like string looks like a TSV/CSV manifest."""
    if source is None:
        return False
    return str(source).strip().lower().endswith((".tsv", ".csv"))


def _split_train_data_sources(sources):
    """Split user-provided data sources into CC12M manifest and HF shard URLs."""
    cc12m_tsv = None
    hf_urls = []
    for source in (sources or []):
        item = str(source).strip()
        if not item:
            continue
        if _looks_like_cc12m_manifest(item):
            if cc12m_tsv is None:
                cc12m_tsv = item
            elif os.path.abspath(cc12m_tsv) != os.path.abspath(item):
                raise ValueError(f"Multiple CC12M manifests provided: {cc12m_tsv!r} and {item!r}")
            continue
        hf_urls.append(item)
    return {"cc12m_tsv": cc12m_tsv, "hf_urls": hf_urls}


def _count_valid_cc12m_rows(tsv_path):
    """Count non-empty CC12M entries that include URL and caption."""
    total = 0
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            image_url, caption = line.split("\t", 1)
            if image_url.strip() and caption.strip():
                total += 1
    return total


def _infer_hf_example_count(urls):
    """Infer HF WebDataset example count when using the known default shard set."""
    if urls is None or len(urls) == 0:
        return None
    try:
        default_urls = _default_hf_shard_urls()
        if len(urls) == len(default_urls) and all(str(a) == str(b) for a, b in zip(urls, default_urls)):
            return 1_000_000
    except Exception:
        pass
    return None


def _extract_caption_from_sample(sample):
    """Extract a caption string from common WebDataset fields."""
    for key in ("txt", "caption", "text", "prompt"):
        val = sample.get(key)
        if isinstance(val, str) and val.strip() != "":
            return val.strip()

    meta = sample.get("json")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    if isinstance(meta, dict):
        for key in ("short_caption", "long_caption", "caption", "text", "prompt"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip() != "":
                return val.strip()
    return None


def _extract_pil_image_from_sample(sample):
    """Extract and convert an image payload from a WebDataset sample."""
    for key in ("jpg", "jpeg", "png", "webp", "image"):
        if key not in sample:
            continue
        obj = sample[key]
        if isinstance(obj, Image.Image):
            return obj.convert("RGB")
        if isinstance(obj, dict) and "bytes" in obj and obj["bytes"] is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
    return None


def _ensure_cc12m_cache_dir(cache_dir):
    """Create and return the CC12M cache directory path."""
    cache_dir = str(cache_dir).strip() if cache_dir is not None else ""
    if not cache_dir:
        cache_dir = _default_cc12m_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _cc12m_cache_path(image_url, cache_dir):
    """Map an image URL to a deterministic cache file path."""
    parsed = urlparse(str(image_url))
    ext = os.path.splitext(parsed.path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"):
        ext = ".img"
    key = hashlib.sha1(str(image_url).encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, key + ext)


def _fetch_cc12m_image(image_url, timeout=30, retries=3, cache_dir=None, use_cache=True):
    """Download a CC12M image with retries and optional local caching."""
    cache_path = None
    if bool(use_cache):
        cache_dir = _ensure_cc12m_cache_dir(cache_dir)
        cache_path = _cc12m_cache_path(image_url, cache_dir)
        if os.path.isfile(cache_path):
            try:
                with Image.open(cache_path) as img:
                    return img.convert("RGB")
            except Exception:
                try:
                    os.remove(cache_path)
                except Exception:
                    pass

    last_error = None
    request = Request(image_url, headers={"User-Agent": "Mozilla/5.0"})
    for _ in range(max(1, int(retries))):
        try:
            with urlopen(request, timeout=float(timeout)) as response:
                image_bytes = response.read()
            if cache_path is not None:
                try:
                    tmp_path = cache_path + ".tmp"
                    with open(tmp_path, "wb") as f:
                        f.write(image_bytes)
                    os.replace(tmp_path, cache_path)
                except Exception:
                    pass
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.convert("RGB")
        except Exception as exc:
            last_error = exc
    raise last_error


def _iter_cc12m_tsv_batches_parallel(
    tsv_path,
    batch_size,
    rank=0,
    world_size=1,
    cache_dir=None,
    cache_images=True,
    loader_workers=16,
    max_pending=0,
):
    """Stream (images, captions) batches from CC12M TSV using parallel downloads."""
    batch_images = []
    batch_captions = []
    rank = int(rank)
    world_size = max(1, int(world_size))
    cache_dir = _ensure_cc12m_cache_dir(cache_dir) if bool(cache_images) else None
    loader_workers = max(1, int(loader_workers))
    if int(max_pending) <= 0:
        max_pending = loader_workers * 4
    max_pending = max(loader_workers, int(max_pending))

    def _load_image_task(image_url):
        """Worker task: fetch a single image URL and return PIL image or None."""
        try:
            image = _fetch_cc12m_image(image_url, cache_dir=cache_dir, use_cache=bool(cache_images))
            return image
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=loader_workers) as pool:
        futures = {}

        def _drain_done(wait_mode):
            """Collect completed futures and append successful items to current batch."""
            nonlocal batch_images, batch_captions
            if not futures:
                return
            done, _ = wait(list(futures.keys()), return_when=wait_mode)
            for fut in done:
                caption = futures.pop(fut)
                image = fut.result()
                if image is None:
                    continue
                batch_images.append(image)
                batch_captions.append(caption)

        with open(tsv_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                # Deterministic line-level sharding so each rank processes a disjoint subset.
                if world_size > 1 and (line_idx % world_size) != rank:
                    continue
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                image_url, caption = line.split("\t", 1)
                image_url = image_url.strip()
                caption = caption.strip()
                if not image_url or not caption:
                    continue

                fut = pool.submit(_load_image_task, image_url)
                futures[fut] = caption

                if len(futures) >= max_pending:
                    # Backpressure: retire completed downloads before queueing more work.
                    _drain_done(FIRST_COMPLETED)

                while len(batch_images) >= int(batch_size):
                    out = (batch_images[: int(batch_size)], batch_captions[: int(batch_size)])
                    batch_images = batch_images[int(batch_size):]
                    batch_captions = batch_captions[int(batch_size):]
                    yield out

        while futures:
            _drain_done(FIRST_COMPLETED)
            while len(batch_images) >= int(batch_size):
                out = (batch_images[: int(batch_size)], batch_captions[: int(batch_size)])
                batch_images = batch_images[int(batch_size):]
                batch_captions = batch_captions[int(batch_size):]
                yield out

    if len(batch_images) > 0:
        yield batch_images, batch_captions


def _iter_hf_webdataset_batches(
    urls,
    batch_size,
    rank=0,
    world_size=1,
    shard_retries=3,
    retry_sleep=2.0,
):
    """Yield batches from remote HF WebDataset shards."""
    batch_images = []
    batch_captions = []

    def _passthrough_nodesplitter(src, group=None):
        """Disable node-level split because sharding is handled manually by rank."""
        yield from src

    if urls is None or len(urls) == 0:
        urls = _default_hf_shard_urls(num_shards=69)

    if wds is None:
        raise RuntimeError("webdataset is required for DDP training, but it is not installed.")

    # In DDP, shard-level partitioning avoids every rank downloading all remote shards.
    urls = [str(u).strip() for u in urls if str(u).strip()]
    if int(world_size) > 1:
        urls = [u for i, u in enumerate(urls) if (i % int(world_size)) == int(rank)]
    if len(urls) == 0:
        return

    for url in urls:
        shard = str(url).strip()
        if not shard:
            continue

        shard_ok = False
        for attempt in range(1, int(shard_retries) + 1):
            try:
                dataset = wds.WebDataset(
                    [shard],
                    shardshuffle=False,
                    nodesplitter=_passthrough_nodesplitter,
                    workersplitter=wds.split_by_worker,
                ).decode("pil")
                for sample in dataset:
                    image = _extract_pil_image_from_sample(sample)
                    caption = _extract_caption_from_sample(sample)
                    if image is None or caption is None:
                        continue

                    batch_images.append(image)
                    batch_captions.append(caption)
                    if len(batch_images) >= int(batch_size):
                        out = (batch_images, batch_captions)
                        batch_images, batch_captions = [], []
                        yield out

                shard_ok = True
                break
            except Exception:
                if attempt < int(shard_retries):
                    time.sleep(float(retry_sleep))

        if not shard_ok:
            continue

    if len(batch_images) > 0:
        yield batch_images, batch_captions


class TrainingDatasetPipeline:
    """Unify HF shard and CC12M TSV dataset handling behind one entry point."""

    def __init__(
        self,
        mode="cc12m",
        sources=None,
        cc12m_cache_dir=None,
        cc12m_cache_images=True,
        cc12m_loader_workers=16,
        cc12m_max_pending=0,
    ):
        self.mode = str(mode).strip().lower() or "auto"
        valid_modes = {"auto", "hf", "cc12m", "all"}
        if self.mode not in valid_modes:
            raise ValueError(f"Unsupported dataset mode {self.mode!r}. Expected one of {sorted(valid_modes)}")

        self.original_sources = [str(x).strip() for x in (sources or []) if str(x).strip()]
        split_sources = _split_train_data_sources(self.original_sources)
        parsed_cc12m_tsv = split_sources["cc12m_tsv"]
        parsed_hf_urls = split_sources["hf_urls"]

        if self.mode == "auto":
            if parsed_cc12m_tsv and parsed_hf_urls:
                selected_kinds = ("cc12m", "hf")
            elif parsed_cc12m_tsv:
                selected_kinds = ("cc12m",)
            else:
                selected_kinds = ("hf",)
        elif self.mode == "all":
            selected_kinds = ("cc12m", "hf")
        elif self.mode == "cc12m":
            selected_kinds = ("cc12m",)
        else:
            selected_kinds = ("hf",)

        self.use_cc12m = "cc12m" in selected_kinds
        self.use_hf = "hf" in selected_kinds

        self.cc12m_tsv = parsed_cc12m_tsv
        if self.use_cc12m and not self.cc12m_tsv:
            self.cc12m_tsv = _default_cc12m_tsv_path()

        self.hf_urls = list(parsed_hf_urls)
        if self.use_hf and len(self.hf_urls) == 0:
            self.hf_urls = _default_hf_shard_urls()

        if self.use_cc12m:
            self.cc12m_tsv = str(self.cc12m_tsv).strip()
            if not self.cc12m_tsv:
                raise ValueError("CC12M dataset mode selected, but no TSV/CSV manifest path was provided.")
            if not _looks_like_cc12m_manifest(self.cc12m_tsv):
                raise ValueError(f"CC12M manifest must end with .tsv or .csv: {self.cc12m_tsv!r}")
            if not os.path.isfile(self.cc12m_tsv):
                raise FileNotFoundError(f"CC12M manifest file not found: {self.cc12m_tsv}")

        self.cc12m_cache_enabled = bool(cc12m_cache_images)
        self.cc12m_cache_dir = None
        if self.use_cc12m and self.cc12m_cache_enabled:
            self.cc12m_cache_dir = _ensure_cc12m_cache_dir(cc12m_cache_dir)

        self.cc12m_loader_workers = max(1, int(cc12m_loader_workers))
        self.cc12m_max_pending = max(0, int(cc12m_max_pending))

    def describe(self):
        """Build a concise human-readable dataset summary for logs."""
        parts = [f"mode={self.mode}"]
        if self.use_cc12m:
            parts.append(f"cc12m={self.cc12m_tsv}")
        if self.use_hf:
            parts.append(f"hf_shards={len(self.hf_urls)}")
        return ", ".join(parts)

    def infer_total_examples(self):
        """Infer total examples when all selected data sources have known sizes."""
        total = 0
        has_any = False

        if self.use_cc12m:
            try:
                total += int(_count_valid_cc12m_rows(self.cc12m_tsv))
            except Exception:
                return None
            has_any = True

        if self.use_hf:
            hf_count = _infer_hf_example_count(self.hf_urls)
            if hf_count is None:
                return None
            total += int(hf_count)
            has_any = True

        return int(total) if has_any else None

    def iter_batches(self, batch_size, rank=0, world_size=1):
        """Yield `(images, captions)` batches from selected dataset sources."""
        if self.use_cc12m:
            yield from _iter_cc12m_tsv_batches_parallel(
                tsv_path=self.cc12m_tsv,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
                cache_dir=self.cc12m_cache_dir,
                cache_images=self.cc12m_cache_enabled,
                loader_workers=self.cc12m_loader_workers,
                max_pending=self.cc12m_max_pending,
            )

        if self.use_hf:
            yield from _iter_hf_webdataset_batches(
                urls=self.hf_urls,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
            )
