"""Image and video galleries for the caller's uploads, ported from Waifu.

Two cache layers:
- Disk: one JPEG thumbnail per media version in <collection>/.thumbnails/,
  built once. Videos need it most: their thumbnail is a decoded first frame.
- Browser: window.__atlantisMediaCache holds every thumbnail and original
  already received, keyed by media version. It is page-global, so a later
  gallery only asks the server for what this page has never seen.

The lightbox is for zooming, so it always gets the untouched original.
"""

import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import atlantis
from PIL import Image, ImageOps

from .storage import media_folder

logger = logging.getLogger("dynamic_function")

COLLECTION_MIMES = {
    "images": {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    },
    "videos": {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".ogv": "video/ogg",
        ".ogg": "video/ogg",
        ".mov": "video/quicktime",
    },
}
THUMBNAIL_SIZE = (384, 384)
THUMBNAIL_QUALITY = 82
THUMBNAIL_FOLDER = ".thumbnails"
# Thumbnails per pushed script, so a big first render streams in.
THUMBNAIL_BATCH = 16

# Creates the page-global cache on first use; every script that touches it
# starts with this so load order never matters.
_CACHE_JS = "const media = window.__atlantisMediaCache ??= {thumb: new Map(), full: new Map()};"


def _caller() -> str:
    username = atlantis.get_caller()
    if not username:
        raise PermissionError("The media gallery needs a caller")
    return username


def _cache_key(path: Path) -> str:
    """Identify one version of a media file; changes whenever the file does."""
    stat = path.stat()
    version = f"{path.parent.name}\0{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}"
    return hashlib.sha256(version.encode("utf-8")).hexdigest()[:24]


def _list_media(username: str, collection: str) -> list[Path]:
    """Oldest first, which leaves the newest item last, as in Waifu."""
    folder = media_folder(username, collection)
    if not folder.is_dir():
        return []
    mimes = COLLECTION_MIMES[collection]
    paths = [
        path for path in folder.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in mimes
    ]
    return sorted(paths, key=lambda path: (path.stat().st_mtime, path.name))


def _media_path(username: str, collection: str, name: str) -> Path:
    """Resolve a browser-supplied collection and name to one of the caller's files."""
    if collection not in COLLECTION_MIMES:
        raise ValueError(f"Invalid media collection: {collection!r}")
    if name != Path(name).name or Path(name).suffix.lower() not in COLLECTION_MIMES[collection]:
        raise ValueError(f"Invalid {collection} name: {name!r}")
    path = media_folder(username, collection) / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Not found in {collection}: {name!r}")
    return path


def _save_jpeg(image: Image.Image, thumbnail_path: Path) -> None:
    """Write pixels only, atomically, so no source metadata rides along."""
    clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    with tempfile.NamedTemporaryFile(
        dir=thumbnail_path.parent, prefix=".creating_", suffix=".jpg", delete=False
    ) as temp:
        temporary_path = Path(temp.name)
    try:
        clean.save(temporary_path, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True, exif=b"")
        os.replace(temporary_path, thumbnail_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _image_thumbnail(path: Path, thumbnail_path: Path) -> None:
    """Contain the image in a dark square, like Waifu's image gallery."""
    with Image.open(path) as source:
        contained = ImageOps.contain(
            ImageOps.exif_transpose(source).convert("RGB"),
            THUMBNAIL_SIZE,
            method=Image.Resampling.LANCZOS,
        )
    thumbnail = Image.new("RGB", THUMBNAIL_SIZE, (12, 12, 30))
    thumbnail.paste(
        contained,
        ((THUMBNAIL_SIZE[0] - contained.width) // 2, (THUMBNAIL_SIZE[1] - contained.height) // 2),
    )
    _save_jpeg(thumbnail, thumbnail_path)


async def _video_thumbnail(path: Path, thumbnail_path: Path) -> None:
    """Square-crop the first decoded frame, like Waifu's video gallery."""
    width, height = THUMBNAIL_SIZE
    with tempfile.NamedTemporaryFile(
        dir=thumbnail_path.parent, prefix=".frame_", suffix=".jpg", delete=False
    ) as temp:
        frame_path = Path(temp.name)
    try:
        process = await asyncio.create_subprocess_exec(
            "ffmpeg", "-v", "error", "-y",
            "-i", str(path),
            "-map", "0:v:0", "-map_metadata", "-1", "-map_chapters", "-1",
            "-frames:v", "1",
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}",
            "-q:v", "4",
            str(frame_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0 or frame_path.stat().st_size == 0:
            detail = stderr.decode(errors="replace").strip()[-500:]
            raise RuntimeError(f"ffmpeg could not create a thumbnail for {path.name}: {detail}")
        # ffmpeg still writes its encoder name as a JPEG comment; re-save pixels only.
        with Image.open(frame_path) as frame:
            frame.load()
            await asyncio.to_thread(_save_jpeg, frame, thumbnail_path)
    finally:
        frame_path.unlink(missing_ok=True)


async def _ensure_thumbnail(path: Path, collection: str, key: str) -> Path:
    """Return the cached thumbnail for this media version, building it once."""
    thumbnail_path = path.parent / THUMBNAIL_FOLDER / f"{key}.jpg"
    if thumbnail_path.is_file():
        return thumbnail_path
    thumbnail_path.parent.mkdir(exist_ok=True)
    if collection == "images":
        await asyncio.to_thread(_image_thumbnail, path, thumbnail_path)
    else:
        await _video_thumbnail(path, thumbnail_path)
    return thumbnail_path


def _prune_thumbnails(username: str, collection: str, live_keys: set[str]) -> None:
    """Drop thumbnails for media that was removed or changed."""
    folder = media_folder(username, collection) / THUMBNAIL_FOLDER
    if not folder.is_dir():
        return
    for thumbnail in folder.iterdir():
        if thumbnail.suffix == ".jpg" and thumbnail.stem not in live_keys:
            thumbnail.unlink()


def _cache_put_script(kind: str, entries: list[tuple[str, str]]) -> str:
    """Store media in the page cache and fill every element waiting on it.

    Originals become object URLs: a video needs one to seek, and the cache
    then holds a small URL instead of a huge base64 string.
    """
    convert = kind == "full"
    return f"""
    //js
    (async () => {{
      {_CACHE_JS}
      for (const [key, data] of {json.dumps(entries)}) {{
        const src = {"URL.createObjectURL(await (await fetch(data)).blob())" if convert else "data"};
        media.{kind}.set(key, src);
        document.querySelectorAll('[data-media-{kind}="' + key + '"]').forEach(el => {{ el.src = src; }});
      }}
    }})();
    """


# Shared by both galleries. Openers are the elements carrying data-name: the
# thumbnail <img> for images, the play <button> for videos.
_GALLERY_SCRIPT = '''
//js
(() => {
  const root = document.getElementById('{GALLERY_ID}');
  if (!root) return;
  {CACHE_JS}
  const lightbox = root.querySelector('.media-gallery-lightbox');
  const large = lightbox.querySelector('.media-gallery-large');
  const loading = lightbox.querySelector('.media-gallery-loading');
  const closeButton = lightbox.querySelector('.media-gallery-close');
  const previousButton = lightbox.querySelector('.media-gallery-previous');
  const nextButton = lightbox.querySelector('.media-gallery-next');
  const isVideo = large.tagName === 'VIDEO';
  const loadingText = isVideo ? 'Loading video…' : 'Loading image…';
  // Escape the chat message's stacking and clipping contexts.
  document.body.appendChild(lightbox);

  // Route requests into this tool's exec shell; see the sendChatter note in Home/modal.py.
  const request = (tool, payload) => sendChatter(window._accessToken, tool, payload, {EXEC_SHELL});

  const missing = [];
  root.querySelectorAll('img[data-media-thumb]').forEach(img => {
    const src = media.thumb.get(img.dataset.mediaThumb);
    if (src) {
      img.src = src;
      return;
    }
    const item = img.closest('[data-name]');
    missing.push({collection: item.dataset.collection, name: item.dataset.name});
  });
  if (missing.length) {
    request('$Media/media_gallery_thumbs', {items: missing}).catch(error => {
      thread.console.softError('[media gallery thumbnails] ' + error);
    });
  }

  let current = null;
  const openers = () => Array.from(root.querySelectorAll('[data-name]'));
  const unload = () => {
    if (isVideo) large.pause();
    large.dataset.mediaFull = '';
    large.removeAttribute('src');
    if (isVideo) large.load();
    large.style.display = 'none';
  };
  const close = () => {
    lightbox.style.display = 'none';
    unload();
  };
  large.addEventListener(isVideo ? 'loadeddata' : 'load', () => {
    loading.style.display = 'none';
    large.style.display = 'block';
  });
  const show = async opener => {
    const list = openers();
    const index = list.indexOf(opener);
    if (index < 0) return;
    current = opener;
    unload();
    loading.textContent = loadingText;
    loading.style.display = 'block';
    previousButton.style.display = index > 0 ? 'block' : 'none';
    nextButton.style.display = index < list.length - 1 ? 'block' : 'none';
    lightbox.style.display = 'flex';
    const {key, collection, name} = opener.dataset;
    large.dataset.mediaFull = key;
    const src = media.full.get(key);
    if (src) {
      large.src = src;
      return;
    }
    try {
      await request('$Media/media_gallery_open', {collection, name});
    } catch (error) {
      loading.textContent = isVideo ? 'Could not load video' : 'Could not load image';
      thread.console.softError('[media gallery open] ' + error);
    }
  };
  const move = offset => {
    const list = openers();
    const index = list.indexOf(current);
    const target = index + offset;
    if (index >= 0 && target >= 0 && target < list.length) show(list[target]);
  };

  openers().forEach(opener => {
    opener.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      show(opener);
    });
  });
  closeButton.addEventListener('click', close);
  previousButton.addEventListener('click', event => {
    event.stopPropagation();
    move(-1);
  });
  nextButton.addEventListener('click', event => {
    event.stopPropagation();
    move(1);
  });
  if (isVideo) large.addEventListener('click', event => event.stopPropagation());
  lightbox.addEventListener('click', close);
  document.addEventListener('keydown', event => {
    if (lightbox.style.display === 'flex' && event.key === 'ArrowLeft') {
      event.preventDefault();
      move(-1);
      return;
    }
    if (lightbox.style.display === 'flex' && event.key === 'ArrowRight') {
      event.preventDefault();
      move(1);
      return;
    }
    if (event.key === 'Escape' && lightbox.style.display === 'flex') close();
  });
})();
'''


def _lightbox_html(gallery_id: str, noun: str, viewer: str) -> str:
    """Waifu's lightbox chrome: close, previous/next, loading text, viewer."""
    return f"""
          <div id="{gallery_id}_lightbox" class="media-gallery-lightbox"
               style="display:none;position:fixed;inset:0;z-index:2147483647;
                      width:100vw;height:100vh;height:100dvh;background:rgba(0,0,0,.9);
                      align-items:center;justify-content:center;padding:8px;
                      box-sizing:border-box;cursor:zoom-out;user-select:none;-webkit-user-select:none;
                      -webkit-touch-callout:none">
            <button class="media-gallery-close" aria-label="Close" style="position:absolute;z-index:3;
                    top:14px;right:20px;width:46px;height:46px;padding:0;border:0;border-radius:50%;
                    background:rgba(0,0,0,.72);color:white;font-size:38px;line-height:42px;
                    cursor:pointer">×</button>
            <button class="media-gallery-previous" aria-label="Previous {noun}"
                    style="display:none;position:absolute;z-index:3;left:14px;top:50%;transform:translateY(-50%);
                           width:50px;height:64px;padding:0;border:0;border-radius:8px;
                           background:rgba(0,0,0,.62);color:white;font-size:48px;line-height:56px;
                           cursor:pointer">‹</button>
            <button class="media-gallery-next" aria-label="Next {noun}"
                    style="display:none;position:absolute;z-index:3;right:14px;top:50%;transform:translateY(-50%);
                           width:50px;height:64px;padding:0;border:0;border-radius:8px;
                           background:rgba(0,0,0,.62);color:white;font-size:48px;line-height:56px;
                           cursor:pointer">›</button>
            {viewer}
          </div>
    """


async def _render_gallery(gallery_id: str, container_style: str, heading: str, cards: list[str], lightbox: str) -> None:
    await atlantis.client_html(
        f"""
        <div id="{gallery_id}" style="border:2px solid #6c2b97;border-radius:10px;margin:15px 0;
                    background:linear-gradient(135deg,#0c0c1e,#1a0833);
                    width:min(calc(75vw - 16px),calc(100vw - 48px));
                    max-width:min(calc(75vw - 16px),calc(100vw - 48px));
                    min-width:0;box-sizing:border-box;{container_style}">
          <style>
            #{gallery_id} .media-gallery-grid{{
              display:grid;
              grid-template-columns:repeat(auto-fill,minmax(min(128px,100%),1fr));
              gap:8px;
              width:100%;
            }}
            #{gallery_id} .media-gallery-card{{min-width:0}}
            #{gallery_id} img[data-media-thumb]:not([src]){{visibility:hidden}}
          </style>
          <h3 style="color:#9b4dca;margin:0 0 12px">{heading}</h3>
          <div class="media-gallery-grid">{"".join(cards)}</div>
          {lightbox}
        </div>
        """
    )
    await atlantis.client_script(
        _GALLERY_SCRIPT.replace("{GALLERY_ID}", gallery_id)
        .replace("{CACHE_JS}", _CACHE_JS)
        .replace("{EXEC_SHELL}", json.dumps(atlantis.get_exec_shell_path()))
    )


def _live_keys(username: str, collection: str) -> list[tuple[Path, str]]:
    items = [(path, _cache_key(path)) for path in _list_media(username, collection)]
    _prune_thumbnails(username, collection, {key for _, key in items})
    return items


@public
async def media_gallery_images():
    """Show the caller's uploaded images in a grid."""
    username = _caller()
    items = _live_keys(username, "images")
    if not items:
        await atlantis.client_log("🖼️ No images found")
        return
    cards = []
    for path, key in items:
        safe_name = html.escape(path.name, quote=True)
        cards.append(
            f"""
            <figure class="media-gallery-card"
                    style="position:relative;display:flex;flex-direction:column;margin:0;
                           background:rgba(12,12,30,.72);padding:8px;border-radius:8px">
              <img class="media-gallery-thumb" data-media-thumb="{key}" data-key="{key}"
                   data-collection="images" data-name="{safe_name}" alt="{safe_name}"
                   width="{THUMBNAIL_SIZE[0]}" height="{THUMBNAIL_SIZE[1]}"
                   style="width:100%;aspect-ratio:1/1;object-fit:contain;border-radius:6px;
                          display:block;cursor:zoom-in">
            </figure>
            """
        )
    gallery_id = f"image_gallery_{uuid.uuid4().hex[:10]}"
    viewer = """
            <figure style="position:relative;margin:0;width:100%;height:100%;text-align:center;
                           display:flex;align-items:center;justify-content:center">
              <div class="media-gallery-loading" style="color:white;font-size:16px">Loading image…</div>
              <img class="media-gallery-large" alt="Zoomed gallery image" draggable="false"
                   style="display:none;width:100%;height:100%;object-fit:contain;
                          border-radius:8px;box-shadow:0 0 30px rgba(0,0,0,.8);cursor:zoom-out">
            </figure>
    """
    await _render_gallery(
        gallery_id,
        "padding:15px 24px 96px 15px;overflow:visible",
        f"🖼️ Image Gallery · {len(cards)} images",
        cards,
        _lightbox_html(gallery_id, "image", viewer),
    )


@public
async def media_gallery_videos():
    """Show the caller's uploaded videos in a grid."""
    username = _caller()
    items = _live_keys(username, "videos")
    if not items:
        await atlantis.client_log("🎬 No videos found")
        return
    cards = []
    for path, key in items:
        safe_name = html.escape(path.name, quote=True)
        size_mb = path.stat().st_size / (1024 * 1024)
        cards.append(
            f"""
            <figure class="media-gallery-card"
                    style="position:relative;margin:0;background:rgba(12,12,30,.72);padding:8px;border-radius:8px">
              <button type="button" class="media-gallery-open" data-key="{key}"
                      data-collection="videos" data-name="{safe_name}"
                      aria-label="Play {safe_name}" title="{safe_name} · {size_mb:.1f} MB"
                      style="position:relative;display:block;width:100%;padding:0;border:0;
                             border-radius:6px;overflow:hidden;background:#090914;cursor:pointer">
                <img data-media-thumb="{key}" alt="First frame of {safe_name}"
                     width="{THUMBNAIL_SIZE[0]}" height="{THUMBNAIL_SIZE[1]}"
                     style="width:100%;aspect-ratio:1/1;object-fit:cover;display:block">
                <span aria-hidden="true"
                      style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                             width:36px;height:36px;border-radius:50%;display:grid;place-items:center;
                             padding-left:3px;background:rgba(12,12,30,.75);border:1px solid rgba(255,255,255,.75);
                             color:white;font-size:17px;box-shadow:0 2px 10px rgba(0,0,0,.65)">▶</span>
              </button>
            </figure>
            """
        )
    gallery_id = f"video_gallery_{uuid.uuid4().hex[:10]}"
    viewer = """
            <div class="media-gallery-loading" style="position:absolute;color:#ddd;font-size:15px">
              Loading video…
            </div>
            <video class="media-gallery-large" controls autoplay loop playsinline preload="auto"
                   draggable="false"
                   style="display:none;width:100%;height:100%;
                          object-fit:contain;background:#000;border-radius:8px;
                          box-shadow:0 0 30px rgba(0,0,0,.8);cursor:default"></video>
    """
    await _render_gallery(
        gallery_id,
        "padding:15px 24px 15px 15px;overflow:hidden",
        f"🎬 Video Gallery · {len(cards)} videos",
        cards,
        _lightbox_html(gallery_id, "video", viewer),
    )


@public
async def media_gallery_thumbs(items: list[dict]) -> None:
    """Send thumbnails the page has not cached yet, building any that are missing."""
    username = _caller()
    for start in range(0, len(items), THUMBNAIL_BATCH):
        entries = []
        for item in items[start:start + THUMBNAIL_BATCH]:
            path = _media_path(username, item["collection"], item["name"])
            key = _cache_key(path)
            thumbnail = await _ensure_thumbnail(path, item["collection"], key)
            encoded = base64.b64encode(thumbnail.read_bytes()).decode("ascii")
            entries.append((key, f"data:image/jpeg;base64,{encoded}"))
        await atlantis.client_script(_cache_put_script("thumb", entries))


@public
async def media_gallery_open(collection: str, name: str) -> None:
    """Send one untouched original to the page cache and any lightbox waiting on it."""
    username = _caller()
    path = _media_path(username, collection, name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    source = f"data:{COLLECTION_MIMES[collection][path.suffix.lower()]};base64,{encoded}"
    logger.info(f"media_gallery_open sent {path.stat().st_size} bytes of {collection}/{name!r} to {username}")
    await atlantis.client_script(_cache_put_script("full", [(_cache_key(path), source)]))
