"""Image, video and audio galleries for the caller's uploads, ported from Waifu.

Two cache layers:
- Disk: one JPEG thumbnail per media version in <collection>/.thumbnails/,
  built once. Videos need it most: their thumbnail is a decoded first frame.
- Browser: window.__atlantisMediaCache holds every thumbnail and original
  already received, keyed by media version. It is page-global, so a later
  gallery only asks the server for what this page has never seen.

The lightbox is for zooming, so it always gets the untouched original.
Audio has no thumbnails; Play loads the original into the same page cache.
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
    "audio": {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
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
    elif collection == "videos":
        await _video_thumbnail(path, thumbnail_path)
    else:
        raise ValueError(f"{collection} has no thumbnails")
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

  // Zoom by columns: 1 per row up to as many as keep each image MIN_IMAGE wide.
  // minmax(0,1fr) columns always fit the panel, so zoom only ever adds rows.
  const MIN_IMAGE = 96, CARD_TARGET = 392, GAP = 8;
  const grid = root.querySelector('.media-gallery-grid');
  const smaller = root.querySelector('.media-gallery-smaller');
  const bigger = root.querySelector('.media-gallery-bigger');
  const colsKey = 'media-gallery-cols-' + (root.querySelector('[data-collection]')?.dataset.collection || '');
  const maxCols = () => Math.max(1, Math.floor((grid.clientWidth + GAP) / (MIN_IMAGE + GAP)));
  // wanted is the user's choice; cols is what fits right now. Keeping them apart
  // means a narrow moment (or a 0-width layout) never overwrites the choice.
  let wanted = 0, cols = 2;
  try { wanted = parseInt(localStorage.getItem(colsKey), 10) || 0; } catch (error) {}
  const layout = () => {
    if (!grid.clientWidth) return;
    if (!wanted) wanted = Math.max(1, Math.floor((grid.clientWidth + GAP) / (CARD_TARGET + GAP)));
    cols = Math.min(wanted, maxCols());
    grid.style.setProperty('--cols', cols);
    smaller.disabled = cols >= maxCols();
    bigger.disabled = cols <= 1;
  };
  const zoom = delta => {
    wanted = Math.min(Math.max(1, cols + delta), maxCols());
    try { localStorage.setItem(colsKey, String(wanted)); } catch (error) {}
    layout();
  };
  smaller.addEventListener('click', () => zoom(1));
  bigger.addEventListener('click', () => zoom(-1));
  new ResizeObserver(layout).observe(grid);
  layout();

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
    request('$Media/gallery_thumbs', {items: missing}).catch(error => {
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
  // The lightbox is the "with sound" view: unmute and start explicitly rather
  // than leaning on the autoplay attribute, resuming where the inline player was.
  let resumeAt = 0;
  large.addEventListener(isVideo ? 'loadeddata' : 'load', () => {
    loading.style.display = 'none';
    large.style.display = 'block';
    if (!isVideo) return;
    if (resumeAt) large.currentTime = resumeAt;
    resumeAt = 0;
    large.muted = false;
    large.play().catch(error => thread.console.softError('[media gallery play] ' + error));
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
      await request('$Media/gallery_open', {collection, name});
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

  // Videos play in place, like a client_video event: muted autoplay with
  // controls, inside the card so the grid never widens. ⤢ opens the lightbox.
  const playInline = async opener => {
    const card = opener.closest('figure');
    const {key, collection, name} = opener.dataset;
    let player = card.querySelector('.media-gallery-inline');
    if (!player) {
      player = document.createElement('video');
      player.className = 'media-gallery-inline';
      player.controls = true;
      player.autoplay = true;
      player.muted = true;
      player.loop = true;
      player.playsInline = true;
      player.style.cssText = 'display:block;width:100%;aspect-ratio:1/1;object-fit:contain;' +
        'background:#000;border-radius:6px';
      card.insertBefore(player, opener);
      const expand = document.createElement('button');
      expand.type = 'button';
      expand.textContent = '⤢';
      expand.title = 'Open large';
      expand.setAttribute('aria-label', 'Open ' + name + ' large');
      expand.style.cssText = 'position:absolute;top:12px;right:12px;z-index:2;width:28px;height:28px;' +
        'padding:0;border:1px solid rgba(255,255,255,.6);border-radius:6px;background:rgba(12,12,30,.75);' +
        'color:white;font-size:16px;line-height:24px;cursor:pointer';
      expand.addEventListener('click', event => {
        event.preventDefault();
        event.stopPropagation();
        player.pause();
        resumeAt = player.currentTime;
        show(opener);
      });
      card.appendChild(expand);
    }
    opener.style.display = 'none';
    player.dataset.mediaFull = key;
    const src = media.full.get(key);
    if (src) {
      player.src = src;
      return;
    }
    try {
      await request('$Media/gallery_open', {collection, name});
    } catch (error) {
      opener.style.display = '';
      player.remove();
      card.querySelector('button[title="Open large"]')?.remove();
      thread.console.softError('[media gallery inline] ' + error);
    }
  };

  openers().forEach(opener => {
    opener.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      if (isVideo) playInline(opener);
      else show(opener);
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


def _gallery_html(gallery_id: str, heading: str, cards: list[str], lightbox: str) -> str:
    # The chat bubble shrink-wraps its content, so percentage widths collapse the
    # grid to its minimum. 100vw asks the bubble for all the room it has, and
    # max-width:100% clamps the panel back to the bubble: the zoom controls then
    # pick how many columns share that width, and the grid grows only downward.
    return f"""
        <div id="{gallery_id}" style="border:2px solid #6c2b97;border-radius:10px;margin:15px 0;
                    background:linear-gradient(135deg,#0c0c1e,#1a0833);
                    padding:15px;width:100vw;max-width:100%;min-width:0;overflow:hidden;
                    box-sizing:border-box">
          <style>
            #{gallery_id} .media-gallery-grid{{
              display:grid;
              grid-template-columns:repeat(var(--cols,2),minmax(0,1fr));
              gap:8px;
              width:100%;
            }}
            #{gallery_id} .media-gallery-card{{min-width:0}}
            #{gallery_id} .media-gallery-zoom button{{width:28px;height:28px;padding:0;border:1px solid #6c2b97;
              border-radius:6px;background:rgba(12,12,30,.72);color:#d9b8f0;font-size:18px;line-height:24px;
              cursor:pointer}}
            #{gallery_id} .media-gallery-zoom button:disabled{{opacity:.35;cursor:default}}
            #{gallery_id} img[data-media-thumb]{{max-width:100%;height:auto}}
            #{gallery_id} img[data-media-thumb]:not([src]){{visibility:hidden}}
          </style>
          <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;
                      gap:8px;margin:0 0 12px">
            <h3 style="color:#9b4dca;margin:0">{heading}</h3>
            <div class="media-gallery-zoom" style="display:flex;align-items:center;gap:6px">
              <button type="button" class="media-gallery-smaller" aria-label="Smaller images" title="Smaller">−</button>
              <button type="button" class="media-gallery-bigger" aria-label="Bigger images" title="Bigger">+</button>
            </div>
          </div>
          <div class="media-gallery-grid">{"".join(cards)}</div>
          {lightbox}
        </div>
        """


def _gallery_script(gallery_id: str) -> str:
    return (
        _GALLERY_SCRIPT.replace("{GALLERY_ID}", gallery_id)
        .replace("{CACHE_JS}", _CACHE_JS)
        .replace("{EXEC_SHELL}", json.dumps(atlantis.get_exec_shell_path()))
    )


def _live_keys(username: str, collection: str) -> list[tuple[Path, str]]:
    items = [(path, _cache_key(path)) for path in _list_media(username, collection)]
    _prune_thumbnails(username, collection, {key for _, key in items})
    return items


@public
async def gallery_images():
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
    # Call atlantis directly from the tool: the client labels output with the caller's name.
    await atlantis.client_html(_gallery_html(
        gallery_id,
        f"🖼️ Image Gallery · {len(cards)} images",
        cards,
        _lightbox_html(gallery_id, "image", viewer),
    ))
    await atlantis.client_script(_gallery_script(gallery_id))


@public
async def gallery_videos():
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
    # Call atlantis directly from the tool: the client labels output with the caller's name.
    await atlantis.client_html(_gallery_html(
        gallery_id,
        f"🎬 Video Gallery · {len(cards)} videos",
        cards,
        _lightbox_html(gallery_id, "video", viewer),
    ))
    await atlantis.client_script(_gallery_script(gallery_id))


_AUDIO_SCRIPT = '''
//js
(() => {
  const root = document.getElementById('{GALLERY_ID}');
  if (!root) return;
  {CACHE_JS}
  // Route requests into this tool's exec shell; see the sendChatter note in Home/modal.py.
  const request = (tool, payload) => sendChatter(window._accessToken, tool, payload, {EXEC_SHELL});
  const clock = seconds => {
    const whole = Math.floor(seconds || 0);
    return Math.floor(whole / 60) + ':' + String(whole % 60).padStart(2, '0');
  };
  const players = () => root.querySelectorAll('.audio-gallery-player');
  root.querySelectorAll('.audio-gallery-item').forEach(item => {
    const {key, name, size} = item.dataset;
    const play = item.querySelector('.audio-gallery-play');
    const player = item.querySelector('.audio-gallery-player');
    const seek = item.querySelector('.audio-gallery-seek');
    const meta = item.querySelector('.audio-gallery-meta');
    const time = item.querySelector('.audio-gallery-time');
    // The cache fills player.src (data-media-full) in every gallery showing this
    // version, so only start playback where Play was actually pressed.
    let wanted = false;
    player.addEventListener('loadedmetadata', () => {
      seek.disabled = false;
      time.textContent = '0:00 / ' + clock(player.duration);
    });
    player.addEventListener('loadeddata', () => {
      meta.textContent = size;
      if (wanted) player.play().catch(() => {});
      wanted = false;
    });
    player.addEventListener('timeupdate', () => {
      if (player.duration) seek.value = player.currentTime / player.duration * 1000;
      time.textContent = clock(player.currentTime) + ' / ' + clock(player.duration);
    });
    player.addEventListener('play', () => {
      players().forEach(other => { if (other !== player) other.pause(); });
      play.textContent = '❚❚';
      play.setAttribute('aria-label', 'Pause ' + name);
      item.classList.add('playing');
    });
    const paused = () => {
      play.textContent = '▶';
      play.setAttribute('aria-label', 'Play ' + name);
      item.classList.remove('playing');
    };
    player.addEventListener('pause', paused);
    player.addEventListener('ended', paused);
    seek.addEventListener('input', () => {
      if (player.duration) player.currentTime = seek.value / 1000 * player.duration;
    });
    play.addEventListener('click', async () => {
      if (player.getAttribute('src')) {
        if (player.paused) player.play().catch(() => {});
        else player.pause();
        return;
      }
      wanted = true;
      const src = media.full.get(key);
      if (src) {
        player.src = src;
        return;
      }
      play.disabled = true;
      meta.textContent = 'Loading…';
      try {
        await request('$Media/gallery_open', {collection: 'audio', name});
      } catch (error) {
        wanted = false;
        thread.console.softError('[media audio open] ' + error);
        meta.textContent = 'Could not load audio.';
      } finally {
        play.disabled = false;
      }
    });
  });
})();
'''


@public
async def gallery_audio():
    """Show the caller's uploaded audio as a playable track list."""
    username = _caller()
    paths = _list_media(username, "audio")
    if not paths:
        await atlantis.client_log("🎙️ No audio found")
        return
    items = []
    for path in paths:
        key = _cache_key(path)
        safe_name = html.escape(path.name, quote=True)
        size = f"{path.stat().st_size / (1024 * 1024):.1f} MB"
        items.append(
            f"""
            <article class="audio-gallery-item" data-key="{key}" data-name="{safe_name}" data-size="{size}">
              <button type="button" class="audio-gallery-play" aria-label="Play {safe_name}">▶</button>
              <div class="audio-gallery-info">
                <strong class="audio-gallery-title" title="{safe_name}">{safe_name}</strong>
                <input type="range" class="audio-gallery-seek" min="0" max="1000" value="0" disabled
                       aria-label="Seek {safe_name}">
                <span class="audio-gallery-meta">{size}</span>
              </div>
              <span class="audio-gallery-time"></span>
              <audio class="audio-gallery-player" data-media-full="{key}" preload="none"></audio>
            </article>
            """
        )
    gallery_id = f"audio_gallery_{uuid.uuid4().hex[:10]}"
    # Call atlantis directly from the tool: the client labels output with the caller's name.
    # Pixel width, not a percentage: the chat bubble shrink-wraps (see _gallery_html).
    await atlantis.client_html(
        f"""
        <div id="{gallery_id}" style="border:2px solid #6c2b97;border-radius:10px;margin:15px 0;
                    background:linear-gradient(135deg,#0c0c1e,#1a0833);padding:15px;
                    width:760px;max-width:100%;min-width:0;box-sizing:border-box;white-space:normal">
          <style>
            #{gallery_id} .audio-gallery-list{{display:flex;flex-direction:column;gap:2px}}
            #{gallery_id} .audio-gallery-item{{display:flex;align-items:center;gap:12px;
              padding:8px 12px;border-radius:6px;transition:background .15s}}
            #{gallery_id} .audio-gallery-item:hover{{background:rgba(108,43,151,.22)}}
            #{gallery_id} .audio-gallery-item.playing{{background:rgba(108,43,151,.32)}}
            #{gallery_id} .audio-gallery-play{{flex:none;width:38px;height:38px;padding:0;
              border:0;border-radius:50%;background:#9b4dca;color:#0c0c1e;font-size:13px;
              line-height:38px;text-align:center;cursor:pointer;transition:transform .1s,background .15s}}
            #{gallery_id} .audio-gallery-play:hover{{background:#b86ee6;transform:scale(1.06)}}
            #{gallery_id} .audio-gallery-play:disabled{{opacity:.45;cursor:wait;transform:none}}
            #{gallery_id} .audio-gallery-info{{flex:1;min-width:0;display:flex;flex-direction:column;gap:3px}}
            #{gallery_id} .audio-gallery-title{{color:#f2e6fa;font-size:14px;font-weight:600;
              overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
            #{gallery_id} .audio-gallery-item.playing .audio-gallery-title{{color:#c98cf0}}
            #{gallery_id} .audio-gallery-seek{{display:none;width:100%;height:4px;margin:2px 0;
              accent-color:#9b4dca;cursor:pointer}}
            #{gallery_id} .audio-gallery-seek:not(:disabled){{display:block}}
            #{gallery_id} .audio-gallery-meta{{color:#8a8aa0;font-size:11px}}
            #{gallery_id} .audio-gallery-time{{flex:none;color:#8a8aa0;font-size:12px;
              font-variant-numeric:tabular-nums}}
          </style>
          <h3 style="color:#9b4dca;margin:0 0 12px">🎙️ Audio Gallery · {len(items)} files</h3>
          <div class="audio-gallery-list">{"".join(items)}</div>
        </div>
        """
    )
    await atlantis.client_script(
        _AUDIO_SCRIPT.replace("{GALLERY_ID}", gallery_id)
        .replace("{CACHE_JS}", _CACHE_JS)
        .replace("{EXEC_SHELL}", json.dumps(atlantis.get_exec_shell_path()))
    )


@public
async def gallery_thumbs(items: list[dict]) -> None:
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
async def gallery_open(collection: str, name: str) -> None:
    """Send one untouched original to the page cache and any lightbox waiting on it."""
    username = _caller()
    path = _media_path(username, collection, name)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    source = f"data:{COLLECTION_MIMES[collection][path.suffix.lower()]};base64,{encoded}"
    logger.info(f"gallery_open sent {path.stat().st_size} bytes of {collection}/{name!r} to {username}")
    await atlantis.client_script(_cache_put_script("full", [(_cache_key(path), source)]))
