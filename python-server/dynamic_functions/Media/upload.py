"""Drag-and-drop uploader that stores images and videos under Media/users/."""

import json
import logging
import uuid

import atlantis

from .dropzone import dropzone_wiring
from .storage import MAX_UPLOAD_BYTES, MEDIA_TYPES, save_upload

logger = logging.getLogger("dynamic_function")


@public
async def upload():
    """Open an uploader for images and videos."""
    upload_id = f"media_upload_{uuid.uuid4().hex[:10]}"
    accept = ",".join(MEDIA_TYPES)
    await atlantis.client_html(
        f"""
        <div id="{upload_id}" style="border:2px solid #1a8a78;border-radius:10px;margin:15px 0;
                    background:rgba(7,15,22,0.58);padding:15px;width:100%;max-width:400px;box-sizing:border-box">
          <style>
            #{upload_id} .media-upload-button{{display:flex;align-items:center;justify-content:center;
              width:min(100%,360px);min-height:240px;margin:0 auto;padding:28px;
              position:relative;overflow:hidden;box-sizing:border-box;
              border:2px dashed #1a8a78;border-radius:8px;background:rgba(20,255,208,.06);
              color:#f7f4ea;cursor:pointer;font-size:14px;text-align:center;
              transition:border-color .15s,background .15s,box-shadow .15s,transform .15s}}
            #{upload_id} .media-upload-button:hover,
            #{upload_id} .media-upload-button.dragover{{border-color:#14ffd0;
              background:rgba(20,255,208,.16);box-shadow:0 0 18px rgba(20,255,208,.25)}}
            #{upload_id} .media-upload-button.dragover{{transform:scale(1.01)}}
            #{upload_id} .media-upload-preview{{display:none;width:100%;height:auto;
              border-radius:6px;pointer-events:none;background:#071016;
              user-select:none;-webkit-user-select:none}}
            #{upload_id} .media-upload-button.has-preview{{padding:0;min-height:0}}
            #{upload_id} .media-upload-button.has-preview .media-upload-preview.active{{display:block}}
            #{upload_id} .media-upload-button.has-preview .media-upload-label{{display:none}}
            #{upload_id} .media-upload-button.uploading .media-upload-label{{display:block;
              position:absolute;z-index:2;padding:10px 14px;border-radius:7px;
              background:rgba(0,0,0,.76);color:white}}
          </style>
          <h3 style="color:#14ffd0;margin:0 0 12px">⬆️ Upload</h3>
          <input class="media-upload-input" type="file" accept="{accept}" style="display:none">
          <button type="button" class="media-upload-button">
            <span class="media-upload-label">Drop an image or video here, or click to choose</span>
            <img class="media-upload-preview media-upload-image-preview"
                 alt="Uploaded image preview" draggable="false">
            <video class="media-upload-preview media-upload-video-preview"
                   aria-label="Uploaded video preview" muted playsinline preload="metadata"
                   draggable="false"></video>
          </button>
          <p class="media-upload-status" style="color:#aaa;font-size:12px;margin:10px 0 0"></p>
        </div>
        """
    )

    # Route the save call into this tool's exec shell so it stays out of the
    # user's history; see the sendChatter note in Home/modal.py.
    exec_shell_js = json.dumps(atlantis.get_exec_shell_path())
    await atlantis.client_script(
        '''
        //js
        (() => {
          const root = document.getElementById('{UPLOAD_ID}');
          if (!root) return;
          const input = root.querySelector('.media-upload-input');
          const button = root.querySelector('.media-upload-button');
          const label = root.querySelector('.media-upload-label');
          const imagePreview = root.querySelector('.media-upload-image-preview');
          const videoPreview = root.querySelector('.media-upload-video-preview');
          const status = root.querySelector('.media-upload-status');
          const accepted = new Set({ACCEPTED});
          const maxBytes = {MAX_BYTES};
          let videoPreviewUrl = '';
          function clearPreview() {
            imagePreview.classList.remove('active');
            imagePreview.removeAttribute('src');
            videoPreview.classList.remove('active');
            videoPreview.pause();
            videoPreview.removeAttribute('src');
            videoPreview.load();
            if (videoPreviewUrl) URL.revokeObjectURL(videoPreviewUrl);
            videoPreviewUrl = '';
          }
          function showPreview(file, content) {
            clearPreview();
            if (file.type.startsWith('image/')) {
              imagePreview.src = content;
              imagePreview.alt = file.name;
              imagePreview.classList.add('active');
            } else {
              videoPreviewUrl = URL.createObjectURL(file);
              videoPreview.src = videoPreviewUrl;
              videoPreview.setAttribute('aria-label', file.name);
              videoPreview.classList.add('active');
              videoPreview.addEventListener('loadeddata', () => videoPreview.pause(), {once: true});
              videoPreview.load();
            }
            button.classList.add('has-preview');
            button.title = 'Click or drop another file to upload';
          }
          async function sendFile(file) {
            if (!file || !accepted.has(file.type)) {
              status.textContent = 'Choose a PNG, JPEG, WebP, MP4, WebM, Ogg, or MOV file.';
              return;
            }
            if (file.size > maxBytes) {
              status.textContent = 'File exceeds ' + (maxBytes / 1024 / 1024) + ' MB.';
              return;
            }
            button.disabled = true;
            button.classList.add('uploading');
            label.textContent = 'Uploading ' + file.name + '…';
            status.textContent = '';
            try {
              const content = await new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = event => resolve(event.target.result);
                reader.onerror = () => reject(new Error('Could not read file'));
                reader.readAsDataURL(file);
              });
              await sendChatter(window._accessToken, '$Media/upload_save', {
                base64Content: content,
                filename: file.name,
                filetype: file.type
              }, {EXEC_SHELL});
              status.textContent = 'Uploaded ' + file.name + '.';
              showPreview(file, content);
            } catch (error) {
              console.error('[media upload] ' + error);
              status.textContent = 'Upload failed: ' + error;
              label.textContent = 'Try again';
            } finally {
              button.classList.remove('uploading');
              button.disabled = false;
              input.value = '';
            }
          }

          {DROPZONE_WIRING}
        })();
        '''.replace("{UPLOAD_ID}", upload_id)
        .replace("{ACCEPTED}", json.dumps(list(MEDIA_TYPES)))
        .replace("{MAX_BYTES}", str(MAX_UPLOAD_BYTES))
        .replace("{EXEC_SHELL}", exec_shell_js)
        .replace("{DROPZONE_WIRING}", dropzone_wiring("button", "input", "sendFile"))
    )


@public
async def upload_save(base64Content: str, filename: str, filetype: str) -> str:
    """Store one browser upload in the caller's image or video folder."""
    username = atlantis.get_caller()
    if not username:
        raise PermissionError("upload_save needs a caller")
    saved_path, created = save_upload(username, base64Content, filename, filetype)
    logger.info(f"upload_save {'saved' if created else 'reused'} {saved_path} for {username}")
    return str(saved_path)
