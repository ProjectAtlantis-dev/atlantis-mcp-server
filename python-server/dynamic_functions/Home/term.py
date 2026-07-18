"""Terminal display tools"""

import atlantis
import os

@public
async def term_bg_video(url: str) -> None:
    """Play a video in the terminal background."""
    await atlantis.set_background_video(
        url,
        vertical_align="center",
        loop=False,
        muted=False,
        autoplay=True,
        plays_inline=True,
        remove_on_ended=True,
        toggle_audio=True,
    )


@public
async def term_bg_player(
    url: str,
    muted: bool = False,
    autoplay: bool = True,
    controls: bool = False,
    loop: bool = False,
    remove_on_ended: bool = True,
    frame: bool = False,
    interactive: bool = False,
) -> None:
    """Play a YouTube/player URL in the terminal background.

    frame=True embeds the URL as a generic iframe instead of video media
    (e.g. a locally-served WebGL/WebGPU renderer on http://localhost:5173).
    interactive=True lets a frame receive pointer events."""
    url = str(url or "").strip()
    if not url:
        raise ValueError("url required")
    if not url.startswith(("http://", "https://")):
        raise ValueError("player url must be http(s); use term_bg_video_file for local files")
    await atlantis.set_background_player(
        url,
        vertical_align="center",
        loop=loop,
        muted=muted,
        autoplay=autoplay,
        plays_inline=True,
        remove_on_ended=remove_on_ended,
        controls=controls,
        frame=frame,
        interactive=interactive,
    )


def _local_video_path(video_path: str) -> str:
    """Resolve and validate a local video path."""
    path = str(video_path or "").strip()
    if not path:
        raise ValueError("video_path required")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"Video file not found: {video_path}")
    return path


@public
async def term_bg_video_file(video_path: str) -> None:
    """Play a local video file in the terminal background."""
    await term_bg_video(_local_video_path(video_path))



@public
async def term_glass() -> None:
    """Apply frosted styling to terminal feedback bubbles."""
    await atlantis.client_terminal_script("""
(function(){
    var fb = document.getElementById('feedback');
    if (!fb) return;

    var styleId = 'frostStyle';
    document.querySelectorAll('#' + styleId).forEach(function(existing){ existing.remove(); });
    {
      var s = document.createElement('style');
      s.id = styleId;
      s.textContent =
        '#feedback.frosted .chatbox-receiver{' +
        ' background-image:linear-gradient(to top, rgba(24,24,44,0.24), rgba(10,10,18,0.18)) !important;' +
        ' background-color:rgba(12,14,24,0.10) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' border:1px solid rgba(255,255,255,0.12) !important;' +
        ' box-shadow:0 4px 14px rgba(0,0,0,0.24) !important;' +
        '}' +

        '#feedback.frosted .chatbox-sender{' +
        ' background-image:none !important;' +
        ' background-color:rgba(12,18,28,0.16) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' border:1px solid rgba(255,255,255,0.10) !important;' +
        ' box-shadow:0 4px 14px rgba(0,0,0,0.22) !important;' +
        '}' +

        '#feedback.frosted .bot-table-header{' +
        ' background-color:rgba(8,12,32,0.24) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        '}' +

        '#feedback.frosted .bot-table-cell{' +
        ' background-color:rgba(7,7,9,0.16) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        '}' +

        '#feedback.frosted .bot-table-row-hover:hover .bot-table-cell{' +
        ' background-color:rgba(40,50,80,0.28) !important;' +
        '}' +

        '#feedback.frosted .bot-table-cell.bot-table-cell-selected{' +
        ' background-color:rgba(45,74,107,0.40) !important;' +
        '}' +

        '#feedback.frosted .catCodeBlock{' +
        ' background-image:linear-gradient(to top, rgba(24,24,44,0.20), rgba(10,10,18,0.12)) !important;' +
        ' background-color:rgba(7,7,12,0.14) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' border:1px solid rgba(255,255,255,0.13) !important;' +
        ' box-shadow:0 4px 14px rgba(0,0,0,0.24) !important;' +
        '}' +

        '#feedback.frosted .catCode,' +
        '#feedback.frosted .catCode span{' +
        ' background-color:transparent !important;' +
        '}' +

        '#feedback.frosted .catHiddenAbove,' +
        '#feedback.frosted .catHiddenBelow{' +
        ' background-color:rgba(7,7,12,0.08) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        '}' +

        'body.terminal-frosted #botTableStickyHeaderShim{' +
        ' background-color:transparent !important;' +
        '}' +

        'body.terminal-frosted #botTableStickyHeaderShim .bot-table-header{' +
        ' background-color:rgba(8,12,32,0.82) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        '}' +

        'body.terminal-frosted .monaco-editor{' +
        ' background-color:rgba(8,10,18,0.34) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' --vscode-focusBorder:rgba(255,255,255,0.22) !important;' +
        ' --focus-border:rgba(255,255,255,0.22) !important;' +
        '}' +

        'body.terminal-frosted .monaco-editor *{' +
        ' --vscode-focusBorder:rgba(255,255,255,0.22) !important;' +
        ' --focus-border:rgba(255,255,255,0.22) !important;' +
        '}' +

        'body.terminal-frosted .monaco-editor.focused,' +
        'body.terminal-frosted .monaco-editor .focused,' +
        'body.terminal-frosted .monaco-editor :focus,' +
        'body.terminal-frosted .monaco-editor :focus-visible{' +
        ' outline-color:rgba(255,255,255,0.22) !important;' +
        ' box-shadow:none !important;' +
        '}' +

        'body.terminal-frosted .monaco-editor .editor-scrollable,' +
        'body.terminal-frosted .monaco-editor .margin,' +
        'body.terminal-frosted .monaco-editor .glyph-margin,' +
        'body.terminal-frosted .monaco-editor .margin-view-overlays,' +
        'body.terminal-frosted .monaco-editor .margin-view-overlays > div,' +
        'body.terminal-frosted .monaco-editor .margin-view-overlays .line-numbers,' +
        'body.terminal-frosted .monaco-editor .margin-view-overlays .current-line,' +
        'body.terminal-frosted .monaco-editor .margin-view-overlays .cldr,' +
        'body.terminal-frosted .monaco-editor .lines-content.monaco-editor-background,' +
        'body.terminal-frosted .monaco-editor .view-lines{' +
        ' background-color:transparent !important;' +
        '}' +

        'body.terminal-frosted .monaco-editor .lines-content.monaco-editor-background{' +
        ' background-image:linear-gradient(to top, rgba(24,24,44,0.30), rgba(10,10,18,0.20)) !important;' +
        '}' +

        'body.terminal-frosted .jsPanel,' +
        'body.terminal-frosted .jsPanel .jsPanel-content,' +
        'body.terminal-frosted .jsPanel-content{' +
        ' background:transparent !important;' +
        ' background-color:transparent !important;' +
        '}' +

        'body.terminal-frosted .jsPanel .panel-buttonbar{' +
        ' background:rgba(12,18,28,0.16) !important;' +
        ' background-color:rgba(12,18,28,0.16) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        '}' +

        'body.terminal-frosted [id^="displayname-"],' +
        'body.terminal-frosted [id^="modalmenu-"]{' +
        ' background-image:linear-gradient(to bottom, rgba(20,34,48,0.30), rgba(20,50,60,0.22)) !important;' +
        ' background-color:rgba(12,18,28,0.18) !important;' +
        ' -webkit-backdrop-filter:blur(8px) saturate(118%);' +
        ' backdrop-filter:blur(8px) saturate(118%);' +
        ' border-color:rgba(255,255,255,0.16) !important;' +
        ' box-shadow:0 12px 34px rgba(0,0,0,0.30) !important;' +
        '}' +

        'body.terminal-frosted [id^="displayname-"] input,' +
        'body.terminal-frosted [id^="modalmenu-"] .menu-choice{' +
        ' background:rgba(7,15,22,0.24) !important;' +
        ' background-color:rgba(7,15,22,0.24) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' border-color:rgba(255,255,255,0.15) !important;' +
        '}' +

        'body.terminal-frosted [id^="displayname-"] input:focus{' +
        ' outline-color:rgba(255,255,255,0.28) !important;' +
        '}' +

        'body.terminal-frosted [id^="displayname-"] button{' +
        ' background:rgba(20,120,120,0.34) !important;' +
        ' background-color:rgba(20,120,120,0.34) !important;' +
        ' -webkit-backdrop-filter:blur(6px) saturate(112%);' +
        ' backdrop-filter:blur(6px) saturate(112%);' +
        ' border:1px solid rgba(255,255,255,0.16) !important;' +
        '}' +

        'body.terminal-frosted [id^="displayname-"] button:hover,' +
        'body.terminal-frosted [id^="modalmenu-"] .menu-choice:hover,' +
        'body.terminal-frosted [id^="modalmenu-"] .menu-choice:focus{' +
        ' background:rgba(20,255,208,0.16) !important;' +
        ' background-color:rgba(20,255,208,0.16) !important;' +
        ' border-color:rgba(255,255,255,0.24) !important;' +
        '}';

      document.head.appendChild(s);
    }

    if (window.terminalFrostBorderTimer) {
      clearTimeout(window.terminalFrostBorderTimer);
      window.terminalFrostBorderTimer = null;
    }

    fb.classList.add('frosted');
    document.body.classList.add('terminal-frosted');
  })();
""")

@public
async def term_default() -> None:
    """Restore default terminal feedback bubble styling."""
    await atlantis.client_terminal_script("""
(function(){
  var fb = document.getElementById('feedback');
  if (window.terminalFrostBorderTimer) clearTimeout(window.terminalFrostBorderTimer);
  if (fb) fb.classList.remove('frosted');
  document.body.classList.remove('terminal-frosted');
  document.querySelectorAll('#frostStyle').forEach(function(s){ s.remove(); });

  // Repair any frost styling left behind by older script versions that wrote
  // directly to chat bubbles instead of relying only on the stylesheet above.
  // Do not touch table elements here: their grid lines and optional color
  // overrides are inline styles created by TableHelper.
  if (fb) {
    fb.querySelectorAll('.chatbox-receiver, .chatbox-sender').forEach(function(el){
      el.style.removeProperty('background-image');
      el.style.removeProperty('background-color');
      el.style.removeProperty('-webkit-backdrop-filter');
      el.style.removeProperty('backdrop-filter');
      el.style.removeProperty('box-shadow');
      el.style.removeProperty('border');
    });
  }
})();
""")

@visible
async def term_show() -> str:
    """Print the atlantis terminal key for this call (session key + shell)."""
    key = atlantis.get_terminal_key()
    if not key:
        raise RuntimeError("No terminal key in this call context")
    return key
