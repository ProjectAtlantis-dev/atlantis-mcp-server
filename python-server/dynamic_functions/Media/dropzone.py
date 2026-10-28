"""Shared browser behavior for Atlantis file dropzones."""

import re


_JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")


def dropzone_wiring(zone: str, file_input: str, on_file: str) -> str:
    """Return consistent click, input, and nested-safe drag/drop event wiring.

    Arguments are JavaScript identifier names already declared by the caller.
    Validation and previews intentionally remain operation-specific.
    """
    identifiers = (zone, file_input, on_file)
    if not all(_JS_IDENTIFIER.fullmatch(identifier) for identifier in identifiers):
        raise ValueError("Dropzone wiring arguments must be JavaScript identifiers")

    return f"""
      (() => {{
        let dragDepth = 0;
        const clearDragHighlight = () => {{
          dragDepth = 0;
          {zone}.classList.remove('dragover');
        }};
        {zone}.addEventListener('click', () => {file_input}.click());
        {file_input}.addEventListener('change', event =>
          {on_file}(event.target.files && event.target.files[0]));
        {zone}.addEventListener('dragenter', event => {{
          event.preventDefault();
          event.stopPropagation();
          dragDepth++;
          {zone}.classList.add('dragover');
        }});
        {zone}.addEventListener('dragover', event => {{
          event.preventDefault();
          event.stopPropagation();
          if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
        }});
        {zone}.addEventListener('dragleave', event => {{
          event.preventDefault();
          event.stopPropagation();
          dragDepth--;
          if (dragDepth <= 0) clearDragHighlight();
        }});
        {zone}.addEventListener('drop', event => {{
          event.preventDefault();
          event.stopPropagation();
          clearDragHighlight();
          {on_file}(event.dataTransfer.files && event.dataTransfer.files[0]);
        }});
      }})();
    """
