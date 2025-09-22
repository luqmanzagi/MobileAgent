
"""
Gemini-port of api.py (multimodal, compat-safe)

- Uses Google Gen AI SDK (google-genai).
- No reliance on a `config` object (some environments don't expose it).
- Supports multimodal content: text + images (local path, http(s) URL, or base64).
- Aggregates any "system" messages and injects them as a priming message.

Install:
    pip install -U google-genai pillow requests
Docs:
    - Explicit API key: https://ai.google.dev/gemini-api/docs/api-key#provide-api-key-explicitly
    - SDK reference: https://googleapis.github.io/python-genai/
"""
import base64
import io
import time
import mimetypes
import os
from pathlib import Path
from typing import List, Tuple, Any, Iterable

import requests
from PIL import Image  # for resize/encode and mime inference

# Gemini SDK
from google import genai
from google.genai import types

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tif', '.tiff'}


def resize_encode_image(image_path: str, screen_scale_ratio: float = 0.5) -> str:
    """Resize an image and return base64-encoded PNG string (unchanged helper)."""
    with Image.open(image_path) as img:
        new_width = max(1, int(img.width * screen_scale_ratio))
        new_height = max(1, int(img.height * screen_scale_ratio))
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        buf = io.BytesIO()
        resized_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# --------------------------- multimodal helpers ---------------------------
def _detect_mime_from_bytes(data: bytes, fallback: str = "application/octet-stream") -> str:
    """Best-effort mime detection via PIL; fall back to generic."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or '').lower()
            if fmt == 'jpeg':
                return 'image/jpeg'
            if fmt:
                return f'image/{fmt}'
    except Exception:
        pass
    return fallback


def _guess_mime_from_path(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def _part_from_image_bytes(data: bytes, mime_type: str) -> types.Part:
    """Create a Gemini image Part, compatible across SDK variants."""
    # Prefer the convenience constructor if available
    try:
        return types.Part.from_bytes(data=data, mime_type=mime_type)  # type: ignore[attr-defined]
    except Exception:
        # Fallback structure
        return types.Part(inline_data=types.Blob(mime_type=mime_type, data=data))


def _bytes_from_base64_data_url(url: str) -> bytes:
    # Expect formats like: data:image/png;base64,AAAA...
    prefix = 'base64,'
    idx = url.find(prefix)
    if idx == -1:
        raise ValueError("Unsupported data URL (missing base64,)")
    b64 = url[idx + len(prefix):]
    return base64.b64decode(b64)


def _is_likely_image_path(text: str) -> bool:
    if not text:
        return False
    p = Path(text)
    return (p.exists() and p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _text_part(text: str) -> types.Part:
    return types.Part.from_text(text=str(text))


def _image_part_from_path(path: str, screen_scale_ratio: float = 1.0) -> types.Part:
    # Optionally resize large images
    try:
        with Image.open(path) as img:
            if screen_scale_ratio and screen_scale_ratio != 1.0:
                new_w = max(1, int(img.width * screen_scale_ratio))
                new_h = max(1, int(img.height * screen_scale_ratio))
                img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            # Preserve original format if possible; default to PNG
            fmt = img.format or 'PNG'
            img.save(buf, format=fmt)
            data = buf.getvalue()
            mime_type = f"image/{fmt.lower() if fmt else 'png'}"
    except Exception:
        # Fallback: raw read + mime guess
        data = Path(path).read_bytes()
        mime_type = _guess_mime_from_path(path) or _detect_mime_from_bytes(data)
    return _part_from_image_bytes(data, mime_type)


def _image_part_from_url(url: str) -> types.Part:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.content
    mime_type = resp.headers.get('Content-Type') or _detect_mime_from_bytes(data)
    return _part_from_image_bytes(data, mime_type)


def _image_part_from_base64(b64: str, mime_hint: str = "image/png") -> types.Part:
    data = base64.b64decode(b64)
    mime_type = _detect_mime_from_bytes(data, fallback=mime_hint)
    return _part_from_image_bytes(data, mime_type)


def _normalize_to_parts(content: Any) -> Iterable[types.Part]:
    """Accepts a variety of content shapes and yields Gemini Parts.

    Supported:
    - str text
    - str path to local image file
    - dict OpenAI-like blocks: {'type':'text','text':...} or {'type':'image_url','image_url':{'url':...}}
    - dict simple image: {'type':'image', 'path':...} or {'image_path':...} or {'base64':...}
    - list of any of the above (flattened)
    """
    # 1) Text or path-like string
    if isinstance(content, str):
        if _is_likely_image_path(content):
            yield _image_part_from_path(content)
        else:
            yield _text_part(content)
        return

    # 2) List/tuple -> flatten
    if isinstance(content, (list, tuple)):
        for item in content:
            yield from _normalize_to_parts(item)
        return

    # 3) Dict variants
    if isinstance(content, dict):
        # OpenAI-like content blocks
        if content.get('type') == 'text' and 'text' in content:
            yield _text_part(str(content['text']))
            return

        if content.get('type') == 'image_url':
            img = content.get('image_url')
            if isinstance(img, dict):
                url = img.get('url') or img.get('path') or img.get('image_path')
            else:
                url = img
            if isinstance(url, str):
                if url.startswith('data:'):
                    yield _part_from_image_bytes(_bytes_from_base64_data_url(url), _detect_mime_from_bytes(_bytes_from_base64_data_url(url)))
                elif url.startswith('http://') or url.startswith('https://'):
                    yield _image_part_from_url(url)
                elif _is_likely_image_path(url):
                    yield _image_part_from_path(url)
                else:
                    # Not a recognized image input; treat as text
                    yield _text_part(str(url))
            return

        # Simple image dicts
        if 'image_path' in content or 'path' in content:
            p = content.get('image_path') or content.get('path')
            if isinstance(p, str) and _is_likely_image_path(p):
                yield _image_part_from_path(p)
                return

        if 'image_base64' in content or 'base64' in content or 'b64' in content:
            b64 = content.get('image_base64') or content.get('base64') or content.get('b64')
            if isinstance(b64, str):
                yield _image_part_from_base64(b64)
                return

        # Fallback: stringify
        yield _text_part(str(content))
        return

    # Unknown type -> stringify
    yield _text_part(str(content))


def _to_gemini_contents(chat: List[Tuple[str, Any]]):
    """Convert (role, content) tuples into Gemini SDK contents + system_prompt.

    - Aggregates any 'system' messages into one priming text block.
    - Converts 'user' -> role='user', 'assistant'/'model' -> role='model'.
    - Content can be text, image paths/urls/base64, or OpenAI-style blocks.
    """
    contents: list = []
    system_parts: list[str] = []

    for role, content in chat:
        if role == "system":
            system_parts.append(str(content))
            continue

        role_map = {"user": "user", "assistant": "model", "model": "model"}
        gem_role = role_map.get(role, "user")
        parts = list(_normalize_to_parts(content))
        if not parts:
            parts = [types.Part.from_text(text="")]
        contents.append(types.Content(role=gem_role, parts=parts))

    # Inject system prompt as a priming user message when config/system_instruction is unavailable
    if system_parts:
        priming = "SYSTEM PROMPT:\n\n" + "\n\n".join(system_parts)
        contents.insert(0, types.Content(role="user", parts=[types.Part.from_text(text=priming)]))

    return contents


# --- snip: imports stay the same ---

def inference_chat(chat: List[Tuple[str, Any]], model: str, api_url: str, token: str) -> Tuple[str, dict]:
    """Call Gemini with a chat history (text + images).

    Returns:
        (text, usage) where usage = {"prompt": int, "completion": int}
    """
    client = genai.Client(api_key=token)  # explicit API key

    contents = _to_gemini_contents(chat)

    last_err = None
    for _ in range(5):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=contents if contents else " ",
            )
            text = getattr(resp, "text", str(resp))

            # --- usage extraction with safe fallbacks ---
            prompt_tokens = 0
            completion_tokens = 0
            try:
                um = getattr(resp, "usage_metadata", None)
                if um:
                    # Gemini typically reports these fields
                    prompt_tokens = int(getattr(um, "prompt_token_count", 0) or 0)
                    completion_tokens = int(getattr(um, "candidates_token_count", 0) or 0)
            except Exception:
                pass

            return text, {"prompt": prompt_tokens, "completion": completion_tokens}
        except Exception as e:
            last_err = e
            time.sleep(1.5)

    raise RuntimeError(f"Gemini call failed after retries: {last_err}")
